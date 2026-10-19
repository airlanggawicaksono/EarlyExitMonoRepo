"""Exp2 session runner (slice C2b; contracts K9, K15, K16, K17).

    python -m exp2.serve.run_session --plan PLAN.json --types TYPES.json --out SESSION_DIR
        [--adapter real|fake] [--fake-spec-dir DIR] [--dry-run --fake-root DIR] [--ready-timeout 600]

Order (nothing is spawned before step 1 passes):
    1. validate the plan; for an open plan verify the freeze file and plan.freeze.content_sha256     exit 2 / 3
    2. start the root helper (dry with --dry-run), power mode, snapshot, jetson_clocks --store, telemetry
    3. idle window "pre"
    4. start the family servers ONE AT A TIME (spawn, wait for the port file, connect, read `ready`)   exit 5
    5. idle window "resident"
    6. every plan condition: apply_arm, settle, run, manifest.json  (a failing condition is marked valid=false and
       the next one still runs)
    7. stop servers and telemetry (stdin EOF), restore jetson_clocks, write session.json

Layout (K9): <out>/session.json, <out>/telemetry/, <out>/servers/ (port files, logs), <out>/<condition_id>/.
--adapter fake needs <fake-spec-dir>/<family>.json (exp2.fake_spec/1) for every family of the plan.
Exit codes: 0 ok, 2 usage / invalid plan or types, 3 freeze refused, 4 board (ClockError), 5 server not ready,
6 finished but at least one condition is valid=false.
"""
import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np

from exp2 import arrivals, contracts, wire
from exp2.board.clocks import Board
from exp2.board.sysfs import RootHelper
from exp2.contracts import ClockError, ContractError, FreezeError
from exp2.policies import Policy
from exp2.serve.dispatcher import Dispatcher

REPO_ROOT = Path(__file__).resolve().parents[2]
EXIT_OK, EXIT_USAGE, EXIT_FREEZE, EXIT_BOARD, EXIT_SERVER, EXIT_INVALID = 0, 2, 3, 4, 5, 6
STOP_WAIT_S = 15.0
RECV_READY_S = 30.0


class ServerNotReady(RuntimeError):
    pass


def _log(msg):
    print("[exp2.run_session] %s" % msg, file=sys.stderr, flush=True)


def _dump(path, obj):
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(obj, fh, indent=2)
        fh.write("\n")


def load_plan(path):
    """Parse and validate the plan; for kind 'open' verify the freeze. -> (plan, freeze | None).
    ContractError: bad plan. FreezeError: freeze file or hash refused."""
    with open(path, encoding="utf-8") as fh:
        plan = contracts.validate_plan(json.load(fh))
    if plan["kind"] != "open":
        return plan, None
    fpath = Path(plan["freeze"]["path"])
    if not fpath.is_absolute():
        fpath = Path(path).resolve().parent / fpath
    freeze = contracts.verify_freeze_file(fpath)
    if freeze["content_sha256"] != plan["freeze"]["content_sha256"]:
        raise FreezeError("plan.freeze.content_sha256 %s != freeze file content_sha256 %s"
                          % (plan["freeze"]["content_sha256"], freeze["content_sha256"]))
    return plan, freeze


def git_info():
    def git(*args):
        return subprocess.run(["git", *args], cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=30)
    try:
        commit = git("rev-parse", "HEAD").stdout.strip() or "unknown"
        dirty = bool(git("status", "--porcelain").stdout.strip())
        diff = hashlib.sha256(git("diff", "HEAD").stdout.encode("utf-8")).hexdigest() if dirty else None
    except (OSError, subprocess.SubprocessError):
        return {"commit": "unknown", "dirty": True, "diff_sha256": None}
    return {"commit": commit, "dirty": dirty, "diff_sha256": diff}


class Session:
    def __init__(self, args, plan, freeze, types):
        self.args, self.plan, self.freeze, self.types = args, plan, freeze, types
        self.sess = plan["session"]
        self.out = Path(args.out)
        self.board = None
        self.telemetry = None
        self.procs = {}                  # family -> Popen
        self.conns = {}                  # family -> Connection
        self.servers = {}                # family -> session.json entry
        self.idle = {}
        self.cond_ids = []
        self.n_invalid = 0
        self.jc_store = None

    # ---------------------------------------------------------------------------------------- board, telemetry
    def start_telemetry(self):
        tdir = self.out / "telemetry"
        cmd = [sys.executable, "-m", "exp2.board.telemetry", "--out", str(tdir)]
        if self.args.dry_run:
            cmd += ["--sysfs-root", str(self.args.fake_root), "--no-tegrastats"]
        self.telemetry = subprocess.Popen(cmd, stdin=subprocess.PIPE, cwd=str(REPO_ROOT))

    def stop_telemetry(self):
        proc = self.telemetry
        if proc is None:
            return
        try:
            proc.stdin.close()                                   # EOF on stdin stops the sampler
            proc.wait(timeout=STOP_WAIT_S)
        except (OSError, subprocess.TimeoutExpired):
            proc.kill()
            proc.wait()
        self.telemetry = None

    def f_idle(self):
        levels = [c["gpu_mhz"] for c in self.plan["conditions"] if c["arm"] == "pinned"]
        return max(levels) if levels else None

    def idle_window(self, duration_s):
        """Pin the highest plan level (governor if the plan pins nothing), settle, then time a window."""
        f = self.f_idle()
        back = self.board.apply_arm("pinned" if f is not None else "governor", f, self.sess["emc_mhz"])
        time.sleep(self.sess["settle_s"])
        t0 = time.monotonic_ns()
        time.sleep(duration_s)
        return {"t0_ns": t0, "t1_ns": time.monotonic_ns(), "gpu_mhz": back["gpu_mhz"]}

    # ---------------------------------------------------------------------------------------- servers
    def start_servers(self):
        """One at a time: spawn, wait for the port file, connect, read ready. t_ready(i) <= t_spawn(i+1)."""
        sdir = self.out / "servers"
        sdir.mkdir(parents=True, exist_ok=True)
        for fam in self.sess["families"]:
            cmd = [sys.executable, "-m", "exp2.serve.server", "--family", fam,
                   "--exits", ",".join(str(e) for e in self.sess["exits"][fam]), "--adapter", self.args.adapter,
                   "--out", str(sdir), "--preprocess", self.sess["preprocess"],
                   "--n-inputs", str(self.sess["n_inputs"])]
            if self.args.adapter == "fake":
                cmd += ["--fake-spec", str(Path(self.args.fake_spec_dir) / ("%s.json" % fam))]
            port_file = sdir / ("server_%s.port" % fam)
            if port_file.exists():
                port_file.unlink()
            t_spawn_wall, t_spawn = time.time_ns(), time.monotonic_ns()
            with open(sdir / ("server_%s.log" % fam), "wb") as log:
                proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                        cwd=str(REPO_ROOT))
            self.procs[fam] = proc
            try:
                port = wire.wait_port_file(port_file, self.args.ready_timeout, proc)
                conn = wire.connect(port)
                self.conns[fam] = conn
                ready = contracts.validate_ready(wire.recv(conn, timeout_s=RECV_READY_S))
            except (TimeoutError, RuntimeError, EOFError, OSError, ContractError) as exc:
                raise ServerNotReady("server %s not ready: %s: %s" % (fam, type(exc).__name__, exc)) from exc
            self.servers[fam] = {"pid": proc.pid, "t_spawn_ns": t_spawn, "t_ready_ns": time.monotonic_ns(),
                                 "t_spawn_wall_ns": t_spawn_wall, "ready": ready}
            _log("server %s ready after %.1f s" % (fam, (self.servers[fam]["t_ready_ns"] - t_spawn) / 1e9))

    def stop_servers(self):
        for fam, conn in self.conns.items():
            try:
                wire.send(conn, {"op": "stop"})
            except (OSError, ValueError):
                pass
        for fam, proc in self.procs.items():
            try:
                proc.wait(timeout=STOP_WAIT_S if fam in self.conns else 0.1)
            except subprocess.TimeoutExpired:
                proc.kill()                                      # accept() has no timeout: never leave it behind
                proc.wait()
        for conn in self.conns.values():
            try:
                conn.close()
            except OSError:
                pass
        self.conns, self.procs = {}, {}

    # ---------------------------------------------------------------------------------------- conditions
    def run_condition(self, disp, cond):
        cid = cond["condition_id"]
        cdir = self.out / cid
        cdir.mkdir(parents=True, exist_ok=True)
        t_wall0, t_mono0 = time.time_ns(), time.monotonic_ns()
        valid, reason, readback, summary = True, None, None, None
        try:
            back = self.board.apply_arm(cond["arm"], cond["gpu_mhz"], self.sess["emc_mhz"])
            readback = {k: back[k] for k in ("gpu_mhz", "emc_mhz", "governor_free")}
            time.sleep(self.sess["settle_s"])
            if cond["mode"] == "idle":
                time.sleep(cond["duration_s"])
            elif cond["mode"] == "closed":
                summary = disp.run_closed(cond, cdir)
            else:
                sched = arrivals.schedule(cond["workload"], cond["n_requests"], cond["seed"])
                summary = disp.run_open(cond, sched, Policy(cond["policy"], self.types), cdir)
            if summary is not None and summary["dead_servers"]:
                valid, reason = False, "server died: %s" % json.dumps(summary["dead_servers"], sort_keys=True)
            elif summary is not None and summary["freq"]["errors"]:
                valid, reason = False, "board fault: %s" % json.dumps(summary["freq"]["errors"][0])
        except Exception as exc:                                 # ClockError, ValueError, dispatcher failure
            traceback.print_exc()
            valid, reason = False, "%s: %s" % (type(exc).__name__, exc)
        t_mono1 = time.monotonic_ns()
        if not valid:
            self.n_invalid += 1
            _log("condition %s INVALID: %s" % (cid, reason))
        emc = readback["emc_mhz"] if readback else self.sess["emc_mhz"]
        _dump(cdir / "manifest.json", {
            "schema": contracts.SCHEMAS["manifest"], "condition_id": cid, "mode": cond["mode"], "arm": cond["arm"],
            "gpu_mhz": cond["gpu_mhz"], "emc_mhz": emc, "t_wall_start_ns": t_wall0, "t_mono_start_ns": t_mono0,
            "t_mono_end_ns": t_mono1, "settle_s": self.sess["settle_s"], "valid": valid, "invalid_reason": reason,
            "board_readback": readback,
            "n_requests": summary["n_requests"] if summary else None,
            "n_warmup": cond.get("n_warmup"), "seed": cond.get("seed"), "policy_key": cond.get("policy_key"),
            "policy_spec": cond.get("policy"), "freeze_condition_id": cond.get("freeze_condition_id"),
            "workload": cond.get("workload"), "alpha": cond.get("alpha"), "cell": cond.get("cell"),
            "split": cond.get("split"), "repeat": cond.get("repeat")})
        self.cond_ids.append(cid)

    # ---------------------------------------------------------------------------------------- main flow
    def run(self):
        """Returns the exit code. Servers and telemetry are always stopped; jetson_clocks restored if stored."""
        self.out.mkdir(parents=True, exist_ok=True)
        t_wall_start, t_mono_start = time.time_ns(), time.monotonic_ns()
        code, snap_start, snap_end = EXIT_OK, None, None
        try:
            if self.args.dry_run:
                helper = RootHelper.start(mode="dry", fake_root=self.args.fake_root)
            else:
                helper = RootHelper.start(mode="sudo")
            self.board = Board(helper)
            self.board.ensure_power_mode()
            snap_start = self.board.snapshot()
            self.jc_store = str(self.out / "jetson_clocks.store")
            self.board.store_jetson_clocks(self.jc_store)
            self.start_telemetry()
            self.idle["pre"] = self.idle_window(self.sess["idle_window_s"])
            self.start_servers()
            self.idle["resident"] = self.idle_window(self.sess["idle_window_s"])
            disp = Dispatcher(self.conns, self.types, board=self.board, n_inputs=self.sess["n_inputs"])
            for cond in self.plan["conditions"]:
                self.run_condition(disp, cond)
            code = EXIT_INVALID if self.n_invalid else EXIT_OK
        except ClockError as exc:
            _log("board error: %s" % exc)
            code = EXIT_BOARD
        except ServerNotReady as exc:
            _log(str(exc))
            code = EXIT_SERVER
        finally:
            self.stop_servers()
            self.stop_telemetry()
            if self.board is not None:
                try:
                    if self.jc_store is not None:
                        self.board.restore_jetson_clocks(self.jc_store)
                    snap_end = self.board.snapshot()
                except ClockError as exc:
                    _log("restore failed: %s" % exc)
                    code = EXIT_BOARD
                self.board.close()
        if "resident" in self.idle:                              # the session got as far as running conditions
            self.write_session(code, t_wall_start, t_mono_start, snap_start, snap_end)
        return code

    def write_session(self, code, t_wall_start, t_mono_start, snap_start, snap_end):
        _dump(self.out / "session.json", {
            "schema": contracts.SCHEMAS["session"],
            "session_id": "s" + time.strftime("%Y%m%dT%H%M%S", time.gmtime(t_wall_start // 10 ** 9)),
            "plan_sha256": self.plan["sha256"], "plan_kind": self.plan["kind"],
            "freeze_content_sha256": self.freeze["content_sha256"] if self.freeze else None,
            "t_wall_start_ns": t_wall_start, "t_mono_start_ns": t_mono_start, "t_wall_end_ns": time.time_ns(),
            "git": git_info(),
            "host": {"python": platform.python_version(), "numpy": np.__version__,
                     "platform": "%s-%s" % (sys.platform, platform.machine()),
                     "boot_id": str((snap_start or {}).get("boot_id", "unknown")).strip()},
            "board": {"dry": bool(self.args.dry_run), "snapshot_start": snap_start or {}, "snapshot_end": snap_end or {},
                      "jetson_clocks_store": self.jc_store},
            "idle": self.idle, "servers": self.servers, "conditions": self.cond_ids, "exit_code": code})


def build_parser():
    p = argparse.ArgumentParser(prog="python -m exp2.serve.run_session", description=__doc__.split("\n", 1)[0])
    p.add_argument("--plan", required=True)
    p.add_argument("--types", required=True, help="exp2.types/1 file (the dispatcher and the policies need it)")
    p.add_argument("--out", required=True)
    p.add_argument("--adapter", choices=("real", "fake"), default="real")
    p.add_argument("--fake-spec-dir", default=None, help="folder with <family>.json for --adapter fake")
    p.add_argument("--dry-run", action="store_true", help="fake sysfs tree (needs --fake-root), no telemetry tegrastats")
    p.add_argument("--fake-root", default=None)
    p.add_argument("--ready-timeout", type=float, default=600.0, help="seconds to wait for each server")
    return p


def main(argv=None):
    p = build_parser()
    try:
        args = p.parse_args(argv)
    except SystemExit as exc:
        return int(exc.code or 0)
    if args.dry_run != (args.fake_root is not None):
        _log("error: --dry-run and --fake-root go together")
        return EXIT_USAGE
    if args.adapter == "fake" and not args.fake_spec_dir:
        _log("error: --adapter fake needs --fake-spec-dir")
        return EXIT_USAGE
    try:
        plan, freeze = load_plan(args.plan)
    except FreezeError as exc:
        _log("freeze refused: %s" % exc)
        return EXIT_FREEZE
    except (ContractError, OSError, ValueError) as exc:
        _log("invalid plan: %s: %s" % (type(exc).__name__, exc))
        return EXIT_USAGE
    try:
        types = contracts.load_types(args.types)
    except (ContractError, OSError, ValueError) as exc:
        _log("invalid types file: %s: %s" % (type(exc).__name__, exc))
        return EXIT_USAGE
    return Session(args, plan, freeze, types).run()


if __name__ == "__main__":
    sys.exit(main())
