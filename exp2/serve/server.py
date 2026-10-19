"""Exp2 family server (slice C1): one process per model family, every configured early exit resident.

    python -m exp2.serve.server --family F --exits 5,23 --adapter real|fake [--fake-spec P] --out DIR
        [--preprocess per_request|preload] [--n-inputs 32] [--warmup-iters 20] [--no-checkpoints]
        [--sync event|spin] [--seed 0]                                                       exit 0, 2

Start-up, in this order (K5/K6/K7, BRIEF_v1 C1):
    1. TORCHINDUCTOR_* env, count the files already in the inductor cache      (prepare_inductor_env)
    2. families.setup_compile(...)  -> "on" | "eager_fallback" | "off", reported in ready.versions.compile
    3. adapter.load(exits, cfg) + make_inputs, then warm EVERY exit up to a stable CV   (ExitServer.warm_up)
    4. bind, write <out>/server_<family>.port, accept ONE connection, send `ready`
    5. single-threaded loop: run -> done, ping -> pong, stop -> exit 0, EOF -> exit 0

Truncation (K7), one definition in exp2.contracts. For a `run`:
    begin; for each checkpoint: advance, adapter.sync(), read the clock, stop after the first checkpoint that
    finished later than the deadline (its segment still ran on the GPU); contracts.truncation_outcome() turns the
    times into (reached, outcome); finish(state, reached) runs the head AFTER the decision. The head is measured
    into t_end_ns but never charged to the deadline. miss -> completion 0. `--no-checkpoints` -> one checkpoint
    (the target) and one sync, so the outcome is complete or miss.

Errors never kill the loop silently: an adapter exception or an unservable request comes back as a `done` with
outcome "error" and `err`. A message with an unknown op, or an unparseable one, stops the server loudly (the
dispatcher then sees EOF at once instead of waiting for a reply that never comes).

torch, transformers and cv2 are never imported here; only the adapters (exp2.serve.fam_*) load them.
"""
import argparse
import os
import statistics
import sys
import time

from exp2 import contracts, wire
from exp2.serve import families

CV_WINDOW = 5                 # warm-up stability is judged on the last CV_WINDOW inferences
CV_MAX = 0.05                 # stop warming once their coefficient of variation is below this
MIN_WARMUP_ITERS = CV_WINDOW
WARMUP_MAX_FACTOR = 5         # hard cap: at most WARMUP_MAX_FACTOR * --warmup-iters inferences per exit
REAL_INPUT_MODE = "dataset"   # make_inputs mode for --adapter real (flip to "synthetic" for an offline board)
FAKE_INPUT_MODE = "synthetic"
PREPROCESS_MODES = ("per_request", "preload")


def _log(msg):
    print("[exp2.server] %s" % msg, file=sys.stderr, flush=True)


def _die(msg):
    print("exp2.serve.server: error: %s" % msg, file=sys.stderr, flush=True)
    raise SystemExit(2)


def cv(samples):
    """Sample stdev / mean of `samples`; 0.0 when undefined (fewer than 2 samples, or a mean of 0)."""
    if len(samples) < 2:
        return 0.0
    mean = statistics.fmean(samples)
    if mean <= 0:
        return 0.0
    return statistics.stdev(samples) / mean


def prepare_inductor_env():
    """Set the inductor cache env before anything loads; return how many files the cache already holds.

    TORCHINDUCTOR_CACHE_DIR keeps an explicit value, else ~/.cache/exp2_inductor. FX_GRAPH_CACHE=1 and
    COMPILE_THREADS=1 are forced (parallel inductor workers fail to pickle on Tegra). The file count lands in
    ready.warmup.inductor_cache_files_at_start, so a cold and a warm cache are told apart afterwards.
    """
    cache_dir = os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR",
                                      os.path.join(os.path.expanduser("~"), ".cache", "exp2_inductor"))
    os.environ["TORCHINDUCTOR_FX_GRAPH_CACHE"] = "1"
    os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = "1"
    return sum(len(files) for _, _, files in os.walk(cache_dir))


def versions_info(compile_state):
    """ready.versions. torch/transformers/cuda come from modules that are ALREADY loaded: None when they are not."""
    torch = sys.modules.get("torch")
    transformers = sys.modules.get("transformers")
    return {"python": "%d.%d.%d" % tuple(sys.version_info[:3]),
            "torch": getattr(torch, "__version__", None),
            "transformers": getattr(transformers, "__version__", None),
            "cuda": getattr(getattr(torch, "version", None), "cuda", None),
            "compile": compile_state}


class ExitServer:
    """The request loop for one family. The adapter is K6; `clock` returns int ns like time.monotonic_ns."""

    def __init__(self, adapter, family, exits, *, use_checkpoints=True, n_inputs=32, clock=time.monotonic_ns):
        if getattr(adapter, "family", None) != family:
            raise ValueError("adapter family %r != server family %r" % (getattr(adapter, "family", None), family))
        exits = list(exits)
        if not exits or len(set(exits)) != len(exits):
            raise ValueError("exits must be a non-empty list without duplicates, got %r" % (exits,))
        for e in exits:
            contracts.type_id(family, e)                       # range check, ContractError (a ValueError)
        if not isinstance(n_inputs, int) or n_inputs < 1:
            raise ValueError("n_inputs must be a positive int, got %r" % (n_inputs,))
        self.adapter, self.family, self.exits = adapter, family, exits
        self.use_checkpoints, self.n_inputs, self._clock = use_checkpoints, n_inputs, clock
        self._ckpts = {}

    # ------------------------------------------------------------------------------------------ start-up
    def load(self, cfg, seed, input_mode):
        """adapter.load + make_inputs; check the adapter's checkpoints against contracts. Returns load seconds."""
        t0 = self._clock()
        self.adapter.load(self.exits, cfg)
        self.adapter.make_inputs(self.n_inputs, seed, input_mode)
        for e in self.exits:
            want = contracts.default_checkpoints(self.family, e)
            got = list(self.adapter.checkpoints(e))
            if got != want:
                raise ValueError("adapter checkpoints for %s are %s, contracts.default_checkpoints says %s"
                                 % (contracts.type_id(self.family, e), got, want))
            self._ckpts[e] = got if self.use_checkpoints else [got[-1]]
        return (self._clock() - t0) / 1e9

    def warm_up(self, min_iters):
        """Warm every exit through the serving path (checkpoints, sync, head), one exit after the other.

        Per exit: one cold inference (first_infer_s), then reset_peak and more inferences until there are at
        least `min_iters` and the CV of the last CV_WINDOW is below CV_MAX, or until the cap. act_mb is the peak
        allocation of those warm inferences above the allocation right after reset_peak. A CV that never settles
        is reported in cv_last5, not hidden.
        """
        if not isinstance(min_iters, int) or min_iters < 1:
            raise ValueError("min_iters must be a positive int, got %r" % (min_iters,))
        cap = min_iters * WARMUP_MAX_FACTOR
        out = {"first_infer_s": {}, "n_warm": {}, "cv_last5": {}, "act_mb": {}}
        for e in self.exits:
            durs = []
            self._timed_inference(e, 0, durs)
            self.adapter.reset_peak()
            base = self.adapter.mem_stats()["torch_peak_alloc_mb"]
            while len(durs) < cap:
                if len(durs) >= min_iters and cv(durs[-CV_WINDOW:]) < CV_MAX:
                    break
                self._timed_inference(e, len(durs) % self.n_inputs, durs)
            peak = self.adapter.mem_stats()["torch_peak_alloc_mb"]
            key = str(e)
            out["first_infer_s"][key] = durs[0]
            out["n_warm"][key] = len(durs)
            out["cv_last5"][key] = cv(durs[-CV_WINDOW:])
            out["act_mb"][key] = max(0.0, peak - base)
        return out

    def _timed_inference(self, exit, input_idx, durs):
        t0 = self._clock()
        self._execute(exit, input_idx, None, {})
        durs.append((self._clock() - t0) / 1e9)

    def ready_message(self, *, load_s, warm, compile_state, cache_files, input_source):
        """The K5 `ready` message, checked with contracts.validate_ready before it can be sent."""
        mem = self.adapter.mem_stats()
        msg = {"op": "ready", "family": self.family, "pid": os.getpid(), "exits_warmed": list(self.exits),
               "warmup": dict({"load_s": load_s}, **warm, inductor_cache_files_at_start=cache_files),
               "input_source": input_source, "versions": versions_info(compile_state),
               "mem": {"torch_reserved_mb": mem["torch_reserved_mb"], "rss_mb": mem["rss_mb"]},
               "t_ready_ns": self._clock()}
        contracts.validate_ready(msg)
        return msg

    # ------------------------------------------------------------------------------------------ one request (K7)
    def _execute(self, exit, input_idx, deadline_ns, rec):
        """Run one request. Fills rec["t_start"], rec["ckpt"] as it goes (so a failure keeps the partial trace).

        Returns (reached, outcome). The head for `reached` runs here, after the decision; t_end is set last.
        """
        adapter, ckpts = self.adapter, self._ckpts[exit]
        rec["ckpt"] = pairs = []
        rec["t_start"] = self._clock()
        state = adapter.begin(input_idx, exit)
        t_ckpt = []
        for c in ckpts:
            adapter.advance(state, c)
            adapter.sync()
            t = self._clock()
            t_ckpt.append(t)
            pairs.append([c, t])
            if deadline_ns is not None and t > deadline_ns:
                break                                            # this segment overran: it ran, nothing after it does
        reached, outcome = contracts.truncation_outcome(ckpts, t_ckpt, deadline_ns)
        adapter.finish(state, reached)
        rec["t_end"] = self._clock()
        return reached, outcome

    def handle_run(self, msg):
        """A K5 `done` for a `run` message. Never raises for a bad request or a failing adapter: outcome "error"."""
        done = {"op": "done", "req_id": msg.get("req_id"), "type_id": msg.get("type_id"), "t_recv_ns": self._clock(),
                "t_start_ns": None, "t_end_ns": None, "reached": None, "target": None, "completion": 0.0,
                "outcome": "error", "ckpt": [], "tokens_done": None, "err": None}
        rec = {}
        try:
            family, exit = contracts.parse_type_id(msg.get("type_id"))
            # target first: a message with a bad field but a parseable type_id still gets a valid `done`
            done["target"] = contracts.default_checkpoints(family, exit)[-1]
            contracts.validate_run(msg)
            if family != self.family:
                raise ValueError("family %r is not this server's family %r" % (family, self.family))
            if exit not in self._ckpts:
                raise ValueError("exit %d is not warmed (warmed exits: %s)" % (exit, self.exits))
            if msg["input_idx"] >= self.n_inputs:
                raise ValueError("input_idx %d outside 0..%d" % (msg["input_idx"], self.n_inputs - 1))
            reached, outcome = self._execute(exit, msg["input_idx"], msg["deadline_ns"], rec)
        except Exception as e:                                   # reported to the dispatcher, the loop goes on
            err = "%s: %s" % (type(e).__name__, e)
            _log("request %s failed: %s" % (msg.get("req_id"), err))
            started = rec.get("t_start")
            done.update(t_start_ns=started, t_end_ns=None if started is None else self._clock(),
                        ckpt=rec.get("ckpt", []), err=err)
            return done
        done.update(t_start_ns=rec["t_start"], t_end_ns=rec["t_end"], reached=reached, outcome=outcome,
                    completion=contracts.completion(family, exit, reached), ckpt=rec["ckpt"],
                    tokens_done=reached if family == "llama" else None)
        return done

    # ------------------------------------------------------------------------------------------ loop
    def serve(self, conn):
        """Answer messages one at a time until stop ("stop") or until the dispatcher goes away ("disconnect")."""
        while True:
            try:
                msg = wire.recv(conn)
            except (EOFError, OSError) as e:
                _log("dispatcher connection closed (%s); exiting" % type(e).__name__)
                return "disconnect"
            op = msg.get("op")
            if op == "stop":
                return "stop"
            if op == "ping":
                reply = {"op": "pong", "t_ns": self._clock()}
            elif op == "run":
                reply = self.handle_run(msg)
            else:
                raise ValueError("unknown op %r in message %r" % (op, msg))
            try:
                wire.send(conn, reply)
            except OSError as e:
                _log("could not reply (%s); exiting" % type(e).__name__)
                return "disconnect"


# ---------------------------------------------------------------------------------------------- process level
def parse_args(argv=None):
    p = argparse.ArgumentParser(prog="python -m exp2.serve.server", allow_abbrev=False,
                                description="Exp2 family server: warm every exit, then serve run/ping/stop (K5).")
    p.add_argument("--family", required=True, choices=contracts.FAMILIES)
    p.add_argument("--exits", required=True, help="comma separated exit indices, e.g. 5,23")
    p.add_argument("--adapter", choices=("real", "fake"), default="real")
    p.add_argument("--fake-spec", help="exp2.fake_spec/1 JSON (only with --adapter fake)")
    p.add_argument("--out", required=True, help="directory for server_<family>.port")
    p.add_argument("--preprocess", choices=PREPROCESS_MODES, default="per_request")
    p.add_argument("--n-inputs", type=int, default=32)
    p.add_argument("--warmup-iters", type=int, default=20, help="minimum warm inferences per exit (>= %d)"
                   % MIN_WARMUP_ITERS)
    p.add_argument("--no-checkpoints", action="store_true", help="one checkpoint (the target) and one sync")
    p.add_argument("--sync", choices=families.SYNC_MODES, default="event")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    try:
        args.exits = [int(x) for x in args.exits.split(",")]
    except ValueError:
        p.error("--exits must be comma separated integers, got %r" % args.exits)
    if len(set(args.exits)) != len(args.exits):
        p.error("--exits has duplicates: %s" % args.exits)
    for e in args.exits:
        try:
            contracts.type_id(args.family, e)
        except contracts.ContractError as err:
            p.error("--exits: %s" % err)
    if args.n_inputs < 1:
        p.error("--n-inputs must be >= 1")
    if args.warmup_iters < MIN_WARMUP_ITERS:
        p.error("--warmup-iters must be >= %d (the CV window)" % MIN_WARMUP_ITERS)
    if args.adapter == "fake" and not args.fake_spec:
        p.error("--adapter fake needs --fake-spec")
    if args.adapter == "real" and args.fake_spec:
        p.error("--fake-spec only goes with --adapter fake")
    return args


def build_adapter(args):
    """The family adapter: families.get_adapter for real, a FakeAdapter built from --fake-spec for fake."""
    if args.adapter == "real":
        return families.get_adapter(args.family)
    from exp2.serve import fake_adapter

    try:
        adapter = fake_adapter.FakeAdapter.from_file(args.fake_spec)
    except (OSError, ValueError) as e:                           # unreadable file, bad JSON, bad spec, virtual clock
        _die("--fake-spec %s: %s" % (args.fake_spec, e))
    if adapter.family != args.family:
        _die("--fake-spec is for family %r, not %r" % (adapter.family, args.family))
    missing = [e for e in args.exits if str(e) not in adapter.spec["types"]]
    if missing:
        _die("--fake-spec has no entry for exit(s) %s" % missing)
    return adapter


def run(args, adapter, compile_state, cache_files, clock=time.monotonic_ns):
    """Load, warm, publish the port, accept the dispatcher, send ready, serve. Returns the exit code."""
    os.makedirs(args.out, exist_ok=True)
    cfg = {"preprocess": args.preprocess, "n_inputs": args.n_inputs, "seed": args.seed, "sync": args.sync,
           "compile": compile_state == "on",
           "input_source": REAL_INPUT_MODE if args.adapter == "real" else FAKE_INPUT_MODE}
    if compile_state == "eager_fallback":
        _log("WARNING: torch.compile unavailable, running EAGER (reported in ready.versions.compile)")
    srv = ExitServer(adapter, args.family, args.exits, use_checkpoints=not args.no_checkpoints,
                     n_inputs=args.n_inputs, clock=clock)
    load_s = srv.load(cfg, args.seed, cfg["input_source"])
    warm = srv.warm_up(args.warmup_iters)
    ready = srv.ready_message(load_s=load_s, warm=warm, compile_state=compile_state, cache_files=cache_files,
                              input_source=args.preprocess)
    listener = wire.listen()
    try:
        wire.write_port_file(os.path.join(args.out, "server_%s.port" % args.family), wire.listener_port(listener))
        conn = listener.accept()
        try:
            wire.send(conn, ready)
            _log("ready (%s exits %s), serving" % (args.family, args.exits))
            reason = srv.serve(conn)
        finally:
            conn.close()
    finally:
        listener.close()
    _log("stopped (%s)" % reason)
    return 0


def main(argv=None):
    args = parse_args(argv)
    cache_files = prepare_inductor_env()
    compile_state = families.setup_compile(args.adapter == "real")
    adapter = build_adapter(args)
    return run(args, adapter, compile_state, cache_files)


if __name__ == "__main__":
    sys.exit(main())
