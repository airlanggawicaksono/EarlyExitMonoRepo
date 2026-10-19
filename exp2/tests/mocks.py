"""Mocks and builders shared by the exp2 tests (BRIEF_v1 sec. 3.5). Standard library plus exp2.contracts only.

    FakeClock, FakeConn.pair()     time and the wire, so no test sleeps or opens a socket
    FakeBoard                      a BoardControl that records calls, injects ClockError, lets the read-back differ
    StubSlowdown, UnitSlowdown     SlowdownModel stand-ins:  s = n / c(n)  and  s = 1
    StubPower                      PowerModelP stand-in (K13 formula)
    make_types, make_simresult     small TypeTable / SimResult builders
    StubSim                        a sim_fn driven by a table; records calls; builds best-response maps and 2-cycles
    load_fixture                   JSON fixtures under exp2/tests/fixtures
    write_session and the synthetic_* builders live further down in this file.

Nothing here reads a real clock, the network, a GPU or the Jetson.
"""
import collections
import csv
import json
import statistics
import threading
from pathlib import Path

from exp2 import contracts as C

FIXTURES = Path(__file__).resolve().parent / "fixtures"
WALL0_NS = 1_792_491_300_000_000_000      # 2026-10-20T10:15:00Z, the wall time of the synthetic Run-1 session
MONO0_NS = 5_000_000_000


def fixture_path(rel):
    return FIXTURES / rel


def load_fixture(rel):
    """Parse a JSON file under exp2/tests/fixtures (rel uses forward slashes)."""
    with open(fixture_path(rel), encoding="utf-8") as fh:
        return json.load(fh)


# ===================================================================================================== clock
class FakeClock:
    """Callable returning int nanoseconds, like time.monotonic_ns. Time moves only when a test moves it."""

    def __init__(self, start_ns=0, wall_offset_ns=WALL0_NS):
        self._now = int(start_ns)
        self.wall_offset_ns = int(wall_offset_ns)
        self._lock = threading.Lock()

    def __call__(self):
        with self._lock:
            return self._now

    def advance_ns(self, ns):
        ns = int(ns)
        if ns < 0:
            raise ValueError("a clock cannot go backwards: advance_ns(%d)" % ns)
        with self._lock:
            self._now += ns
            return self._now

    def advance_s(self, seconds):
        return self.advance_ns(round(seconds * 1e9))

    def sleep(self, seconds):
        """Drop-in for time.sleep: moves the clock, does not wait."""
        self.advance_s(seconds)

    def set_ns(self, ns):
        with self._lock:
            if ns < self._now:
                raise ValueError("a clock cannot go backwards: set_ns(%d) < %d" % (ns, self._now))
            self._now = int(ns)

    def monotonic(self):
        return self() / 1e9

    def wall_ns(self):
        return self.wall_offset_ns + self()


# ===================================================================================================== wire
class FakeConn:
    """In-memory stand-in for a multiprocessing Connection: send_bytes, recv_bytes, poll, close.

    FakeConn.pair() returns two connected ends. Without a clock it behaves like a real pipe (blocking, thread safe).
    With clock=FakeClock it is a single-threaded discrete-event channel: a message sent at t is delivered at
    t + latency_ns, poll(timeout) moves the clock to the delivery time (or burns the whole timeout when nothing
    is due), and a recv with nothing in flight raises RuntimeError("deadlock ...") instead of hanging.
    `sent` keeps every payload this end sent.
    """

    def __init__(self, clock=None, latency_ns=0):
        self._clock = clock
        self._latency = int(latency_ns)
        self._inbox = collections.deque()          # (t_deliver_ns, payload)
        self._cv = threading.Condition()
        self._peer = None
        self._peer_closed = False
        self.closed = False
        self.sent = []

    @classmethod
    def pair(cls, clock=None, latency_ns=0):
        a, b = cls(clock, latency_ns), cls(clock, latency_ns)
        a._peer, b._peer = b, a
        return a, b

    def _check_open(self):
        if self.closed:
            raise OSError("handle is closed")

    def send_bytes(self, data):
        self._check_open()
        if self._peer.closed:
            raise BrokenPipeError("peer is closed")
        data = bytes(data)
        self.sent.append(data)
        due = (self._clock() + self._latency) if self._clock is not None else 0
        with self._peer._cv:
            self._peer._inbox.append((due, data))
            self._peer._cv.notify_all()

    def sent_msgs(self):
        return [json.loads(b.decode("utf-8")) for b in self.sent]

    def _ready(self):
        return bool(self._inbox) or self._peer_closed

    def poll(self, timeout=0.0):
        self._check_open()
        with self._cv:
            if self._clock is None:
                self._cv.wait_for(self._ready, None if timeout is None else max(0.0, timeout))
                return self._ready()
            if self._peer_closed and not self._inbox:
                return True
            now = self._clock()
            if self._inbox and self._inbox[0][0] <= now:
                return True
            wait_ns = None if timeout is None else round(max(0.0, timeout) * 1e9)
            if self._inbox:
                due = self._inbox[0][0]
                if wait_ns is None or now + wait_ns >= due:
                    self._clock.advance_ns(due - now)
                    return True
            elif wait_ns is None:
                raise RuntimeError("deadlock: poll(None) with nothing in flight on a virtual-clock FakeConn")
            self._clock.advance_ns(wait_ns)
            return False

    def recv_bytes(self):
        self._check_open()
        with self._cv:
            if self._clock is None:
                self._cv.wait_for(self._ready)
            elif not self._inbox and not self._peer_closed:
                raise RuntimeError("deadlock: recv_bytes with nothing in flight on a virtual-clock FakeConn")
            if self._inbox:
                due, data = self._inbox.popleft()
                if self._clock is not None and due > self._clock():
                    self._clock.advance_ns(due - self._clock())
                return data
            raise EOFError

    def close(self):
        with self._cv:
            if self.closed:
                return
            self.closed = True
            self._cv.notify_all()
        with self._peer._cv:
            self._peer._peer_closed = True
            self._peer._cv.notify_all()


# ===================================================================================================== board
Call = collections.namedtuple("Call", "name args")


class FakeBoard:
    """BoardControl that touches nothing. Every call is appended to `calls` (also the ones that fail).

    readback   what set_gpu_level returns: None (the commanded level), a number, a dict {commanded: read-back} or a
               function. The commanded level is still what `gpu_mhz` holds, so a test can tell them apart.
    emc_readback  the same for lock_emc.
    fail_on    {method name: exception} fails every call of that method; {method name: {n: exception}} fails only
               the n-th call (1-based). The exception may be an instance or a class.
    """

    GPU_LEVELS = (306, 408, 510, 612, 714, 816, 918, 1020)

    def __init__(self, gpu_levels=GPU_LEVELS, *, fail_on=None, readback=None, emc_readback=None,
                 power_mode=(C.POWER_MODE_NAME, C.POWER_MODE_ID)):
        self._levels = list(gpu_levels)
        self.fail_on = dict(fail_on or {})
        self._readback = readback
        self._emc_readback = emc_readback
        self.power_mode = tuple(power_mode)
        self.calls = []
        self._lock = threading.Lock()
        self.gpu_mhz = None
        self.readback_mhz = None
        self.emc_mhz = None
        self.governor_free = True
        self.closed = False
        self._stored = None

    def names(self):
        return [c.name for c in self.calls]

    def _enter(self, name, *args):
        with self._lock:
            self.calls.append(Call(name, args))
            nth = sum(1 for c in self.calls if c.name == name)
        failure = self.fail_on.get(name)
        if isinstance(failure, dict):
            failure = failure.get(nth)
        if failure is not None:
            raise failure

    @staticmethod
    def _read(spec, value):
        if spec is None:
            return value
        if callable(spec):
            return spec(value)
        if isinstance(spec, dict):
            return spec.get(value, value)
        return spec

    # ---- BoardControl
    def query_power_mode(self):
        self._enter("query_power_mode")
        return self.power_mode

    def ensure_power_mode(self, mode_id=C.POWER_MODE_ID):
        self._enter("ensure_power_mode", mode_id)
        name = C.POWER_MODE_NAME if mode_id == C.POWER_MODE_ID else "MODE_%s" % mode_id
        self.power_mode = (name, mode_id)

    def run_jetson_clocks(self):
        self._enter("run_jetson_clocks")
        self.gpu_mhz = self.readback_mhz = max(self._levels)
        self.governor_free = False

    def store_jetson_clocks(self, path):
        self._enter("store_jetson_clocks", path)
        self._stored = (path, self.snapshot_state())

    def restore_jetson_clocks(self, path):
        self._enter("restore_jetson_clocks", path)
        if self._stored is None or self._stored[0] != path:
            raise C.ClockError("nothing stored at %s" % path)
        state = self._stored[1]
        self.gpu_mhz, self.readback_mhz = state["gpu_mhz"], state["readback_mhz"]
        self.emc_mhz, self.governor_free = state["emc_mhz"], state["governor_free"]

    def gpu_levels(self):
        self._enter("gpu_levels")
        return list(self._levels)

    def set_gpu_level(self, mhz):
        self._enter("set_gpu_level", mhz)
        if mhz not in self._levels:
            raise ValueError("GPU level %r not offered by the board: %s" % (mhz, self._levels))
        self.gpu_mhz = mhz
        self.readback_mhz = int(self._read(self._readback, mhz))
        self.governor_free = False
        return self.readback_mhz

    def set_gpu_governor_free(self):
        self._enter("set_gpu_governor_free")
        self.governor_free = True

    def lock_emc(self, mhz):
        self._enter("lock_emc", mhz)
        if mhz not in C.EMC_LOCKABLE_MHZ:
            raise ValueError("EMC %r MHz is not lockable: %s" % (mhz, C.EMC_LOCKABLE_MHZ))
        self.emc_mhz = float(self._read(self._emc_readback, mhz))
        return self.emc_mhz

    def unlock_emc(self):
        self._enter("unlock_emc")
        self.emc_mhz = None

    def apply_arm(self, arm, gpu_mhz, emc_mhz):
        self._enter("apply_arm", arm, gpu_mhz, emc_mhz)
        if arm not in C.ARMS:
            raise ValueError("arm %r not in %s" % (arm, C.ARMS))
        if arm == "pinned":
            if gpu_mhz is None:
                raise ValueError("arm 'pinned' needs gpu_mhz")
            self.set_gpu_level(gpu_mhz)
        else:
            self.set_gpu_governor_free()
        if arm == "stock":
            self.unlock_emc()
        elif emc_mhz is not None:
            self.lock_emc(emc_mhz)
        return {"arm": arm, "gpu_mhz": self.readback_mhz if arm == "pinned" else None, "emc_mhz": self.emc_mhz,
                "governor_free": self.governor_free}

    def snapshot_state(self):
        return {"gpu_mhz": self.gpu_mhz, "readback_mhz": self.readback_mhz, "emc_mhz": self.emc_mhz,
                "governor_free": self.governor_free}

    def snapshot(self):
        self._enter("snapshot")
        return dict(self.snapshot_state(), power_mode=list(self.power_mode))

    def close(self):
        self._enter("close")
        self.closed = True


# ===================================================================================================== models
class StubSlowdown:
    """SlowdownModel with slowdown s = n / c(n), n = |A|. c maps n to total throughput; c(n) for an n above the
    largest key uses the largest key's value, so c={1: 1.0, 2: 0.93} means c(n >= 2) = 0.93. predict(i, (i,), f) = 1."""

    def __init__(self, c=None, *, name="M2"):
        self.c = {1: 1.0, 2: 0.93} if c is None else {int(k): float(v) for k, v in c.items()}
        self.name = name

    def _c(self, n):
        return self.c[max(k for k in self.c if k <= n)]

    def fit(self, corun_rows, solo_rows, types):
        for row in corun_rows:
            if row.get("split") == "heldout":
                raise C.HeldOutError("fit received a split == 'heldout' row")
        return self

    def predict(self, i, A, f_mhz):
        n = len(A)
        if n < 1:
            raise ValueError("A must contain the type itself")
        return 1.0 if n == 1 else n / self._c(n)

    def sample(self, i, A, f_mhz, rng):
        return self.predict(i, A, f_mhz)

    def to_json(self):
        return {"schema": C.SCHEMAS["slowdown"], "name": self.name,
                "params": {"c": {str(k): v for k, v in self.c.items()}}, "residuals": {},
                "heldout_mape": {self.name: 0.0}, "fit_rows_sha256": "0" * 64}


class UnitSlowdown(StubSlowdown):
    """s = 1 whatever A is."""

    def __init__(self):
        super().__init__({1: 1.0}, name="M0")

    def predict(self, i, A, f_mhz):
        return 1.0

    def to_json(self):
        out = super().to_json()
        out["params"] = {}
        return out


class StubPower:
    """PowerModelP with the K13 formula p(A, f) = p_idle(f) + dp(f) * min(1, U_A / eta(f)), U_A = sum_i u_i(f).

    p_idle, dp, eta: a number or {f_mhz: number}.  u: a number (every type), {type_id: number}, f(type_id, f_mhz),
    or pass types=TypeTable to use each type's own u[f].  p((), f) is p_idle(f) exactly.
    """

    def __init__(self, p_idle=5.0, dp=5.0, eta=1.0, u=0.5, *, levels=(612, 1020), types=None):
        self.p_idle, self.dp, self.eta, self.u, self.types = p_idle, dp, eta, u, types
        self.levels = tuple(levels)

    @staticmethod
    def _at(v, f):
        return v[f] if isinstance(v, dict) else v

    def _u(self, type_id, f):
        if self.types is not None:
            return self.types.get(type_id).u[f]
        if callable(self.u):
            return self.u(type_id, f)
        return self.u[type_id] if isinstance(self.u, dict) else self.u

    def p(self, A, f_mhz):
        f = int(f_mhz)
        idle = self._at(self.p_idle, f)
        if not A:
            return idle
        used = sum(self._u(i, f) for i in A)
        return idle + self._at(self.dp, f) * min(1.0, used / self._at(self.eta, f))

    def to_json(self):
        levels = sorted({int(f) for v in (self.p_idle, self.dp, self.eta) if isinstance(v, dict) for f in v}
                        or self.levels)
        return {"schema": C.SCHEMAS["power"],
                "p_idle_w": {str(f): self._at(self.p_idle, f) for f in levels},
                "dp_w": {str(f): self._at(self.dp, f) for f in levels},
                "eta": {str(f): self._at(self.eta, f) for f in levels}}


# ===================================================================================================== types
DEFAULT_TYPES = {
    "bert:5": {"tau": {612: 0.0412, 1020: 0.029}, "u": {612: 0.71, 1020: 0.66}, "act_mb": 38.0},
    "vision:24": {"tau": {612: 0.094, 1020: 0.06}, "u": {612: 0.902, 1020: 0.82}, "act_mb": 51.0},
}


def _round(x):
    return float("%.9g" % x)


def _tau_stats(value):
    if isinstance(value, dict):
        out = {"p50": value["p50"], "mean": value["mean"], "p95": value["p95"]}
        out["samples"] = list(value.get("samples", [value["p50"]]))
        return out
    return {"p50": value, "mean": value, "p95": value, "samples": [value]}


def _per_level(value, levels, scale_down):
    """Value for every level. A dict gives the levels it names; a missing level is derived from the highest given
    one (scale_down: tau grows as 1/f), a scalar applies to every level. Levels not in `levels` are dropped."""
    if not isinstance(value, dict):
        return {f: value for f in levels}
    top = max(value)
    out = {}
    for f in levels:
        if f in value:
            out[f] = value[f]
        elif scale_down:
            ref = _tau_stats(value[top])
            scale = top / f
            out[f] = {"p50": _round(ref["p50"] * scale), "mean": _round(ref["mean"] * scale),
                      "p95": _round(ref["p95"] * scale), "samples": [_round(x * scale) for x in ref["samples"]]}
        else:
            out[f] = value[top]
    return out


def make_types_json(spec=None, *, alpha=5.0, f_levels=None, corun_levels=None, heldout=None, resident_mb=1300.0):
    """K1 dict for a few types.

    spec {type_id: {"tau": x, "klass": ?, "u": ?, "act_mb": ?, "head_frac": ?}}. tau is a number (every level), a
    {f: number} dict, or {f: {"p50","mean","p95","samples"}}. klass defaults to the median rule of the contract;
    u to 0.5; head_frac to 0.1 (0 for llama); ckpt_frac is linear up to 1 - head_frac.
    """
    spec = DEFAULT_TYPES if spec is None else spec
    given = {int(f) for s in spec.values() if isinstance(s["tau"], dict) for f in s["tau"]}
    levels = tuple(sorted(f_levels)) if f_levels else tuple(sorted(given)) or (612, 1020)
    corun = tuple(sorted(corun_levels)) if corun_levels else levels
    fmax = max(levels)
    tau = {tid: _per_level({int(f): v for f, v in s["tau"].items()} if isinstance(s["tau"], dict) else s["tau"],
                           levels, True) for tid, s in spec.items()}
    means = {tid: _tau_stats(tau[tid][fmax])["mean"] for tid in spec}
    median = statistics.median(means.values())
    types = []
    for tid, s in spec.items():
        fam, ex = C.parse_type_id(tid)
        ckpts = C.default_checkpoints(fam, ex)
        head = s.get("head_frac", 0.0 if fam == "llama" else 0.1)
        frac = s.get("ckpt_frac") or {str(c): _round((1.0 - head) * (k + 1) / len(ckpts)) for k, c in enumerate(ckpts)}
        u = _per_level({int(f): v for f, v in s["u"].items()} if isinstance(s.get("u"), dict) else s.get("u", 0.5),
                       levels, False)
        types.append({
            "type_id": tid, "family": fam, "exit": ex,
            "klass": s.get("klass") or ("short" if means[tid] <= median else "long"), "ckpts": ckpts,
            "tau_s": {str(f): _tau_stats(tau[tid][f]) for f in levels}, "ckpt_frac": frac,
            "head_frac": head, "u": {str(f): u[f] for f in levels}, "act_mb": s.get("act_mb", 40.0)})
    fams = {t["family"]: {"resident_mb": resident_mb} for t in types}
    return {"schema": C.SCHEMAS["types"], "synthetic": True, "alpha": alpha, "f_levels_mhz": list(levels),
            "corun_levels_mhz": list(corun), "heldout_level_mhz": heldout, "f_max_mhz": fmax,
            "llama_tokens": C.LLAMA_TOKENS, "llama_prompt_tokens": C.LLAMA_PROMPT_TOKENS, "families": fams,
            "types": types,
            "provenance": {"klass_rule": "short iff tau_mean(f_max) <= median over types",
                           "sources": ["mocks.make_types"], "sha256": {}}}


def make_types(spec=None, **kw):
    """TypeTable built from make_types_json (validated by TypeTable.from_json)."""
    return C.TypeTable.from_json(make_types_json(spec, **kw))


# ===================================================================================================== simulator
def make_simresult(spec_name="stub", per_type=None, *, energy_per_req_j=1.0, energy_ci=None, mean_power_w=10.0,
                   throughput=10.0, mean_in_system=1.0, utilization=0.5, unstable=False, tagged=None, reps=None,
                   trace=None, runtime_s=0.0):
    """A validated SimResult. per_type {type_id: overrides of TypeStats fields}; the defaults describe a type that
    always completes. mean_completion defaults to 1 - mean_shortfall (and the other way round)."""
    out = {}
    for tid, over in (per_type if per_type is not None else {"bert:5": {}}).items():
        row = {"n": 1000, "mean_R": 0.05, "p50_R": 0.04, "p95_R": 0.1, "p99_R": 0.12, "complete_rate": 1.0,
               "truncation_rate": 0.0, "miss_rate": 0.0, "expired_rate": 0.0}
        row.update(over)
        if "mean_completion" not in row:
            row["mean_completion"] = 1.0 - row.get("mean_shortfall", 0.0)
        row.setdefault("mean_shortfall", 1.0 - row["mean_completion"])
        out[tid] = C.TypeStats.from_json(row)
    if energy_ci is None:
        energy_ci = (energy_per_req_j * 0.98, energy_per_req_j * 1.02)
    if reps is None:
        reps = [{"seed": 0, "energy_per_req_j": energy_per_req_j, "n_slope": 0.0, "per_type": {}, "tagged": None}]
    res = C.SimResult(spec_name, out, energy_per_req_j, tuple(energy_ci), mean_power_w, throughput, mean_in_system,
                      utilization, unstable, tagged, tuple(reps), trace, runtime_s)
    C.validate_simresult(res.to_json())
    return res


SimCall = collections.namedtuple("SimCall", "key spec args kwargs")


class StubSim:
    """sim_fn driven by a table. Call it like simulate(): the PolicySpec (or its json) is found in the arguments
    or in the `spec` keyword. The lookup tries (k_short, k_long, order, f_mhz), then (k_short, k_long), then
    `default`; otherwise KeyError names the key. A table value is a SimResult or a dict of make_simresult keywords
    (per_type, energy_per_req_j, ...). Every call is appended to `calls` as SimCall(key, spec, args, kwargs)."""

    def __init__(self, table, *, default=None):
        self.table = dict(table)
        self.default = default
        self.calls = []

    @staticmethod
    def _find_spec(args, kwargs):
        cands = [kwargs["spec"]] if "spec" in kwargs else list(args)
        for a in cands:
            if isinstance(a, C.PolicySpec):
                return a
            if isinstance(a, dict) and "k_short" in a:
                return C.PolicySpec.from_json(a)
        raise TypeError("StubSim call carries no PolicySpec")

    def __call__(self, *args, **kwargs):
        spec = self._find_spec(args, kwargs)
        key = (spec.k_short, spec.k_long, spec.order, spec.f_mhz)
        self.calls.append(SimCall(key, spec, args, kwargs))
        for k in (key, key[:2]):
            if k in self.table:
                entry = self.table[k]
                break
        else:
            if self.default is None:
                raise KeyError("StubSim has no entry for %r" % (key,))
            entry = self.default
        if isinstance(entry, C.SimResult):
            return entry
        entry = dict(entry)
        return make_simresult(spec.name, entry.pop("per_type", None), **entry)

    @classmethod
    def from_response_maps(cls, br_short, br_long, K, types=("bert:5", "vision:24")):
        """Table over K x K in which the short class's best reply to k_long is br_short[k_long] and the long class's
        best reply to k_short is br_long[k_short]. A class has mean_shortfall 0 at its best reply and a larger,
        distinct shortfall at every other k (0.1 + 0.01 * index of k in K), so a best-response iteration follows
        the maps: fixed points where both maps agree, a cycle where they chase each other."""
        K = tuple(K)
        short_t, long_t = types

        def cost(best, k):
            return 0.0 if k == best else 0.1 + 0.01 * K.index(k)

        table = {}
        for ks in K:
            for kl in K:
                per_type = {}
                for tid, sf in ((short_t, cost(br_short[kl], ks)), (long_t, cost(br_long[ks], kl))):
                    per_type[tid] = {"mean_shortfall": sf, "mean_R": 0.05 + sf, "complete_rate": 1.0 - sf,
                                     "truncation_rate": sf}
                energy = 1.0 + 0.1 * K.index(ks) + 0.05 * K.index(kl)
                table[(ks, kl)] = {"per_type": per_type, "energy_per_req_j": energy}
        return cls(table)


# ===================================================================================================== golden cases
ROOT = Path(__file__).resolve().parents[2]


def repo_path(rel):
    return ROOT / rel


def load_golden(name):
    """S0a golden cases: 'admission_cases', 'admission_traces' or 'truncation_cases' (a list of dicts)."""
    return load_fixture(name + ".json")


def golden_types(trace):
    """The TypeTable a golden trace points to with its `types_ref`."""
    return C.load_types(repo_path(trace["types_ref"]))


# ===================================================================================================== sessions
# Everything below is deterministic (no random numbers, no math library calls): the same call writes the same bytes
# on every machine, so tests can compare a regenerated session with the committed fixture.
SESSION_CLOSED_ID = "s20261020T101500"
SESSION_OPEN_ID = "s20261022T080000"
WALL_OPEN_NS = 1_792_656_000_000_000_000       # 2026-10-22T08:00:00Z
GIT_COMMIT = "0123456789abcdef0123456789abcdef01234567"
RECV_NS, START_NS, DONE_NS = 150_000, 50_000, 200_000   # t_srv_recv - t_admit, t_start - t_srv_recv, t_done - t_end
GAP_NS = 100_000                                          # closed loop: next admit this long after the last t_done
EMC_MHZ = 3199.0

# A smooth part (so the series is positively autocorrelated, r1 about 0.3, like a real one) plus an irregular part.
_NOISE_A = (0.000, 0.014, 0.020, 0.014, 0.000, -0.014, -0.020, -0.014)
_NOISE_B = (0.010, -0.012, 0.006, -0.004, 0.013, -0.009, 0.003, -0.014, 0.008, -0.002, 0.000)
_FRAC_JITTER = (0.0, 0.01, -0.01, 0.005, -0.005)
_RIPPLE_MW = (0, 30, -30, 10, -10)
_RIPPLE_LOAD = (0, 4, -4, 2)


def synth_noise(i, phase=0):
    """Relative deviation of request i from its nominal time (|x| < 0.04)."""
    return _NOISE_A[(i + phase) % 8] + _NOISE_B[i % 11]


def _dump_json(path, obj, indent=2):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(obj, indent=indent, ensure_ascii=True) + "\n")


def _dump_csv(path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(header)
        w.writerows(rows)


def make_ready(family, exits, *, pid=4242, t_ready_ns=0, load_s=1.2, first_infer_s=0.2, act_mb=38.0,
               torch_reserved_mb=1380.0, rss_mb=2110.0, cache_files=0, compile_mode="on", input_source="per_request"):
    """A K5 `ready` message that passes contracts.validate_ready."""
    exits = list(exits)

    def per_exit(v):
        return {str(e): (v[e] if isinstance(v, dict) else v) for e in exits}

    return {"op": "ready", "family": family, "pid": pid, "exits_warmed": exits,
            "warmup": {"load_s": load_s, "first_infer_s": per_exit(first_infer_s), "n_warm": per_exit(20),
                       "cv_last5": per_exit(0.012), "act_mb": per_exit(act_mb),
                       "inductor_cache_files_at_start": cache_files},
            "input_source": input_source,
            "versions": {"python": "3.10.12", "torch": "2.8.0", "transformers": "4.45.0", "cuda": "12.6",
                         "compile": compile_mode},
            "mem": {"torch_reserved_mb": torch_reserved_mb, "rss_mb": rss_mb}, "t_ready_ns": t_ready_ns}


def make_manifest(condition_id, mode, *, t_wall_start_ns, t_mono_start_ns, t_mono_end_ns, settle_s, arm="pinned",
                  gpu_mhz=None, emc_mhz=EMC_MHZ, valid=True, invalid_reason=None, board_readback=None,
                  n_requests=None, n_warmup=None, seed=None, policy_key=None, policy_spec=None,
                  freeze_condition_id=None, workload=None, alpha=None, cell=None, split=None, repeat=None):
    """A K9 manifest (exp2.manifest/1) with every key present, in the documented order."""
    if board_readback is None and valid:
        board_readback = {"gpu_mhz": gpu_mhz, "emc_mhz": emc_mhz, "governor_free": arm != "pinned"}
    return {"schema": C.SCHEMAS["manifest"], "condition_id": condition_id, "mode": mode, "arm": arm, "gpu_mhz": gpu_mhz,
            "emc_mhz": emc_mhz, "t_wall_start_ns": t_wall_start_ns, "t_mono_start_ns": t_mono_start_ns,
            "t_mono_end_ns": t_mono_end_ns, "settle_s": settle_s, "valid": valid, "invalid_reason": invalid_reason,
            "board_readback": board_readback, "n_requests": n_requests, "n_warmup": n_warmup, "seed": seed,
            "policy_key": policy_key, "policy_spec": policy_spec, "freeze_condition_id": freeze_condition_id,
            "workload": workload, "alpha": alpha, "cell": cell, "split": split, "repeat": repeat}


def closed_rows(condition_id, type_id, klass, *, t_begin_ns, t_stop_ns, gpu_mhz, base_tau_s, ckpt_frac, n_warmup,
                n_in_service, slowdown=1.0, seed=0, phase=0):
    """K8 rows of one closed-loop tenant: back-to-back requests from t_begin_ns until t_stop_ns (the request in
    flight at t_stop finishes). tau = t_done - t_admit = base_tau_s * slowdown * (1 + synth_noise(i, phase)) ns,
    cut into t_srv_recv / t_start / t_end with the fixed overheads; checkpoint k lands at ckpt_frac[k] of the
    service time (a +-1% jitter per request); the head takes the rest. ckpt_frac is aligned with the checkpoints."""
    family, exit_ = C.parse_type_id(type_id)
    ckpts = C.default_checkpoints(family, exit_)
    if len(ckpt_frac) != len(ckpts):
        raise ValueError("ckpt_frac needs one entry per checkpoint of %s" % type_id)
    rows, t, i = [], t_begin_ns, 0
    while t < t_stop_ns:
        tau_ns = round(base_tau_s * 1e9 * slowdown * (1.0 + synth_noise(i, phase)))
        service = tau_ns - RECV_NS - START_NS - DONE_NS
        t_recv, t_start = t + RECV_NS, t + RECV_NS + START_NS
        t_end = t_start + service
        t_done = t_end + DONE_NS
        jit = 1.0 + _FRAC_JITTER[i % 5]
        rows.append({
            "condition_id": condition_id, "req_id": "r%06d" % (i + 1), "type_id": type_id, "family": family,
            "exit": exit_, "klass": klass, "target": ckpts[-1], "t_arrival_ns": t, "t_arrival_actual_ns": t,
            "t_deadline_ns": None, "t_admit_ns": t, "t_srv_recv_ns": t_recv, "t_start_ns": t_start, "t_end_ns": t_end,
            "t_done_ns": t_done, "server": family, "policy": "closed", "policy_spec": None, "f_mhz": gpu_mhz,
            "n_in_service_at_admit": n_in_service, "queue_len_at_arrival": 0, "reached": ckpts[-1], "completion": 1.0,
            "outcome": "complete", "tokens_done": ckpts[-1] if family == "llama" else None,
            "ckpt": [[c, t_start + round(service * f * jit)] for c, f in zip(ckpts, ckpt_frac)],
            "is_warmup": i < n_warmup, "seed": seed, "mode": "closed", "err": None})
        t = t_done + GAP_NS
        i += 1
    return rows


class Timeline:
    """Piecewise-constant telemetry truth. add(t0, t1, mw=, load=, gpu=) overrides earlier segments where they overlap;
    outside every segment the defaults apply. A value may be a function of t_ns."""

    def __init__(self, base_mw, base_load, base_gpu):
        self.base = {"mw": base_mw, "load": base_load, "gpu": base_gpu}
        self.segs = []

    def add(self, t0, t1, **vals):
        self.segs.append((t0, t1, vals))

    def _at(self, t, key):
        for t0, t1, vals in reversed(self.segs):
            if t0 <= t < t1 and key in vals:
                v = vals[key]
                return v(t) if callable(v) else v
        return self.base[key]

    def at_mw(self, t):
        return self._at(t, "mw")

    def at_load(self, t):
        return self._at(t, "load")

    def at_gpu(self, t):
        return self._at(t, "gpu")


def write_session(root, conditions, *, plan, servers, idle, timeline, t_end_ns, freeze=None,
                  session_id=SESSION_CLOSED_ID, dirname=None, t_mono_start_ns=MONO0_NS, t_wall_start_ns=WALL0_NS,
                  power_hz=10, clocks_hz=5):
    """Write one session directory in the K9 layout under root/dirname (default: session_id), return its Path.

    conditions: list of {"manifest": dict, "requests": [rows], "freq_switch": [[t_req, t_done, target, readback], ...]}.
    servers:    {family: {"pid", "t_spawn_ns", "t_ready_ns", "ready": make_ready(...)}}.
    idle:       {"pre": {"t0_ns", "t1_ns", "gpu_mhz"}, "resident": {...}}.
    timeline:   Timeline with the power (mW), GPU load (permille) and GPU MHz at every instant. power.csv gets a
                fixed ripple on top so the series is not flat; vdd_in_mw is exactly mv * ma / 1000.
    The plan and freeze dicts are only used for their hashes; write them yourself where the test needs them.
    """
    sdir = Path(root) / (dirname or session_id)

    def wall(t):
        return t_wall_start_ns + (t - t_mono_start_ns)

    p_period = round(1e9 / power_hz)
    n_power = (t_end_ns - t_mono_start_ns) // p_period + 1
    prows = []
    for k in range(n_power):
        t = t_mono_start_ns + k * p_period
        mv = 5000 + 8 * (k % 5)
        ma = round((timeline.at_mw(t) + _RIPPLE_MW[k % 5]) * 1000 / mv)
        prows.append([t, wall(t), mv, ma, mv * ma / 1000])
    _dump_csv(sdir / "telemetry" / "power.csv", C.POWER_CSV_COLS, prows)

    c_period = round(1e9 / clocks_hz)
    n_clocks = (t_end_ns - t_mono_start_ns) // c_period + 1
    header = list(C.CLOCKS_CSV_FIXED_COLS) + ["temp_cpu_thermal_mc", "temp_gpu_thermal_mc"]
    crows = []
    for k in range(n_clocks):
        t = t_mono_start_ns + k * c_period
        mhz = timeline.at_gpu(t)
        load = max(0, min(1000, timeline.at_load(t) + _RIPPLE_LOAD[k % 4]))
        ready_n = sum(1 for s in servers.values() if s["t_ready_ns"] <= t)
        crows.append([t, wall(t), mhz * 1_000_000, mhz * 1_000_000, mhz * 1_000_000, load, EMC_MHZ,
                      3_650_000 - 90_000 * ready_n, 1_190_400, 41_000 + 250 * (k % 3), 39_000 + 250 * (k % 4)])
    _dump_csv(sdir / "telemetry" / "clocks.csv", header, crows)

    _dump_json(sdir / "telemetry" / "meta.json", {
        "schema": C.SCHEMAS["telemetry_meta"],
        "vdd_in": {"hwmon": "/sys/class/hwmon/hwmon3", "channel": 2, "label": "VDD_IN", "update_interval_ms": 1,
                   "samples": 4},
        "target_hz": power_hz, "achieved_hz": float(power_hz), "clocks_target_hz": clocks_hz,
        "devfreq": "/sys/class/devfreq/17000000.gpu", "emc_source": "tegrastats",
        "thermal_zones": ["cpu-thermal", "gpu-thermal"], "boot_id": "synthetic-boot-id",
        "t_mono_start_ns": t_mono_start_ns, "t_wall_start_ns": t_wall_start_ns, "n_rows": n_power})

    for cond in conditions:
        cid = cond["manifest"]["condition_id"]
        _dump_json(sdir / cid / "manifest.json", cond["manifest"])
        path = sdir / cid / "requests.jsonl"
        with open(path, "w", encoding="utf-8", newline="\n") as fh:
            for row in cond["requests"]:
                fh.write(json.dumps(row, separators=(",", ":")) + "\n")
        _dump_csv(sdir / cid / "freq_switch.csv", C.FREQ_SWITCH_COLS, cond["freq_switch"])

    snap = {"gpu_mhz": 1020, "emc_mhz": EMC_MHZ, "governor_free": True,
            "power_mode": [C.POWER_MODE_NAME, C.POWER_MODE_ID]}
    _dump_json(sdir / "session.json", {
        "schema": C.SCHEMAS["session"], "session_id": session_id, "plan_sha256": plan["sha256"],
        "plan_kind": plan["kind"], "freeze_content_sha256": None if freeze is None else freeze["content_sha256"],
        "t_wall_start_ns": t_wall_start_ns, "t_mono_start_ns": t_mono_start_ns, "t_wall_end_ns": wall(t_end_ns),
        "git": {"commit": GIT_COMMIT, "dirty": False, "diff_sha256": None},
        "host": {"python": "3.10.12", "numpy": "1.26.4", "platform": "linux-aarch64", "boot_id": "synthetic-boot-id"},
        "board": {"dry": True, "snapshot_start": snap, "snapshot_end": snap,
                  "jetson_clocks_store": "/tmp/exp2_jc_store"},
        "idle": idle,
        "servers": {fam: {"pid": s["pid"], "t_spawn_ns": s["t_spawn_ns"], "t_ready_ns": s["t_ready_ns"],
                          "t_spawn_wall_ns": wall(s["t_spawn_ns"]), "ready": s["ready"]}
                    for fam, s in servers.items()},
        "conditions": [c["manifest"]["condition_id"] for c in conditions], "exit_code": 0})
    return sdir


# ---- Run 1 (closed loop) -------------------------------------------------------------------------------
CLOSED_LEVELS = (1020, 612)
TAU_S = {"bert:5": {612: 0.0412, 1020: 0.029}, "vision:24": {612: 0.094, 1020: 0.06}}
U_PERMILLE = {"bert:5": {612: 710, 1020: 660}, "vision:24": {612: 902, 1020: 820}}
PAIR_SLOWDOWN = {"bert:5": {612: 1.9, 1020: 2.05}, "vision:24": {612: 1.45, 1020: 1.5}}
CKPT_FRAC = {"bert:5": (0.62, 0.90), "vision:24": (0.15, 0.30, 0.45, 0.60, 0.75, 0.90)}
HEAD_FRAC = {"bert:5": 0.10, "vision:24": 0.10}
KLASS = {"bert:5": "short", "vision:24": "long"}
CELL = ("bert:5", "vision:24")
P_BASE_MW, P_IDLE_PRE_MW = 5400, 5050
P_LOAD_MW = {"bert": 8400, "vision": 8800}
P_IDLE_RES_MW = {612: 5100, 1020: 5620}
P_SOLO_MW = {"bert:5": {612: 8200, 1020: 9410}, "vision:24": {612: 9600, 1020: 11800}}
P_PAIR_MW = {612: 11200, 1020: 13600}
LOAD_BASE_PERMILLE, LOAD_PAIR_PERMILLE = 30, 990
CLOSED_SETTLE_S, CLOSED_IDLE_WINDOW_S, CLOSED_N_WARMUP = 1.0, 2.5, 2
CLOSED_DURATION_S = {"idle": 2.0, "bert:5": 0.8, "vision:24": 1.0, "pair": 1.5}


def closed_condition_specs():
    """The eight Run-1 conditions in plan order: (condition_id, mode, f, cell, split, duration_s).
    Co-run condition ids are '<cell_id>_f<f>_r<repeat>': the analyzer reads the cell id off the front."""
    out = [("idle_f%d" % f, "idle", f, None, None, CLOSED_DURATION_S["idle"]) for f in CLOSED_LEVELS]
    for tid in CELL:
        for f in CLOSED_LEVELS:
            out.append(("solo_%s_f%d_r0" % (tid.replace(":", "-"), f), "closed", f, (tid,), "solo",
                        CLOSED_DURATION_S[tid]))
    for f in CLOSED_LEVELS:
        out.append(("c001_f%d_r0" % f, "closed", f, CELL, "train", CLOSED_DURATION_S["pair"]))
    return out


def closed_plan():
    """exp2.plan/1 (closed) of the synthetic Run-1 session.
    estimated_wall_s = 2 * idle_window_s + sum over conditions of (settle_s + duration_s)."""
    conds = []
    for cid, mode, f, cell, split, dur in closed_condition_specs():
        c = {"condition_id": cid, "mode": mode, "arm": "pinned", "gpu_mhz": f}
        if mode == "idle":
            c["duration_s"] = dur
        else:
            c.update({"cell": list(cell), "duration_s": dur, "n_warmup": CLOSED_N_WARMUP, "split": split, "repeat": 0})
        conds.append(c)
    wall_s = 2 * CLOSED_IDLE_WINDOW_S + sum(CLOSED_SETTLE_S + c["duration_s"] for c in conds)
    plan = {"schema": C.SCHEMAS["plan"], "kind": "closed",
            "session": {"families": ["bert", "vision"], "exits": {"bert": [5], "vision": [24]}, "emc_mhz": EMC_MHZ,
                        "settle_s": CLOSED_SETTLE_S, "idle_window_s": CLOSED_IDLE_WINDOW_S, "n_inputs": 32,
                        "preprocess": "per_request"},
            "freeze": None, "conditions": conds, "estimated_wall_s": wall_s}
    return C.with_plan_sha256(plan)


def synthetic_closed_session(root, *, session_id=SESSION_CLOSED_ID, dirname=None):
    """Write the synthetic Run-1 session (idle, solo and pair conditions at 612 and 1020 MHz) under root and return
    its directory. The truth behind the numbers is in the constants above: nominal solo tau TAU_S, pair slowdown
    PAIR_SLOWDOWN, GPU load U_PERMILLE, power P_*_MW, all perturbed by synth_noise."""
    plan = closed_plan()
    T0 = MONO0_NS

    def ns(s):
        return T0 + round(s * 1e9)

    tl = Timeline(P_BASE_MW, LOAD_BASE_PERMILLE, 1020)
    tl.add(ns(0.5), ns(3.0), mw=P_IDLE_PRE_MW)
    servers = {}
    for (fam, tid, spawn, ready, pid, load_s, first, act, rsv, rss) in (
            ("bert", "bert:5", 3.2, 4.6, 4242, 1.2, 0.2, 38.0, 1380.0, 2110.0),
            ("vision", "vision:24", 4.8, 6.4, 4243, 1.4, 0.2, 51.0, 1260.0, 1980.0)):
        tl.add(ns(spawn), ns(ready), mw=P_LOAD_MW[fam], load=150)
        servers[fam] = {"pid": pid, "t_spawn_ns": ns(spawn), "t_ready_ns": ns(ready),
                        "ready": make_ready(fam, [int(tid.split(":")[1])], pid=pid, t_ready_ns=ns(ready) - 500_000,
                                            load_s=load_s, first_infer_s=first, act_mb=act, torch_reserved_mb=rsv,
                                            rss_mb=rss)}
    tl.add(ns(6.9), ns(8.9), mw=P_IDLE_RES_MW[1020])
    idle = {"pre": {"t0_ns": ns(0.5), "t1_ns": ns(3.0), "gpu_mhz": 1020},
            "resident": {"t0_ns": ns(6.9), "t1_ns": ns(8.9), "gpu_mhz": 1020}}

    conds, t_c = [], ns(9.0)
    for cid, mode, f, cell, split, dur in closed_condition_specs():
        run0 = t_c + round(CLOSED_SETTLE_S * 1e9)
        stop = run0 + round(dur * 1e9)
        rows = []
        if mode == "closed":
            for j, tid in enumerate(cell):
                rows += closed_rows(cid, tid, KLASS[tid], t_begin_ns=run0 + 3_000_000 * j, t_stop_ns=stop, gpu_mhz=f,
                                    base_tau_s=TAU_S[tid][f], slowdown=PAIR_SLOWDOWN[tid][f] if len(cell) > 1 else 1.0,
                                    ckpt_frac=CKPT_FRAC[tid], n_warmup=CLOSED_N_WARMUP, n_in_service=len(cell) - 1,
                                    phase=5 * j)
        end = max([r["t_done_ns"] for r in rows] + [stop]) + 1_000_000
        tl.add(t_c, end + 500_000_000, gpu=f)
        tl.add(t_c, run0, mw=P_IDLE_RES_MW[f])
        if mode == "idle":
            tl.add(run0, end, mw=P_IDLE_RES_MW[f])
        elif len(cell) == 1:
            tl.add(run0, end, mw=P_SOLO_MW[cell[0]][f], load=U_PERMILLE[cell[0]][f])
        else:
            tl.add(run0, end, mw=P_PAIR_MW[f], load=LOAD_PAIR_PERMILLE)
        rows.sort(key=lambda r: (r["t_admit_ns"], r["type_id"]))
        for n, r in enumerate(rows):
            r["req_id"] = "r%06d" % (n + 1)
        mani = make_manifest(
            cid, mode, t_wall_start_ns=WALL0_NS + (t_c - T0), t_mono_start_ns=t_c, t_mono_end_ns=end,
            settle_s=CLOSED_SETTLE_S, gpu_mhz=f, n_requests=len(rows),
            n_warmup=0 if mode == "idle" else CLOSED_N_WARMUP, seed=0, cell=None if cell is None else list(cell),
            split=split, repeat=None if mode == "idle" else 0)
        conds.append({"manifest": mani, "requests": rows, "freq_switch": [[t_c + 10_000_000, t_c + 90_000_000, f, f]]})
        t_c = end + 500_000_000
    return write_session(root, conds, plan=plan, servers=servers, idle=idle, timeline=tl,
                         t_end_ns=t_c + 500_000_000, session_id=session_id, dirname=dirname)


# ---- Run 2 (open loop) ---------------------------------------------------------------------------------
_GAP_PATTERN = (0.4, 1.7, 0.9, 1.3, 0.6, 1.5, 0.8, 0.8)      # mean 1.0: inter-arrival gaps in units of 1 / rate
DP_MW = {612: 3100, 1020: 4100}                                # extra power at full load, per GPU level
OPEN_STRESS = {"default": {6: 6.0, 11: 14.0}, "serial": {4: 6.0}, "g1": {4: 6.0}}    # {policy_key: {request index: x}}
OPEN_ERRORS = {"F_sens": {8: "synthetic: CUDA error: out of memory"}}          # {policy_key: {request index: err}}
OPEN_SPAWN_AFTER_S = 3.2                                      # first server spawns this long after the pre-idle window


def open_schedule(workload, n, seed=0):
    """Deterministic ScheduleItems for a workload (no random numbers): gaps cycle through _GAP_PATTERN / rate,
    types follow a smooth weighted round robin over 10 slots. Times are relative to the condition start."""
    ids, probs = list(workload["types"]), list(workload["probs"])
    counts = [round(p * 10) for p in probs]
    if sum(counts) != 10 or any(abs(p * 10 - c) > 1e-6 for p, c in zip(probs, counts)):
        raise ValueError("probs must be multiples of 0.1 for open_schedule, got %r" % (probs,))
    cur, seq = [0] * len(ids), []
    for _ in range(10):
        cur = [c + k for c, k in zip(cur, counts)]
        j = max(range(len(ids)), key=lambda x: (cur[x], -x))
        seq.append(ids[j])
        cur[j] -= 10
    t, items = 0.0, []
    for i in range(n):
        t += _GAP_PATTERN[(i + seed) % 8] / workload["rate_per_s"]
        items.append({"req_id": "r%06d" % (i + 1), "type_id": seq[(i + seed) % 10], "t_arrival_ns": round(t * 1e9),
                      "t_deadline_ns": None, "k_override": None})
    return items


def open_rows(condition_id, spec, workload, types, *, t0_ns, n_requests, n_warmup, seed, alpha, stress=None,
              errors=None):
    """K8 rows of one open-loop condition under a FIFO model that honours the spec's k limits.

    A request is admitted when fewer than k(class) requests are in service; waiting past its deadline expires it.
    Service time = tau_p50(f) * (1 + 0.45 * n_in_service) * (1 + noise) * stress[i]; checkpoints land at the types'
    ckpt_frac; the first checkpoint after the deadline stops the request (contracts.truncation_outcome decides the
    outcome) and the head then runs for reached. f = spec.f_mhz for a static rule, else f_max (governor).
    errors {request index: message} makes that request answer with an error 1 ms after it starts.
    """
    spec = spec if isinstance(spec, C.PolicySpec) else C.PolicySpec.from_json(spec)
    f_eff = spec.f_mhz if spec.f_rule == "static" else types.f_max_mhz
    stress, errors = stress or {}, errors or {}
    items = C.with_deadlines(open_schedule(workload, n_requests, seed), types, alpha)
    rows, busy, left_queue = [], [], []
    for i, it in enumerate(items):
        tid = it["type_id"]
        fam, ex = C.parse_type_id(tid)
        ts = types.get(tid)
        ckpts = list(ts.ckpts)
        t_arr = t0_ns + it["t_arrival_ns"]
        dl = None if it["t_deadline_ns"] is None else t0_ns + it["t_deadline_ns"]
        k = C.k_of(spec, ts.klass)
        t = t_arr
        while True:
            live = [d for d in busy if d > t]
            if len(live) < k:
                break
            t = min(live)
        row = {"condition_id": condition_id, "req_id": it["req_id"], "type_id": tid, "family": fam, "exit": ex,
               "klass": ts.klass, "target": ckpts[-1], "t_arrival_ns": t_arr,
               "t_arrival_actual_ns": t_arr + 100_000 + 20_000 * (i % 3), "t_deadline_ns": dl}
        queue_len = sum(1 for e in left_queue if e > t_arr)
        if dl is not None and t >= dl:
            left_queue.append(dl)
            row.update({"t_admit_ns": None, "t_srv_recv_ns": None, "t_start_ns": None, "t_end_ns": None,
                        "t_done_ns": dl, "n_in_service_at_admit": None, "reached": None, "completion": 0.0,
                        "outcome": "expired_in_queue", "ckpt": []})
        else:
            left_queue.append(t)
            tau_ns = round(ts.tau_s[f_eff]["p50"] * 1e9 * (1.0 + 0.45 * len(live)) * (1.0 + synth_noise(i, seed))
                           * stress.get(i, 1.0))
            service = tau_ns - RECV_NS - START_NS - DONE_NS
            t_start = t + RECV_NS + START_NS
            jit = 1.0 + _FRAC_JITTER[i % 5]
            times = []
            if i in errors:
                reached, outcome, t_end = None, "error", t_start + 1_000_000
            else:
                for c in ckpts:
                    times.append(t_start + round(service * ts.ckpt_frac[c] * jit))
                    if dl is not None and times[-1] > dl:
                        break
                reached, outcome = C.truncation_outcome(ckpts, times, dl)
                if outcome == "complete":
                    t_end = t_start + service
                elif outcome == "truncated":
                    t_end = times[-1] + round(service * ts.head_frac)
                else:
                    t_end = times[-1]
            t_done = t_end + DONE_NS
            busy.append(t_done)
            row.update({"t_admit_ns": t, "t_srv_recv_ns": t + RECV_NS, "t_start_ns": t_start, "t_end_ns": t_end,
                        "t_done_ns": t_done, "n_in_service_at_admit": len(live), "reached": reached,
                        "completion": round(C.completion(fam, ex, reached), 6), "outcome": outcome,
                        "ckpt": [[c, tt] for c, tt in zip(ckpts, times)] if outcome != "error" else []})
        row.update({"server": fam, "policy": spec.name, "policy_spec": spec.to_json(),
                    "f_mhz": spec.f_mhz if spec.f_rule == "static" else "governor", "queue_len_at_arrival": queue_len,
                    "tokens_done": None, "is_warmup": i < n_warmup, "seed": seed, "mode": "open",
                    "err": errors.get(i) if row["outcome"] == "error" else None})
        rows.append({key: row[key] for key in C.REQUEST_KEYS})
    return rows


def synthetic_open_session(root, plan, freeze, types, *, session_id=SESSION_OPEN_ID, dirname=None, stress=None,
                           errors=None):
    """Write a synthetic Run-2 session that executes `plan` (kind 'open', e.g. fixtures/plan_run2.json) for `freeze`.

    The idle windows, server start-up and every condition follow plan.session. Conditions run in plan order; the
    wall clock starts after freeze.created_wall_ns. `stress` {policy_key: {request index: factor}} stretches single
    requests so some of them truncate, miss or make their neighbours expire (default OPEN_STRESS); `errors`
    {policy_key: {request index: message}} makes single requests fail (default OPEN_ERRORS)."""
    stress = OPEN_STRESS if stress is None else stress
    errors = OPEN_ERRORS if errors is None else errors
    sess = plan["session"]
    T0 = MONO0_NS

    def ns(s):
        return T0 + round(s * 1e9)

    idle_s, settle_s = sess["idle_window_s"], sess["settle_s"]
    tl = Timeline(P_BASE_MW, LOAD_BASE_PERMILLE, 1020)
    pre = (ns(1.0), ns(1.0 + idle_s))
    tl.add(*pre, mw=P_IDLE_PRE_MW)
    servers, t_spawn = {}, 1.0 + idle_s + OPEN_SPAWN_AFTER_S
    for pid, fam in enumerate(sess["families"], start=4242):
        t_ready = t_spawn + 1.4
        tl.add(ns(t_spawn), ns(t_ready), mw=P_LOAD_MW[fam], load=150)
        servers[fam] = {"pid": pid, "t_spawn_ns": ns(t_spawn), "t_ready_ns": ns(t_ready),
                        "ready": make_ready(fam, sess["exits"][fam], pid=pid, t_ready_ns=ns(t_ready) - 500_000)}
        t_spawn = t_ready + 0.2
    res = (ns(t_spawn + 0.5), ns(t_spawn + 0.5 + idle_s))
    tl.add(*res, mw=P_IDLE_RES_MW[1020])
    idle = {"pre": {"t0_ns": pre[0], "t1_ns": pre[1], "gpu_mhz": 1020},
            "resident": {"t0_ns": res[0], "t1_ns": res[1], "gpu_mhz": 1020}}

    fconds = {c["condition_id"]: c for c in freeze["conditions"]}
    conds, t_c = [], res[1] + 500_000_000
    for pc in plan["conditions"]:
        fz = fconds[pc["freeze_condition_id"]]
        spec = C.PolicySpec.from_json(pc["policy"])
        f_eff = spec.f_mhz if spec.f_rule == "static" else types.f_max_mhz
        run0 = t_c + round(settle_s * 1e9)
        rows = open_rows(pc["condition_id"], spec, pc["workload"], types, t0_ns=run0, n_requests=pc["n_requests"],
                         n_warmup=pc["n_warmup"], seed=pc["seed"], alpha=pc["alpha"],
                         stress=stress.get(pc["policy_key"]), errors=errors.get(pc["policy_key"]))
        end = max(r["t_done_ns"] for r in rows) + 1_000_000_000
        spans = [(r["t_admit_ns"], r["t_end_ns"]) for r in rows if r["t_admit_ns"] is not None]

        def n_busy(t, spans=spans):
            return sum(1 for a, b in spans if a <= t < b)

        tl.add(t_c, end + 500_000_000, gpu=f_eff)
        tl.add(t_c, run0, mw=P_IDLE_RES_MW[f_eff])
        tl.add(run0, end, mw=lambda t, f=f_eff, nb=n_busy: P_IDLE_RES_MW[f] + DP_MW[f] * min(1.0, 0.55 * nb(t)),
               load=lambda t, nb=n_busy: min(990, LOAD_BASE_PERMILLE + 450 * nb(t)))
        mani = make_manifest(
            pc["condition_id"], "open", t_wall_start_ns=WALL_OPEN_NS + (t_c - T0), t_mono_start_ns=t_c,
            t_mono_end_ns=end, settle_s=settle_s, arm=pc["arm"], gpu_mhz=pc["gpu_mhz"], n_requests=len(rows),
            n_warmup=pc["n_warmup"], seed=pc["seed"], policy_key=pc["policy_key"], policy_spec=pc["policy"],
            freeze_condition_id=pc["freeze_condition_id"], workload=pc["workload"], alpha=pc["alpha"],
            repeat=pc["repeat"])
        switch = [[t_c + 10_000_000, t_c + 90_000_000, f_eff, f_eff]] if pc["arm"] == "pinned" else []
        conds.append({"manifest": mani, "requests": rows, "freq_switch": switch})
        t_c = end + 500_000_000
    return write_session(root, conds, plan=plan, servers=servers, idle=idle, timeline=tl, freeze=freeze,
                         t_end_ns=t_c + 500_000_000, session_id=session_id, dirname=dirname,
                         t_wall_start_ns=WALL_OPEN_NS, power_hz=5, clocks_hz=1)
