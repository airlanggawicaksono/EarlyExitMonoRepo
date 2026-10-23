"""exp2 contracts: the single definition of every name, schema, validator and shared rule (BRIEF_v1 sec. 3.3-3.4).

stdlib only (no numpy). Python 3.10 syntax. Change this file first, then the code that uses it.

Units
    *_ns        int from time.monotonic_ns(); the only exception is *_wall_ns (time.time_ns()).
    *_s         float seconds.        GPU f: int MHz.        EMC: float MHz.
    JSON null <-> K_INF for k (concurrency limits): PolicySpec stores K_INF, to_json() writes null.

Truncation (K7), one definition
    default_checkpoints(), truncation_outcome(), completion(). A segment that crosses the deadline still runs
    on the GPU; its checkpoint time is recorded, then the server stops. The head for `reached` runs after the
    decision and is not charged to the deadline. expired_in_queue and miss have completion 0.

Admission semantics (K4; implemented in exp2/policies.py by S1b, used by dispatcher AND simulator)
    Policy(spec, types, *, family_exclusive=True).admit(view) -> list[req_id]; pure and deterministic.
    n starts at len(in_service) and grows with every admit inside the call. Candidate order:
        fcfs: arrival order | stf: tau p50 at f_max ascending, then arrival | class: short first, then arrival.
    r is admitted iff: its family is not busy (including families admitted earlier in the same call; skipped when
    family_exclusive=False) AND n < k_limit(spec, r) AND (lane_cap is None or in-service of r's class < cap).
    A request that cannot be admitted never blocks the ones behind it.
    edgeserving: candidates are only the head of each per-type queue. For a head h:
        S(h) = sum_{r != h, r queued} min(exp((now - a_r + tau_h^p95(f)) / D_r - 1), C),
        D_r = (t_deadline - t_arrival) in seconds,
        f = f_current_mhz or f_max if None. Admit the h with the smallest S; tie -> earliest arrival.
        Edgeserving needs deadlines on every queued request.
    desired_f(view): None for static/governor.
        greenllm: the lowest corun level for which every queued r satisfies
            (now - a_r) + sum_{q ahead of r in its lane, including r} tau_q^p50(f) + residual(f) <= D_r,
            residual(f) = tau_in(f) * max(0, 1 - (now - t_admit) / tau_in(f_current)); none feasible -> f_max.
            (lane = request class; D_r = absolute t_deadline - a_r.)
        occupancy: the smallest corun level >= f_base * n_sys ** (1 / alpha_p); none -> the highest level.
            n_sys = len(queue) + len(in_service).
        Empty queue (greenllm): every condition holds vacuously, so the answer is the lowest corun level.
    AdmissionCore(policy, types, f_initial_mhz): the ONE state-to-view object. API arrive(req, now_ns),
    done(req_id, now_ns), view(now_ns); each call returns Decision(admitted, expired, commanded_f,
    n_in_service_at_admit). Per event: apply event; drop queued requests with now_ns >= t_deadline_ns
    (outcome expired_in_queue, t_done_ns = t_deadline_ns); policy.admit(view); policy.desired_f(view) AFTER the
    admit, and a non-None value that differs from the last command becomes the new command. f_current_mhz in the
    view is the last COMMANDED level, never a read-back.

Exit choice (CoExit pilot; implemented in exp2/exits.py). A spec with params.exit_rule may run a request at another
exit of its family. The policy then offers admit_exits(view) -> [(req_id, run_type_id)]; AdmissionCore calls it
instead of admit(), records the pairs in Decision.run_type_ids (aligned with admitted; () when every admit runs its
queued type_id) and puts InServiceReq(rid, run_type_id, family, klass, now_ns, t_deadline_ns) in service. A run
type of another family is a RuntimeError. Policies without admit_exits are unchanged.
    ExitPolicy(spec, types, slowdown=None).admit_exits(view) -> [(req_id, run_type_id)]; admit() = the req_ids;
    desired_f() = None; family_exclusive = True. tau(t) = types.tau(t, f, params.tau_stat or "p95"),
    f = view.f_current_mhz or f_max. floor/target type of family m = "m:EXIT_MENU[m][0|1]"; a family outside
    EXIT_MENU -> ContractError. s_t(A) = slowdown.predict(t, A, f) (memoised; A = run types incl. t).
    Candidates (target/floor/greedy/coexit): view.queue in arrival order; skip a request whose family has a job in
    service (incl. admitted earlier in this call). A skipped request never blocks the ones behind it.
        target: run the target type.   floor: run the floor type.
        greedy: target if now + tau(target) * s_target(A + target) <= deadline, else floor
                (A = run types in service incl. this call's admits).
        coexit: capacity gate first: if >= 1 job is in service (admits earlier in the call count) and the predicted
                capacity of A = run types in service + target(r), sum over A of 1/s_t(A), is below
                1 + params.cap_margin (default 0; pilot_specs uses COEXIT_CAP_MARGIN), r WAITs (serial: the
                predicted co-run gain is not larger than the slowdown predictor's error). Otherwise:
                options = [target if r is on time in rollout(target)] + [floor] + [WAIT if >= 1 job in service];
                L(o) = number of predicted-late requests (in service + r + queued) in rollout(o);
                choose the first option in that order with L(o) == min L. WAIT -> r is not admitted now, and r
                keeps its family's turn: no later queued request of that family is a candidate in this call (FIFO
                per family, as in the other methods). "Jobs in service" for WAIT include admits earlier in the
                call. A request without a deadline is never late; k_override is ignored by exit policies.
        rollout(o): deterministic fluid look-ahead, no future arrivals.
            in service: w = max(0, tau(t) - (now - t_admit)/s_t(A0)), A0 = run types of view.in_service;
                        jobs admitted earlier in this call: w = tau(t).
            r: target/floor -> starts now with w = tau(type); WAIT -> r heads its family queue and that family is
               blocked until the first completion in the rollout.
            queued (all other queued, FIFO per family): each starts at its FLOOR exit at the first instant its
               family has no active job (families started in order of head arrival).
            Between events every active job advances 1/s_t(A) solo-seconds per second; A = run types active.
            Late: predicted finish > t_deadline_ns (no deadline -> never late).
        edgeserving (k = 1, only when nothing is in service, at most one admit): for each family head h (oldest
            queued of that family): e*(h) = deepest exit with now + tau(e) <= t_deadline(h), else floor;
            S(h) = sum over the other queued q of min(exp((now - a_q + tau(e*(h))) / D_q - 1), C),
            D_q = t_deadline_q - a_q; admit argmin S (tie -> earliest arrival) at e*(h). (EdgeServing Sec. V.C
            eq. 3-7, B = 1, eq. 6 per request deadline; the fallback to floor is ours, the paper states none.)
    Pilot metrics (exp2/model/coexit_pilot.py): success = outcome complete and t_done_ns <= t_deadline_ns (the head
    counts); slo_violation_ratio = 1 - successes / n over non-warm-up requests (expired, truncated, miss and late
    complete all fail); R = t_done - t_arrival, R = D for expired; p95/p99 by stats.p_nearest_rank; normalized
    depth = depth(run type) / depth(target type) for a success and 0 for a failure; share_floor = floor runs /
    admitted requests.
    Exit choice on the board (Run 3). The dispatcher runs Decision.run_type_ids: on admit, type_id/exit/target of the
    request row become the RUN type, and the row keeps its arrival type in the optional key req_type_id (present
    exactly on open rows of an exit policy; same family; a row that never ran keeps type_id == req_type_id). Run 2
    rows are unchanged. A Run 3 plan condition may carry deadlines_s {family: seconds} (alpha null); the dispatcher
    sets t_deadline_ns with with_family_deadlines. Run 3 analysis (exp2.analyze.run3) uses the Pilot metrics.

Session layout (K9). A condition id is a directory name, so it matches [A-Za-z0-9_][A-Za-z0-9_.-]* (no ':')
    <out>/<session_id>/session.json
    <out>/<session_id>/telemetry/{power.csv,clocks.csv,meta.json}
    <out>/<session_id>/servers/server_<family>.{port,log}
    <out>/<session_id>/<condition_id>/{manifest.json,requests.jsonl,freq_switch.csv}

CLI table (K19; frozen here)
    server       python -m exp2.serve.server --family F --exits 5,23 --adapter real|fake [--fake-spec P] --out DIR
                 [--preprocess per_request|preload] [--n-inputs 32] [--warmup-iters 20] [--no-checkpoints]
                 [--sync event|spin] [--seed 0]                                           exit 0, 2
    run_session  python -m exp2.serve.run_session --plan P --types T --out DIR [--adapter real|fake]
                 [--fake-spec P] [--dry-run] [--ready-timeout-s 1800]                     exit 0, 2 usage, 3 freeze
                 refused (an open plan also exits 3 when sha256(--types) != freeze.inputs.types_sha256, before any
                 spawn), 5 server died or timeout (ClockError -> manifest valid=false, session continues)
    telemetry    python -m exp2.board.telemetry --out DIR [--hz 20] [--clocks-hz 1] [--duration S]
                 [--sysfs-root /] [--no-tegrastats]                                       exit 0, 2
    clocks       python -m exp2.board.clocks [--dry-run --fake-root DIR] snapshot|set-gpu MHZ|lock-emc MHZ|
                 apply-arm ARM [--gpu MHZ] [--emc MHZ]                                    exit 0, 2, 4 (ClockError)
    closed_loop  python -m exp2.analyze.closed_loop --sessions S [S..] --out DIR          exit 0, 2
                 (also writes screen.csv; see "Run 1 variants and screening")
    open_loop    python -m exp2.analyze.open_loop --sessions S [S..] --freeze F --out DIR exit 0, 2, 3
    plan_run1    python -m exp2.model.plan_run1 --exits bert=5,23 vision=6,24 llama=3,15 yolo=1,5
                 --solo-levels 408,612,816,1020 --heldout-level 714 --corun-levels 612,1020 --repeats 3
                 --heldout-frac 0.3 --duration 30 --seed 0 --out P                        exit 0, 2
    build_types  python -m exp2.model.build_types --run1-csv DIR --alpha 5 --out T        exit 0, 2
    slowdown     python -m exp2.model.slowdown --run1-csv DIR --types T --out M.json      exit 0, 2
    power        python -m exp2.model.power --run1-csv DIR --types T --slowdown M.json --out P.json   exit 0, 2
    freeze       python -m exp2.model.freeze build --types T --slowdown M --power P --conditions C --out DIR
                 | verify --freeze F                                                      exit 0, 2, 3
    plan_run2    python -m exp2.model.plan_run2 --freeze F --out P                        exit 0, 2
    freeze_build python -m exp2.model.freeze_build --run1-csv DIR --types T --slowdown M --power P
                 --conditions C --run1-plan P1 [--run1b-plan P2] --n-requests N --n-warmup W --n-reps R
                 --seed S [--workers K] --out DIR                                         exit 0, 2
    pilot        python -m exp2.pilot.pilot [--dry-run] --out DIR [--steps 0,1,...]
                 [--config exp2/configs/pilot.json]                                       exit 0, 1 gate failed, 2

Run 1 variants and screening (D4, D7, AM1)
    Several closed sessions (Run 1, Run 1b, Run 1d) are analysed together. Repeats of a later session are shifted
    by the count already present for (type, f) (solo), so repeat indices stay unique. The co-run baseline
    tau_solo_p50 comes from the same session when it has a solo row at (type, f), else it is pooled over sessions.
    U_A is pooled. Run 1c (D7, cold start) empties the inductor, triton and CUDA caches and drops the page cache
    (applied in run1c.sh) before each cold measurement.
    K_SET = (1, 2, K_INF): the concurrency limits searched and simulated. With 3 resident families k=3 admits the
    same set as K_INF, so K_SET is complete.
    Screening: the window of a closed condition in clocks.csv is [t_mono_start + settle_s, t_mono_end].
    gpu_lock_frac = share of rows with |gpu_cur_hz / 1e6 - gpu_mhz| <= SCREEN_LOCK_TOL_MHZ (pinned arms only);
    emc_lock_frac likewise on emc_mhz (only if logged). screen_ok = 1 iff GPU lock >= SCREEN_GPU_LOCK_MIN (pinned)
    and EMC lock >= SCREEN_EMC_LOCK_MIN (if logged). max_tj_c = max of SCREEN_TJ_COL / 1000 (else the max of all
    temp_ columns). Swap and pgmajfault columns are empty for old clocks.csv. A skipped condition is written to
    screen.csv with kept=0 and a reason.
    AM1: a closed condition (solo or co-run) in which any tenant has fewer than MIN_MEASURED_REQ_PER_TENANT measured
    (non warm-up) requests inside its window is skipped with reason "few_requests:<type_id>=<n>" (see
    few_requests_reason), never raised. Rows that fail AM1 are dropped before the later-session repeat shift.

Run 2 plan (D11)
    Measured keys are the freeze choices in order. A choice equal to an earlier one is an alias of it; a grid point
    equal to a choice is that choice's alias. A measured key takes the max of REPEATS over its aliases
    (REPEATS: F, default, default_pinned, corun, serial, serial_stf, solo_profile_freq = 3; every other key 1).
    Blocks b = 0..N_BLOCKS-1: a 3-repeat key puts repeat r in block r; the 1-repeat runs are shuffled and dealt
    round-robin to blocks 0, 1, 2; each block is permuted with numpy PCG64(order_seed); blocks run in order.
    order_seed = freeze.sim.seed + ORDER_SEED_OFFSET; the arrival seed stays sim.seed + ARRIVAL_SEED_OFFSET +
    repeat (paired across keys). plan.order_seed (int >= 0) is optional and written only by open plans.
    freeze.measured_runs(cond) -> [(key, spec_json, n_rep)]; freeze.measured_key_map(cond) -> {every choice key
    and grid name: measured key}; policy_list stays [(key, spec)] taken from measured_runs.
    freeze.inputs may carry run1b_plan_sha256 (hex); freezes without it stay valid.

Power model (K13 addendum)
    fit_power uses rows with split != "heldout". eta(f) = params.eta of the slowdown model if present, else eta
    from M2 fitted on the train rows. PowerModel.from_json(obj, types) rebuilds a model from to_json().

Open summary (C2b, exp2.open_summary/2, extended by /3 below)
    open_loop reads several sessions (--sessions S [S..] --freeze F [--plans P [P..]] --out DIR). Wilson 95% (z 1.96)
    on the pooled per-type complete counts: status feasible if min over types of the lower bound >= COMPLETE_TARGET,
    borderline if only the point estimate is, else infeasible. Regret candidates = measured keys of F and of every
    feasible grid point (freeze.measured_key_map). energy_diff["<fcid>_<key>"] = mean_ci of E_F,r - E_key,r over
    repeats (method "paired_t"), keys with >= 2 repeats. Overshoot = (t_done - t_deadline) / 1e9 for non-warmup
    complete, truncated and miss rows with t_done > t_deadline; late_rate = their share of the type's non-warmup
    rows; overshoot_* are empty when no row is late. counts: runs_valid, runs_invalid, runs_duplicate (same
    condition_id in an earlier session, skipped), runs_screen_flagged (screen_ok = 0).

R32 design decisions (DECISIONS_R32.md, frozen before any slowdown fit or Run 2 simulation)
    constant                     value                         decision
    R_SIM                        10                            D32-1  simulation replications per policy
    F_FEAS_ONE_SIDED_LEVEL       0.95                          D32-1  F is feasible for a type only if mean - t_{0.95,R-1}
                                                                      * sd / sqrt(R) of the predicted complete rate >= 0.95
                                                                      for every type; an infeasible measured F is a failure
                                                                      of H4, never "regret undefined"
    RUN2_N_MEASURED, RUN2_N_WARMUP 500, 50                       D32-2  requests per run after warm-up
    GRID_LEVELS_MHZ              (612, 1020)                   D32-2  levels of single-run grid points (F and the
                                                                      frequency baselines still use the co-run levels)
    SESOI_LOG_ENERGY             0.0513                        D32-4  |log ratio| = 5% energy, smallest effect of interest
    PRIMARY_REFS                 default_pinned, serial_stf    D32-4  two primary tests, Holm-adjusted
    SECONDARY_REFS               default, serial,              D32-4  Holm within the secondary family
                                 solo_profile_freq
    HOLM_ALPHA                   0.05                          D32-4
    BOOT_BLOCK_REQUESTS          50                            D32-6  moving-block length (requests) inside a run
    BOOT_RESAMPLES, BOOT_SEED    2000, 20261020                D32-6  runs are resampled, then blocks inside each run
    MIN_REQ_SENSITIVITY          5                             D32-7  AM1 sensitivity refit (MIN_MEASURED_REQ_PER_TENANT 10)
    GAME_TIE_TOL                 0.005                         D32-8  Nash tolerance tau (absolute shortfall)
    SCREEN_TJ_FLAG_C             85.0                          D32-9  max_tj_c above this is flagged (report, never drop)
    CONFIRM_REPEATS              (1, 2)                        D32-5  repeats of a Run 2b confirm run

Open summary v3 (exp2.open_summary/3, D32-4, D32-5, D32-6). validate_open_summary also accepts /2 documents
    (no new keys); schema /3 requires all of the following in addition to the /2 keys.
    primary    {ref: stat} with ref exactly PRIMARY_REFS; secondary {ref: stat} with ref a subset of SECONDARY_REFS.
               stat = {log_ratio_mean, ci_lo, ci_hi, p, p_holm, n_pairs}: paired log(E_F / E_ref) pooled over
               conditions x paired repeats (15 pairs in Run 2), t interval, ci_lo <= ci_hi, p and p_holm in [0, 1],
               n_pairs int >= 2. p_holm is Holm-adjusted inside the family at HOLM_ALPHA.
    per_condition[<fcid>_<key>] adds attain_boot_lo_min (2.5% quantile of the per-type complete rate from the
               block bootstrap, min over types, in [0, 1]) and energy_per_work_j (board energy divided by the summed
               reached-checkpoint depth relative to target; null when undefined). status is now decided by the bootstrap
               rule: feasible if attain_boot_lo_min >= COMPLETE_TARGET, borderline if only the point estimate is,
               else infeasible. wilson_lo_min stays as a secondary column.
    sim_accuracy {fcid: {abs_err_pp_max, abs_err_pp_median, class_agreement, spearman_energy, energy_ape_median}}
               abs_err_pp_* = absolute error of the per-type complete rate in percentage points (>= 0);
               class_agreement = share of measured grid keys whose feasibility class matches the prediction, in [0, 1];
               spearman_energy in [-1, 1] or null; energy_ape_median >= 0 or null. The governor default is excluded.

Confirm plan (D32-5, Run 2b). A confirm plan is an open plan (kind "open"). Its conditions carry "confirm_of" (the
    measured key that is re-measured) and repeat in CONFIRM_REPEATS (arrival seeds of repeats 1 and 2).

Slowdown selection (D32-7). The slowdown json may carry "selection" = {"method": "loco", "loco_mdape": {model: MdAPE >= 0},
    "selected": name}. loco = leave-one-cell-out over the training cells (a cell with all its levels left out together).
    selected is the simplest model whose LOCO median absolute percentage error of s_p50 is <= 10% and below M0; it equals
    the json "name".

Back-test (exp2.backtest/1, D32-6). backtest.json = {schema, cells: [{cell_id, f_mhz, type_id, pred_tau_p50_s,
    meas_tau_p50_s, ape}], energy: [{cell_id, f_mhz, pred_j, meas_j, ape}], summary: {tau_mdape, energy_mdape}}. Held-out
    Run 1 cells simulated in closed loop with the fitted slowdown and power models. ape = 100 * |pred - meas| / meas
    (percent, as open_summary ape); summary values are medians of ape (null if the list is empty).

Game (D32-8). Freeze condition.game keeps its keys and optionally gains the group equilibria (list of profiles
    {k_short, k_long}; legacy best-response entries keep start/converged/iters/cycle), optimum_constrained (profile or
    null), poa_eps (number or null) and tie_tol (number >= 0, GAME_TIE_TOL); optimum_constrained, poa_eps and tie_tol
    are all-or-none. All 9 profiles of K_SET x K_SET are enumerated; a pure Nash equilibrium is a profile where no class
    lowers its cost by more than tie_tol by a unilateral threshold change. poa_eps = energy of the worst equilibrium /
    lowest energy over profiles meeting the 95% constraint; null if no profile meets it.

Class rule (D32-8). types.provenance.klass_rule: sort types by solo median service time at f_max; the split is at the
    largest ratio between consecutive sorted types (Run 1: 77 ms -> 640 ms). Types below the split are "short", the rest
    "long".

JSON shapes decided in S0a where BRIEF_v1 left them open (all documented at the validator):
    PolicySpec.lane_cap = {"short": int>=1, "long": int>=1} or null.
    condition_id: [A-Za-z0-9_][A-Za-z0-9_.-]* (brief examples with ':' become '-': solo_bert-5_f1020_r0).
    freeze condition: choices[key] and grid[i] are PolicySpec json; predictions[key] holds "per_type" and
    "energy_per_req_j" (a SimResult json qualifies); plan.sha256 = plan_sha256(plan).
    SimResult.tagged = TypeStats-shaped dict plus "klass" and "k_dev", or null.
    pilot_summary carries "schema": "exp2.pilot/1".
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

# --------------------------------------------------------------------------------------------- constants
FAMILIES = ("bert", "vision", "llama", "yolo")
EXIT_RANGE = {"bert": (0, 23), "vision": (1, 24), "llama": (0, 15), "yolo": (0, 5)}   # inclusive
YOLO_EXIT_MAX_DEPTH = {0: 7, 1: 8, 2: 9, 3: 15, 4: 18, 5: 21}   # ported from AnyTimeYolo/src/benchmark.py:51
LLAMA_TOKENS, LLAMA_PROMPT_TOKENS, CKPT_STRIDE = 32, 128, 4
K_INF = 1_000_000                       # JSON null <-> K_INF
K_SET = (1, 2, K_INF)                   # concurrency limits searched; k=3 with 3 resident families == K_INF
REPEATS = {"F": 3, "default": 3, "default_pinned": 3, "corun": 3, "serial": 3, "serial_stf": 3,
           "solo_profile_freq": 3}      # D11: repeats per measured key; every other key runs once
N_BLOCKS = 3                            # D11
ORDER_SEED_OFFSET, ARRIVAL_SEED_OFFSET = 23, 11   # D11: order_seed = sim.seed + 23; arrival = sim.seed + 11 + repeat
MIN_MEASURED_REQ_PER_TENANT = 10        # AM1
SCREEN_GPU_LOCK_MIN, SCREEN_EMC_LOCK_MIN, SCREEN_LOCK_TOL_MHZ = 0.99, 0.99, 1.0
SCREEN_TJ_COL = "temp_tj_thermal_mc"
SCREEN_TJ_FLAG_C = 85.0                 # D32-9
R_SIM = 10                              # D32-1
F_FEAS_ONE_SIDED_LEVEL = 0.95           # D32-1
GRID_LEVELS_MHZ = (612, 1020)           # D32-2
RUN2_N_MEASURED, RUN2_N_WARMUP = 500, 50   # D32-2
SESOI_LOG_ENERGY = 0.0513               # D32-4
PRIMARY_REFS = ("default_pinned", "serial_stf")   # D32-4
SECONDARY_REFS = ("default", "serial", "solo_profile_freq")   # D32-4
HOLM_ALPHA = 0.05                       # D32-4
BOOT_BLOCK_REQUESTS, BOOT_RESAMPLES, BOOT_SEED = 50, 2000, 20261020   # D32-6
MIN_REQ_SENSITIVITY = 5                 # D32-7
GAME_TIE_TOL = 0.005                    # D32-8
CONFIRM_REPEATS = (1, 2)                # D32-5
CLASSES = ("short", "long")
ORDERS = ("fcfs", "stf", "class", "edgeserving")
F_RULES = ("static", "governor", "greenllm", "occupancy")
ARMS = ("pinned", "governor", "stock")
OUTCOMES = ("complete", "truncated", "miss", "expired_in_queue", "error")
EMC_LOCKABLE_MHZ = (204.0, 665.6, 2133.0, 3199.0)
POWER_MODE_ID, POWER_MODE_NAME = 2, "MAXN_SUPER"
SUDO_PW_ENV = "JETSON_SUDO_PW"          # read at run time only; never written to a file
COMPLETE_TARGET = 0.95
OPEN_STATUSES = ("feasible", "borderline", "infeasible")   # C2b: Wilson 95% lower bound vs COMPLETE_TARGET
TIE_SHORTFALL = 0.005
UNSTABLE_SLOPE_FRAC = 0.005
EDGESERVING_C = 10.0
EXIT_MENU = {"bert": (5, 23), "yolo": (1, 5), "llama": (3, 15)}   # CoExit pilot: (floor, target) exit per family
EXIT_RULES = ("target", "floor", "edgeserving", "greedy", "coexit")
COEXIT_CAP_MARGIN = 0.053   # held-out MAPE of the selected slowdown model M1 (Run 1 back-test): smallest co-run gain

SCHEMAS = {
    "types": "exp2.types/1", "plan": "exp2.plan/1", "freeze": "exp2.freeze/1", "session": "exp2.session/1",
    "manifest": "exp2.manifest/1", "simresult": "exp2.simresult/1", "slowdown": "exp2.slowdown/1",
    "power": "exp2.power/1", "open_summary": "exp2.open_summary/3", "pilot": "exp2.pilot/1",
    "fake_spec": "exp2.fake_spec/1", "telemetry_meta": "exp2.telemetry_meta/1",
    "backtest": "exp2.backtest/1",
}
OPEN_SUMMARY_V2 = "exp2.open_summary/2"   # still accepted by validate_open_summary
SCHEMA_IDS = frozenset(SCHEMAS.values())

_STATS = ("p50", "mean", "p95")
_SPLITS = ("solo", "solo_heldout_level", "train", "heldout")
_MODES = ("open", "closed", "idle")
_ARRIVALS = ("poisson", "gamma")
_PREPROCESS = ("per_request", "preload")
_COMPILE = ("on", "eager_fallback", "off")
_SLOWDOWN_NAMES = ("M0", "M1", "M2", "M3", "M4lin", "M4")
_HEX64 = re.compile(r"[0-9a-f]{64}")
_TEMP_COL = re.compile(r"temp_[a-z0-9_]+_mc")
_SIDECAR = re.compile(r"([0-9a-f]{64})  (\S.*?)\n?")
_TYPE_ID = re.compile(r"([a-z]+):(\d+)")
_COND_ID = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.-]*")


# --------------------------------------------------------------------------------------------- exceptions
class ContractError(ValueError):
    """A value violates a contract (schema, enum, range, cross-field rule)."""


class ClockError(RuntimeError):
    """A board clock/power-mode write was refused or did not read back."""


class FreezeError(RuntimeError):
    """freeze.json (or its sidecar) does not match its hash, or is unreadable."""


class HeldOutError(ValueError):
    """A slowdown fit received a row with split == 'heldout'."""


# --------------------------------------------------------------------------------------------- check helpers
def _is_int(v) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _is_num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _fail(where: str, msg: str):
    raise ContractError(f"{where}: {msg}")


def _dict(v, where, null=False):
    if v is None and null:
        return None
    if not isinstance(v, dict):
        _fail(where, f"expected object, got {type(v).__name__}")
    return v


def _list(v, where, null=False, min_len=0):
    if v is None and null:
        return None
    if not isinstance(v, (list, tuple)):
        _fail(where, f"expected list, got {type(v).__name__}")
    if len(v) < min_len:
        _fail(where, f"expected at least {min_len} item(s), got {len(v)}")
    return v


def _int(v, where, lo=None, hi=None, null=False):
    if v is None and null:
        return None
    if not _is_int(v):
        _fail(where, f"expected int, got {type(v).__name__} {v!r}")
    if lo is not None and v < lo:
        _fail(where, f"{v} < {lo}")
    if hi is not None and v > hi:
        _fail(where, f"{v} > {hi}")
    return v


def _num(v, where, lo=None, hi=None, null=False, positive=False):
    if v is None and null:
        return None
    if not _is_num(v):
        _fail(where, f"expected finite number, got {type(v).__name__} {v!r}")
    if positive and not v > 0:
        _fail(where, f"{v} is not > 0")
    if lo is not None and v < lo:
        _fail(where, f"{v} < {lo}")
    if hi is not None and v > hi:
        _fail(where, f"{v} > {hi}")
    return v


def _str(v, where, choices=None, null=False):
    if v is None and null:
        return None
    if not isinstance(v, str) or v == "":
        _fail(where, f"expected non-empty string, got {v!r}")
    if choices is not None and v not in choices:
        _fail(where, f"{v!r} not in {tuple(choices)}")
    return v


def _bool(v, where, null=False):
    if v is None and null:
        return None
    if not isinstance(v, bool):
        _fail(where, f"expected bool, got {type(v).__name__} {v!r}")
    return v


def _hex(v, where, null=False):
    if v is None and null:
        return None
    if not isinstance(v, str) or not _HEX64.fullmatch(v):
        _fail(where, f"expected 64 lowercase hex chars, got {v!r}")
    return v


def _keys(obj, required, where, optional=()):
    """Exact key set: every required key present, nothing outside required + optional."""
    _dict(obj, where)
    missing = [k for k in required if k not in obj]
    extra = [k for k in obj if k not in required and k not in optional]
    if missing:
        _fail(where, f"missing key(s) {missing}")
    if extra:
        _fail(where, f"unexpected key(s) {extra}")


def _schema(obj, kind, where):
    if obj.get("schema") != SCHEMAS[kind]:
        _fail(where, f"schema must be {SCHEMAS[kind]!r}, got {obj.get('schema')!r}")


def _int_key(k, where):
    try:
        return int(k)
    except (TypeError, ValueError):
        _fail(where, f"key {k!r} is not an integer string")


def _level_map(v, where, levels=None, lo=None, hi=None, positive=False, required=()):
    """{ "1020": number, ... } with integer-string keys. `required` levels must all be present."""
    _dict(v, where)
    for k, x in v.items():
        f = _int_key(k, where)
        if levels is not None and f not in levels:
            _fail(where, f"level {f} is not in f_levels_mhz {sorted(levels)}")
        _num(x, f"{where}[{k}]", lo=lo, hi=hi, positive=positive)
    have = {int(k) for k in v}
    for f in required:
        if f not in have:
            _fail(where, f"missing level {f}")


def _cond_id(v, where):
    """A condition id is also a directory name (<out>/<session>/<condition_id>/): no ':' (illegal on Windows),
    no separators or spaces, and it does not start with '.' or '-'."""
    _str(v, where)
    if not _COND_ID.fullmatch(v):
        _fail(where, f"condition id {v!r} must match [A-Za-z0-9_][A-Za-z0-9_.-]* (it is a directory name)")
    return v


def _s_to_ns(x: float) -> int:
    return int(round(x * 1e9))


# ------------------------------------------------------------------------------------------ type ids, depth, checkpoi
def _check_family_exit(family, exit):
    if family not in FAMILIES:
        _fail("family", f"{family!r} not in {FAMILIES}")
    if not _is_int(exit):
        _fail("exit", f"expected int, got {exit!r}")
    lo, hi = EXIT_RANGE[family]
    if not lo <= exit <= hi:
        _fail("exit", f"{family} exit {exit} outside {lo}..{hi}")


def type_id(family: str, exit: int) -> str:
    _check_family_exit(family, exit)
    return f"{family}:{exit}"


def parse_type_id(s: str) -> tuple[str, int]:
    m = _TYPE_ID.fullmatch(s) if isinstance(s, str) else None
    if not m:
        _fail("type_id", f"cannot parse {s!r} (expected '<family>:<exit>')")
    family, exit = m.group(1), int(m.group(2))
    _check_family_exit(family, exit)
    return family, exit


def depth(family: str, exit: int) -> int:
    """Backbone depth reached at `exit`: bert/llama exit+1; vision exit; yolo YOLO_EXIT_MAX_DEPTH[exit]+1."""
    _check_family_exit(family, exit)
    if family in ("bert", "llama"):
        return exit + 1
    if family == "vision":
        return exit
    return YOLO_EXIT_MAX_DEPTH[exit] + 1


def default_checkpoints(family: str, exit: int) -> list[int]:
    """Checkpoints a request of this type passes through. Units: exit index (bert, vision, yolo) or token (llama)."""
    _check_family_exit(family, exit)
    if family == "bert":
        return [e for e in range(3, exit, CKPT_STRIDE)] + [exit]
    if family == "vision":
        return [e for e in range(4, exit, CKPT_STRIDE)] + [exit]
    if family == "yolo":
        return list(range(0, exit + 1))
    return list(range(1, LLAMA_TOKENS + 1))


def completion(family: str, exit: int, reached) -> float:
    """Fraction of the full depth that finished: depth(reached)/depth(exit); llama reached/LLAMA_TOKENS; None -> 0."""
    _check_family_exit(family, exit)
    if reached is None:
        return 0.0
    if not _is_int(reached):
        _fail("reached", f"expected int or None, got {reached!r}")
    if family == "llama":
        if not 1 <= reached <= LLAMA_TOKENS:
            _fail("reached", f"llama token count {reached} outside 1..{LLAMA_TOKENS}")
        return reached / LLAMA_TOKENS
    lo, hi = EXIT_RANGE[family]
    if not lo <= reached <= exit:
        _fail("reached", f"{family} reached {reached} outside {lo}..{exit}")
    return depth(family, reached) / depth(family, exit)


def truncation_outcome(ckpts, t_ckpt_ns, deadline_ns):
    """-> (reached | None, outcome in {"complete","truncated","miss"}). See module docstring (K7).

    t_ckpt_ns holds the time of every checkpoint that finished, INCLUDING the first late one (the segment keeps
    the GPU), and nothing after it. deadline None -> every checkpoint must be present. Time == deadline is on time.
    ValueError: entries after the first late one, or a short list with no late entry.
    """
    ckpts, t = list(ckpts), list(t_ckpt_ns)
    if not ckpts:
        raise ValueError("ckpts is empty")
    first_late = None
    if deadline_ns is not None:
        for i, ti in enumerate(t):
            if ti > deadline_ns:
                first_late = i
                break
    if first_late is None:
        if len(t) != len(ckpts):
            raise ValueError(f"{len(t)} checkpoint times for {len(ckpts)} checkpoints and none is late")
        return ckpts[-1], "complete"
    if len(t) > first_late + 1:
        raise ValueError(
            f"{len(t) - first_late - 1} checkpoint(s) recorded after the first late one (index {first_late})")
    if first_late == 0:
        return None, "miss"
    return ckpts[first_late - 1], "truncated"


# --------------------------------------------------------------------------------------------- dataclasses
@dataclass(frozen=True)
class TypeSpec:
    """One request type. tau_s {f_mhz: {"p50","mean","p95","samples"}}; ckpt_frac {ckpt: frac of full time};
    u {f_mhz: mean GPU load 0..1}; head_frac = time share of the head (0 for llama)."""
    type_id: str
    family: str
    exit: int
    klass: str
    ckpts: tuple
    tau_s: dict
    ckpt_frac: dict
    head_frac: float
    u: dict
    act_mb: float

    @classmethod
    def from_json(cls, o: dict) -> "TypeSpec":
        return cls(
            type_id=o["type_id"], family=o["family"], exit=o["exit"], klass=o["klass"], ckpts=tuple(o["ckpts"]),
            tau_s={int(f): {"p50": v["p50"], "mean": v["mean"], "p95": v["p95"], "samples": list(v["samples"])}
                   for f, v in o["tau_s"].items()},
            ckpt_frac={int(c): v for c, v in o["ckpt_frac"].items()}, head_frac=o["head_frac"],
            u={int(f): v for f, v in o["u"].items()}, act_mb=o["act_mb"])

    def to_json(self) -> dict:
        return {
            "type_id": self.type_id, "family": self.family, "exit": self.exit, "klass": self.klass,
            "ckpts": list(self.ckpts),
            "tau_s": {str(f): {"p50": v["p50"], "mean": v["mean"], "p95": v["p95"], "samples": list(v["samples"])}
                      for f, v in sorted(self.tau_s.items())},
            "ckpt_frac": {str(c): v for c, v in sorted(self.ckpt_frac.items())}, "head_frac": self.head_frac,
            "u": {str(f): v for f, v in sorted(self.u.items())}, "act_mb": self.act_mb}


@dataclass(frozen=True)
class TypeTable:
    """K1. Build with TypeTable.from_json(obj) (validates) or load_types(path)."""
    alpha: float | None
    f_levels_mhz: tuple
    corun_levels_mhz: tuple
    heldout_level_mhz: int | None
    f_max_mhz: int
    llama_tokens: int
    llama_prompt_tokens: int
    families: dict
    types: tuple
    synthetic: bool = False
    provenance: dict = field(default_factory=dict)
    _index: dict = field(default_factory=dict, init=False, repr=False, compare=False)

    def __post_init__(self):
        for name in ("f_levels_mhz", "corun_levels_mhz", "types"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        object.__setattr__(self, "_index", {t.type_id: t for t in self.types})

    def get(self, type_id: str) -> TypeSpec:
        try:
            return self._index[type_id]
        except KeyError:
            raise ContractError(f"unknown type_id {type_id!r}; known: {sorted(self._index)}") from None

    def type_ids(self) -> list[str]:
        return [t.type_id for t in self.types]

    def tau(self, type_id: str, f_mhz: int, stat: str = "p50") -> float:
        """tau_s[f][stat]; stat in p50|mean|p95. ContractError if the type or level is unknown."""
        if stat not in _STATS:
            _fail("tau", f"stat {stat!r} not in {_STATS}")
        t = self.get(type_id)
        f = int(f_mhz)
        if f not in t.tau_s:
            _fail("tau", f"{type_id} has no tau at {f} MHz (have {sorted(t.tau_s)})")
        return t.tau_s[f][stat]

    @classmethod
    def from_json(cls, o: dict) -> "TypeTable":
        validate_types(o)
        return cls(
            alpha=o["alpha"], f_levels_mhz=tuple(o["f_levels_mhz"]), corun_levels_mhz=tuple(o["corun_levels_mhz"]),
            heldout_level_mhz=o["heldout_level_mhz"], f_max_mhz=o["f_max_mhz"], llama_tokens=o["llama_tokens"],
            llama_prompt_tokens=o["llama_prompt_tokens"], families=dict(o["families"]),
            types=tuple(TypeSpec.from_json(t) for t in o["types"]), synthetic=bool(o.get("synthetic", False)),
            provenance=dict(o["provenance"]))

    def to_json(self) -> dict:
        out = {"schema": SCHEMAS["types"]}
        if self.synthetic:
            out["synthetic"] = True
        out.update({
            "alpha": self.alpha, "f_levels_mhz": list(self.f_levels_mhz),
            "corun_levels_mhz": list(self.corun_levels_mhz),
            "heldout_level_mhz": self.heldout_level_mhz, "f_max_mhz": self.f_max_mhz, "llama_tokens": self.llama_tokens,
            "llama_prompt_tokens": self.llama_prompt_tokens, "families": self.families,
            "types": [t.to_json() for t in self.types], "provenance": self.provenance})
        return out


@dataclass(frozen=True)
class Workload:
    """K2. probs sum to 1; arrival 'poisson' needs cv_a == 1.0, 'gamma' uses shape 1/cv^2, scale mean*cv^2."""
    name: str
    types: tuple
    probs: tuple
    rho: float
    rate_per_s: float
    arrival: str = "poisson"
    cv_a: float = 1.0

    def __post_init__(self):
        object.__setattr__(self, "types", tuple(self.types))
        object.__setattr__(self, "probs", tuple(self.probs))

    @classmethod
    def from_json(cls, o: dict) -> "Workload":
        validate_workload(o)
        return cls(o["name"], tuple(o["types"]), tuple(o["probs"]), o["rho"], o["rate_per_s"], o["arrival"], o["cv_a"])

    def to_json(self) -> dict:
        return {"name": self.name, "types": list(self.types), "probs": list(self.probs), "rho": self.rho,
                "rate_per_s": self.rate_per_s, "arrival": self.arrival, "cv_a": self.cv_a}


@dataclass(frozen=True)
class PolicySpec:
    """K3. k_short/k_long are ints; None or K_INF both mean 'unbounded' and are stored as K_INF.
    lane_cap {"short": n, "long": n} or None. from_json does NOT enforce the cross-field rules:
    call validate_policyspec (Policy.__init__ must)."""
    name: str
    k_short: int
    k_long: int
    order: str = "fcfs"
    lane_cap: dict | None = None
    f_rule: str = "governor"
    f_mhz: int | None = None
    params: dict = field(default_factory=dict)

    def __post_init__(self):
        for name in ("k_short", "k_long"):
            v = getattr(self, name)
            object.__setattr__(self, name, K_INF if v is None else v)
        if self.lane_cap is not None:
            object.__setattr__(self, "lane_cap", dict(self.lane_cap))
        object.__setattr__(self, "params", dict(self.params))

    @classmethod
    def from_json(cls, o: dict) -> "PolicySpec":
        _keys(o, _POLICY_KEYS, "policyspec")
        return cls(o["name"], o["k_short"], o["k_long"], o["order"], o["lane_cap"], o["f_rule"], o["f_mhz"],
                   o["params"])

    def to_json(self) -> dict:
        return {
            "name": self.name, "k_short": None if self.k_short >= K_INF else self.k_short,
            "k_long": None if self.k_long >= K_INF else self.k_long, "order": self.order,
            "lane_cap": None if self.lane_cap is None else dict(self.lane_cap), "f_rule": self.f_rule,
            "f_mhz": self.f_mhz, "params": dict(self.params)}


_POLICY_KEYS = ("name", "k_short", "k_long", "order", "lane_cap", "f_rule", "f_mhz", "params")


@dataclass(frozen=True)
class QueuedReq:
    req_id: str
    type_id: str
    family: str
    klass: str
    t_arrival_ns: int
    t_deadline_ns: int | None = None
    k_override: int | None = None

    @classmethod
    def from_json(cls, o: dict) -> "QueuedReq":
        return cls(o["req_id"], o["type_id"], o["family"], o["klass"], o["t_arrival_ns"], o.get("t_deadline_ns"),
                   o.get("k_override"))

    def to_json(self) -> dict:
        return {"req_id": self.req_id, "type_id": self.type_id, "family": self.family, "klass": self.klass,
                "t_arrival_ns": self.t_arrival_ns, "t_deadline_ns": self.t_deadline_ns, "k_override": self.k_override}


@dataclass(frozen=True)
class InServiceReq:
    """type_id is the type that RUNS (an exit policy may change it from the queued one). t_deadline_ns is set by
    AdmissionCore from the queued request; to_json writes it only when it is not None."""
    req_id: str
    type_id: str
    family: str
    klass: str
    t_admit_ns: int
    t_deadline_ns: int | None = None

    @classmethod
    def from_json(cls, o: dict) -> "InServiceReq":
        return cls(o["req_id"], o["type_id"], o["family"], o["klass"], o["t_admit_ns"], o.get("t_deadline_ns"))

    def to_json(self) -> dict:
        out = {"req_id": self.req_id, "type_id": self.type_id, "family": self.family, "klass": self.klass,
               "t_admit_ns": self.t_admit_ns}
        if self.t_deadline_ns is not None:
            out["t_deadline_ns"] = self.t_deadline_ns
        return out


@dataclass(frozen=True)
class DispatchView:
    """Immutable snapshot a Policy decides on. queue is in ARRIVAL order. f_current_mhz = last COMMANDED level."""
    now_ns: int
    queue: tuple
    in_service: tuple
    f_current_mhz: int | None

    def __post_init__(self):
        object.__setattr__(self, "queue", tuple(self.queue))
        object.__setattr__(self, "in_service", tuple(self.in_service))

    @classmethod
    def from_json(cls, o: dict) -> "DispatchView":
        return cls(o["now_ns"], tuple(QueuedReq.from_json(q) for q in o["queue"]),
                   tuple(InServiceReq.from_json(r) for r in o["in_service"]), o.get("f_current_mhz"))

    def to_json(self) -> dict:
        return {"now_ns": self.now_ns, "f_current_mhz": self.f_current_mhz,
                "queue": [q.to_json() for q in self.queue], "in_service": [r.to_json() for r in self.in_service]}


@dataclass(frozen=True)
class TagSpec:
    """Deviator tag: every `every`-th request of class `klass` gets k_override = k_dev."""
    klass: str
    k_dev: int
    every: int = 50


@dataclass(frozen=True)
class SimConfig:
    """K14. schedule (tuple of ScheduleItem dicts) means replay: n_reps is forced to 1 and SimResult.trace is filled."""
    n_requests: int = 20000
    n_warmup: int = 500
    n_reps: int = 5
    seed: int = 0
    family_exclusive: bool = True
    sample_slowdown: bool = True
    alpha: float | None = 5.0
    switch_lag_s: float = 0.0
    tag: TagSpec | None = None
    schedule: tuple | None = None

    def __post_init__(self):
        if self.schedule is not None:
            object.__setattr__(self, "schedule", tuple(self.schedule))
            object.__setattr__(self, "n_reps", 1)


_TYPE_STATS_KEYS = ("n", "mean_R", "p50_R", "p95_R", "p99_R", "complete_rate", "truncation_rate", "miss_rate",
                    "expired_rate", "mean_completion", "mean_shortfall")


@dataclass(frozen=True)
class TypeStats:
    n: int
    mean_R: float
    p50_R: float
    p95_R: float
    p99_R: float
    complete_rate: float
    truncation_rate: float
    miss_rate: float
    expired_rate: float
    mean_completion: float
    mean_shortfall: float

    @classmethod
    def from_json(cls, o: dict) -> "TypeStats":
        return cls(*[o[k] for k in _TYPE_STATS_KEYS])

    def to_json(self) -> dict:
        return {k: getattr(self, k) for k in _TYPE_STATS_KEYS}


@dataclass(frozen=True)
class SimResult:
    """K14 result (exp2.simresult/1). per_type {type_id: TypeStats}; energy_ci (lo, hi); reps list of dicts;
    tagged TypeStats-shaped dict + klass/k_dev, or None; trace per-request rows (replay only) or None."""
    spec_name: str
    per_type: dict
    energy_per_req_j: float
    energy_ci: tuple
    mean_power_w: float
    throughput: float
    mean_in_system: float
    utilization: float
    unstable: bool
    tagged: dict | None = None
    reps: tuple = ()
    trace: tuple | None = None
    runtime_s: float = 0.0

    def to_json(self) -> dict:
        return {
            "schema": SCHEMAS["simresult"], "spec_name": self.spec_name,
            "per_type": {k: (v.to_json() if isinstance(v, TypeStats) else dict(v)) for k, v in self.per_type.items()},
            "energy_per_req_j": self.energy_per_req_j, "energy_ci": list(self.energy_ci),
            "mean_power_w": self.mean_power_w, "throughput": self.throughput, "mean_in_system": self.mean_in_system,
            "utilization": self.utilization, "unstable": self.unstable, "tagged": self.tagged,
            "reps": [dict(r) for r in self.reps], "trace": None if self.trace is None else list(self.trace),
            "runtime_s": self.runtime_s}

    @classmethod
    def from_json(cls, o: dict) -> "SimResult":
        validate_simresult(o)
        return cls(o["spec_name"], {k: TypeStats.from_json(v) for k, v in o["per_type"].items()},
                   o["energy_per_req_j"], tuple(o["energy_ci"]), o["mean_power_w"], o["throughput"],
                   o["mean_in_system"], o["utilization"], o["unstable"], o["tagged"], tuple(o["reps"]),
                   None if o["trace"] is None else tuple(o["trace"]), o["runtime_s"])


# --------------------------------------------------------------------------------------------- Protocols
@runtime_checkable
class Policy(Protocol):
    """Implemented by exp2.policies.Policy (S1b)."""
    spec: PolicySpec
    types: TypeTable

    def admit(self, view: DispatchView) -> list[str]: ...
    def desired_f(self, view: DispatchView) -> int | None: ...


@runtime_checkable
class FamilyAdapter(Protocol):
    """K6. cfg for load(): preprocess, n_inputs, seed, sync("event"|"spin"), compile(bool),
    input_source("dataset"|"synthetic"). finish(): head at `reached` + small D2H; reached None -> no-op."""
    family: str

    def load(self, exits: list[int], cfg: dict) -> dict: ...
    def make_inputs(self, n: int, seed: int, mode: str) -> None: ...
    def checkpoints(self, exit: int) -> list[int]: ...          # == default_checkpoints(family, exit)
    def begin(self, input_idx: int, exit: int) -> object: ...   # per_request host pre-processing happens here
    def advance(self, state, ckpt: int) -> None: ...            # keeps state at the last on-time checkpoint
    def sync(self) -> None: ...
    def finish(self, state, reached: int | None) -> object: ...
    def mem_stats(self) -> dict: ...                            # torch_reserved_mb, torch_peak_alloc_mb, rss_mb
    def reset_peak(self) -> None: ...


@runtime_checkable
class SlowdownModel(Protocol):
    """K12 (exp2.slowdown/1). predict(i, (i,), f) == 1.0. fit raises HeldOutError on any split == 'heldout' row."""
    name: str

    def fit(self, corun_rows, solo_rows, types) -> "SlowdownModel": ...
    def predict(self, i: str, A: tuple, f_mhz: int) -> float: ...
    def sample(self, i: str, A: tuple, f_mhz: int, rng) -> float: ...   # predict x residual from the (family,|A|) pool
    def to_json(self) -> dict: ...


@runtime_checkable
class PowerModelP(Protocol):
    """K13. p(A, f) = p_idle(f) + dp(f) * min(1, U_A / eta(f)), U_A = sum u_i(f); p((), f) == p_idle(f)."""

    def p(self, A: tuple, f_mhz: int) -> float: ...
    def to_json(self) -> dict: ...


@runtime_checkable
class BoardControl(Protocol):
    """K17 (implemented by exp2.board.clocks.Board). ClockError on any refused write or failed read-back;
    ValueError (before any write) for a level the board does not offer."""

    def query_power_mode(self) -> tuple: ...                   # (name, id)
    def ensure_power_mode(self, mode_id: int = POWER_MODE_ID) -> None: ...
    def run_jetson_clocks(self) -> None: ...
    def store_jetson_clocks(self, path: str) -> None: ...
    def restore_jetson_clocks(self, path: str) -> None: ...
    def gpu_levels(self) -> list[int]: ...
    def set_gpu_level(self, mhz: int) -> int: ...              # returns the read-back cur_freq in MHz
    def set_gpu_governor_free(self) -> None: ...
    def lock_emc(self, mhz: float) -> float: ...               # returns the read-back EMC MHz
    def unlock_emc(self) -> None: ...
    def apply_arm(self, arm: str, gpu_mhz: int | None, emc_mhz: float | None) -> dict: ...  # read-back summary
    def snapshot(self) -> dict: ...
    def close(self) -> None: ...


@runtime_checkable
class RootHelperClient(Protocol):
    """K17 client side (exp2.board.sysfs.RootHelper). oplog lists the state-changing calls in call order."""
    oplog: list

    def write(self, path: str, value) -> str: ...                              # returns the read-back
    def read(self, path: str) -> str: ...
    def run(self, argv: list, timeout_s: float = 30.0) -> tuple: ...           # (rc, out)
    def close(self) -> None: ...


# --------------------------------------------------------------------------------------------- column tuples
REQUEST_KEYS = (
    "condition_id", "req_id", "type_id", "family", "exit", "klass", "target", "t_arrival_ns", "t_arrival_actual_ns",
    "t_deadline_ns", "t_admit_ns", "t_srv_recv_ns", "t_start_ns", "t_end_ns", "t_done_ns", "server", "policy",
    "policy_spec", "f_mhz", "n_in_service_at_admit", "queue_len_at_arrival", "reached", "completion", "outcome",
    "tokens_done", "ckpt", "is_warmup", "seed", "mode", "err")
REQUEST_OPTIONAL_KEYS = ("req_type_id",)   # Run 3: the arrival type of a row whose exit policy ran another type
POWER_CSV_COLS =("t_mono_ns", "t_wall_ns", "vdd_in_mv", "vdd_in_ma", "vdd_in_mw")
CLOCKS_CSV_FIXED_COLS = ("t_mono_ns", "t_wall_ns", "gpu_cur_hz", "gpu_min_hz", "gpu_max_hz", "gpu_load_permille",
                         "emc_mhz", "mem_available_kb", "cpu0_cur_khz")   # then temp_<zone>_mc ..., sorted
CLOCKS_CSV_OPTIONAL_COLS = ("swap_total_kb", "swap_free_kb", "pgmajfault")   # whole block or none, after FIXED
# sources: /proc/meminfo SwapTotal, SwapFree (kB); /proc/vmstat pgmajfault (cumulative)
FREQ_SWITCH_COLS = ("t_req_ns", "t_done_ns", "target_mhz", "readback_mhz")
RUN1_CSV_COLS = {
    "solo": ("type_id", "f_mhz", "repeat", "n", "tau_p50", "tau_mean", "tau_p95", "tau_p50_lo", "tau_p50_hi", "ess",
             "u", "p_board_w", "window_s"),
    "solo_samples": ("type_id", "f_mhz", "repeat", "tau_s"),
    "corun": ("cell_id", "split", "f_mhz", "repeat", "type_id", "family", "A", "n", "n_window", "s_p50", "s_mean",
              "s_p95", "s_p50_lo", "s_p50_hi", "ess", "tau_solo_p50", "U_A", "window_s", "energy_j", "p_board_w"),
    "corun_samples": ("cell_id", "f_mhz", "repeat", "type_id", "s"),
    "idle": ("f_mhz", "kind", "p_idle_w"),
    "ckpt": ("type_id", "f_mhz", "ckpt", "frac_p50", "head_frac_p50"),
    "warmup": ("session_id", "family", "t_load_s", "first_infer_s_max", "warm_energy_j", "cache"),
    "mem": ("family", "exit", "resident_mb", "act_mb"),
}
OPEN_CONDITIONS_COLS = (
    "condition_id", "policy_key", "workload", "rho", "arrival", "alpha", "f_mhz", "repeat", "type_id", "n",
    "complete_rate", "truncation_rate", "miss_rate", "expired_rate", "error_rate", "mean_completion", "p95_R_s",
    "energy_per_req_j", "energy_sess_per_req_j", "throughput_rps", "min_mem_available_mb", "late_rate",
    "overshoot_p95_s", "overshoot_max_s")
COEXIT_PILOT_COLS = (
    "method", "mix", "arrival", "cv_a", "rho", "rate_per_s", "deadline_scale", "rep", "seed", "family", "deadline_s",
    "n", "n_success", "slo_violation_ratio", "p95_R_s", "p99_R_s", "mean_R_s", "mean_norm_depth", "share_floor",
    "n_admitted", "n_complete_late", "n_truncated", "n_miss", "n_expired", "sim_runtime_s")
COEXIT_SUMMARY_COLS = (
    "method", "mix", "arrival", "cv_a", "rho", "deadline_scale", "family", "n_reps", "viol_mean", "viol_ci_lo",
    "viol_ci_hi", "depth_mean", "depth_ci_lo", "depth_ci_hi", "p95_R_s_mean", "p99_R_s_mean", "share_floor_mean",
    "d_viol_vs_edgeserving_mean", "d_viol_vs_edgeserving_lo", "d_viol_vs_edgeserving_hi")
RUN3_RUNS_COLS = (
    "session_id", "condition_id", "method", "mix", "arrival", "cv_a", "rho", "rep", "seed", "kept", "reason", "n_rows",
    "n_error", "screen_ok", "gpu_lock_frac", "emc_lock_frac", "max_tj_c", "min_mem_available_mb", "p99_arrival_lag_ms",
    "p99_dispatch_ms", "wall_s")
RUN3_COMPARE_COLS = (
    "method", "mix", "arrival", "cv_a", "rho", "family", "board_n_reps", "board_viol_mean", "board_viol_ci_lo",
    "board_viol_ci_hi", "sim_viol_mean", "sim_viol_ci_lo", "sim_viol_ci_hi", "sim_viol_same_seeds", "board_depth_mean",
    "sim_depth_mean", "board_d_viol_vs_es_mean", "board_d_viol_vs_es_lo", "board_d_viol_vs_es_hi",
    "sim_d_viol_vs_es_mean", "sim_d_viol_vs_es_lo", "sim_d_viol_vs_es_hi", "same_sign")
SCREEN_CSV_COLS = (
    "session_id", "condition_id", "mode", "arm", "gpu_mhz", "kept", "reason", "n_clock_rows", "gpu_lock_frac",
    "emc_lock_frac", "max_tj_c", "min_mem_available_mb", "max_swap_used_mb", "pgmajfault_delta", "screen_ok")
CSV_KINDS = tuple(RUN1_CSV_COLS) + ("power", "clocks", "freq_switch", "open_conditions", "screen")

_CSV_STR_COLS = frozenset({"type_id", "cell_id", "split", "family", "A", "session_id", "cache", "kind", "condition_id",
                           "policy_key", "workload", "arrival", "mode", "arm", "reason"})
_CSV_INT_COLS = frozenset({"f_mhz", "repeat", "n", "n_window", "ckpt", "exit", "t_mono_ns", "t_wall_ns", "vdd_in_mv",
                           "vdd_in_ma", "gpu_cur_hz", "gpu_min_hz", "gpu_max_hz", "gpu_load_permille",
                           "mem_available_kb", "cpu0_cur_khz", "t_req_ns", "t_done_ns", "target_mhz", "readback_mhz",
                           "swap_total_kb", "swap_free_kb", "pgmajfault", "gpu_mhz", "kept", "screen_ok",
                           "n_clock_rows", "pgmajfault_delta"})


# --------------------------------------------------------------------------------------------- shared functions
def few_requests_reason(counts: dict):
    """AM1. counts = {type_id: measured (non warm-up) requests in the window}. None when every tenant has at least
    MIN_MEASURED_REQ_PER_TENANT, else "few_requests:<type_id>=<n>" for the first failing type_id in sorted order."""
    for tid in sorted(counts):
        if counts[tid] < MIN_MEASURED_REQ_PER_TENANT:
            return f"few_requests:{tid}={counts[tid]}"
    return None


def k_of(spec, klass: str) -> int:
    """Concurrency limit of the class under `spec` (PolicySpec or its json dict). None -> K_INF."""
    if klass not in CLASSES:
        _fail("klass", f"{klass!r} not in {CLASSES}")
    v = spec.get(f"k_{klass}") if isinstance(spec, dict) else getattr(spec, f"k_{klass}")
    return K_INF if v is None else v


def k_limit(spec, req) -> int:
    """Limit that applies to `req` (QueuedReq): its k_override when set, else k_of(spec, req.klass)."""
    return req.k_override if req.k_override is not None else k_of(spec, req.klass)


def deadline_s(types: TypeTable, type_id: str):
    """Relative deadline in seconds: types.alpha * tau_s[f_max].p50; None when alpha is None."""
    if types.alpha is None:
        return None
    return types.alpha * types.tau(type_id, types.f_max_mhz, "p50")


def with_deadlines(items, types: TypeTable, alpha) -> list[dict]:
    """Copy ScheduleItems with t_deadline_ns = t_arrival_ns + alpha * tau p50(f_max); alpha None -> None."""
    out = []
    for it in items:
        new = dict(it)
        if alpha is None:
            new["t_deadline_ns"] = None
        else:
            p50 = types.tau(it["type_id"], types.f_max_mhz, "p50")
            new["t_deadline_ns"] = it["t_arrival_ns"] + _s_to_ns(alpha * p50)
        out.append(new)
    return out


def with_family_deadlines(items, deadlines_s) -> list[dict]:
    """Copy ScheduleItems with t_deadline_ns = t_arrival_ns + round(deadlines_s[family] * 1e9) (Run 3, CoExit pilot)."""
    out = []
    for it in items:
        new = dict(it)
        new["t_deadline_ns"] = it["t_arrival_ns"] + int(round(deadlines_s[parse_type_id(it["type_id"])[0]] * 1e9))
        out.append(new)
    return out


def rate_for_load(rho: float, workload, types: TypeTable) -> float:
    """Arrival rate (1/s) that offers load rho: rho / sum_j p_j * tau_j(f_max).MEAN (the mean, never the p50)."""
    if not _is_num(rho) or rho <= 0:
        _fail("rho", f"must be > 0, got {rho!r}")
    ids = workload["types"] if isinstance(workload, dict) else workload.types
    probs = workload["probs"] if isinstance(workload, dict) else workload.probs
    mean_service = sum(p * types.tau(tid, types.f_max_mhz, "mean") for tid, p in zip(ids, probs))
    if not mean_service > 0:
        _fail("workload", "mean service time is not positive")
    return rho / mean_service


# --------------------------------------------------------------------------------------------- hashing and freeze
def canonical_json(obj) -> str:
    """Deterministic JSON text used by every content hash."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def _hash_without(obj, key: str) -> str:
    if isinstance(obj, dict):
        obj = {k: v for k, v in obj.items() if k != key}
    return hashlib.sha256(canonical_json(obj).encode("ascii")).hexdigest()


def content_sha256(obj) -> str:
    """sha256 of canonical_json(obj minus its top-level "content_sha256")."""
    return _hash_without(obj, "content_sha256")


def plan_sha256(plan: dict) -> str:
    """sha256 of canonical_json(plan minus its top-level "sha256"); this is plan["sha256"]."""
    return _hash_without(plan, "sha256")


def with_content_sha256(obj: dict) -> dict:
    out = dict(obj)
    out["content_sha256"] = content_sha256(out)
    return out


def with_plan_sha256(plan: dict) -> dict:
    out = dict(plan)
    out["sha256"] = plan_sha256(out)
    return out


def sha256_file(path) -> str:
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def sidecar_path(path) -> str:
    return str(path) + ".sha256"


def write_sidecar(path) -> str:
    """Write `<hex>  <basename>\\n` (sha256sum -c format, LF only) next to `path`; returns the hex."""
    digest = sha256_file(path)
    with open(sidecar_path(path), "wb") as fh:
        fh.write(f"{digest}  {os.path.basename(str(path))}\n".encode("ascii"))
    return digest


def verify_freeze_file(path) -> dict:
    """Return the parsed freeze after checking (1) the sidecar hash of the file bytes, (2) content_sha256 of the
    content, (3) the schema. Any failure, including unreadable or malformed files, raises FreezeError."""
    path = str(path)
    try:
        with open(path, "rb") as fh:
            raw = fh.read()
    except OSError as e:
        raise FreezeError(f"cannot read freeze file {path}: {e}") from e
    try:
        with open(sidecar_path(path), "rb") as fh:
            side = fh.read().decode("ascii")
    except (OSError, UnicodeDecodeError) as e:
        raise FreezeError(f"cannot read sidecar {sidecar_path(path)}: {e}") from e
    m = _SIDECAR.fullmatch(side)
    if not m:
        raise FreezeError(f"sidecar {sidecar_path(path)} is not '<sha256>  <name>\\n': {side!r}")
    actual = hashlib.sha256(raw).hexdigest()
    if actual != m.group(1):
        raise FreezeError(f"freeze file hash {actual} != sidecar hash {m.group(1)} ({path})")
    try:
        obj = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as e:
        raise FreezeError(f"freeze file {path} is not valid JSON: {e}") from e
    if not isinstance(obj, dict) or "content_sha256" not in obj:
        raise FreezeError(f"freeze file {path} has no content_sha256")
    calc = content_sha256(obj)
    if obj["content_sha256"] != calc:
        raise FreezeError(f"content_sha256 {obj['content_sha256']} != recomputed {calc} ({path})")
    try:
        validate_freeze(obj)
    except ContractError as e:
        raise FreezeError(f"freeze file {path} fails the contract: {e}") from e
    return obj


# --------------------------------------------------------------------------------------------- validators
def validate_types(o) -> dict:
    w = "types"
    _keys(o, ("schema", "alpha", "f_levels_mhz", "corun_levels_mhz", "heldout_level_mhz", "f_max_mhz", "llama_tokens",
              "llama_prompt_tokens", "families", "types", "provenance"), w, optional=("synthetic",))
    _schema(o, "types", w)
    if "synthetic" in o:
        _bool(o["synthetic"], f"{w}.synthetic")
    _num(o["alpha"], f"{w}.alpha", positive=True, null=True)
    levels = [_int(f, f"{w}.f_levels_mhz", lo=1) for f in _list(o["f_levels_mhz"], f"{w}.f_levels_mhz", min_len=1)]
    if levels != sorted(set(levels)):
        _fail(w, f"f_levels_mhz must be strictly ascending, got {levels}")
    corun = [_int(f, f"{w}.corun_levels_mhz") for f in _list(o["corun_levels_mhz"], f"{w}.corun_levels_mhz", min_len=1)]
    if corun != sorted(set(corun)) or not set(corun) <= set(levels):
        _fail(w, f"corun_levels_mhz {corun} must be ascending and a subset of f_levels_mhz {levels}")
    fmax = _int(o["f_max_mhz"], f"{w}.f_max_mhz")
    if fmax != max(levels) or fmax not in corun:
        _fail(w, f"f_max_mhz {fmax} must equal max(f_levels_mhz) and be a corun level")
    held = _int(o["heldout_level_mhz"], f"{w}.heldout_level_mhz", null=True)
    if held is not None and (held not in levels or held in corun):
        _fail(w, f"heldout_level_mhz {held} must be in f_levels_mhz and not a corun level")
    _int(o["llama_tokens"], f"{w}.llama_tokens", lo=1)
    _int(o["llama_prompt_tokens"], f"{w}.llama_prompt_tokens", lo=1)
    fams = _dict(o["families"], f"{w}.families")
    for fam, d in fams.items():
        _str(fam, f"{w}.families key", choices=FAMILIES)
        _keys(d, ("resident_mb",), f"{w}.families[{fam}]")
        _num(d["resident_mb"], f"{w}.families[{fam}].resident_mb", lo=0)
    prov = _dict(o["provenance"], f"{w}.provenance")
    _keys(prov, ("klass_rule", "sources", "sha256"), f"{w}.provenance")
    types = _list(o["types"], f"{w}.types", min_len=1)
    seen = set()
    for t in types:
        tw = f"{w}.types[{t.get('type_id') if isinstance(t, dict) else '?'}]"
        _keys(t, ("type_id", "family", "exit", "klass", "ckpts", "tau_s", "ckpt_frac", "head_frac", "u", "act_mb"), tw)
        fam, ex = parse_type_id(t["type_id"])
        if t["family"] != fam or t["exit"] != ex:
            _fail(tw, f"family/exit {t['family']!r}/{t['exit']!r} disagree with type_id")
        if t["type_id"] in seen:
            _fail(tw, "duplicate type_id")
        seen.add(t["type_id"])
        if fam not in fams:
            _fail(tw, f"family {fam!r} missing from families")
        _str(t["klass"], f"{tw}.klass", choices=CLASSES)
        ck = default_checkpoints(fam, ex)
        if list(t["ckpts"]) != ck:
            _fail(tw, f"ckpts {t['ckpts']} != default_checkpoints {ck}")
        tau = _dict(t["tau_s"], f"{tw}.tau_s")
        need = set(corun) | {fmax}
        if not need <= {_int_key(f, tw) for f in tau}:
            _fail(tw, f"tau_s must cover corun levels and f_max {sorted(need)}")
        for f, st in tau.items():
            sw = f"{tw}.tau_s[{f}]"
            if _int_key(f, sw) not in levels:
                _fail(sw, f"level not in f_levels_mhz {levels}")
            _keys(st, ("p50", "mean", "p95", "samples"), sw)
            for s in _STATS:
                _num(st[s], f"{sw}.{s}", positive=True)
            if st["p50"] > st["p95"]:
                _fail(sw, "p50 > p95")
            for x in _list(st["samples"], f"{sw}.samples", min_len=1):
                _num(x, f"{sw}.samples", positive=True)
        cf = _dict(t["ckpt_frac"], f"{tw}.ckpt_frac")
        if sorted(_int_key(c, tw) for c in cf) != ck:
            _fail(tw, f"ckpt_frac keys must be the checkpoints {ck}")
        prev = 0.0
        for c in ck:
            v = _num(cf[str(c)], f"{tw}.ckpt_frac[{c}]", lo=0.0, hi=1.0 + 1e-9, positive=True)
            if v < prev - 1e-12:
                _fail(tw, "ckpt_frac must be non-decreasing")
            prev = v
        hf = _num(t["head_frac"], f"{tw}.head_frac", lo=0.0, hi=1.0)
        if fam == "llama" and hf != 0.0:
            _fail(tw, "llama has no separate head: head_frac must be 0")
        if abs(cf[str(ck[-1])] + hf - 1.0) > 0.05 + 1e-9:
            _fail(tw, f"ckpt_frac[target] + head_frac = {cf[str(ck[-1])] + hf} is not within 0.05 of 1")
        _level_map(t["u"], f"{tw}.u", levels=levels, lo=0.0, hi=1.0, required=need)
        _num(t["act_mb"], f"{tw}.act_mb", lo=0.0)
    return o


def validate_workload(o, types: TypeTable | None = None) -> dict:
    w = "workload"
    _keys(o, ("name", "types", "probs", "rho", "rate_per_s", "arrival", "cv_a"), w)
    _str(o["name"], f"{w}.name")
    ids = [parse_type_id(s) and s for s in _list(o["types"], f"{w}.types", min_len=1)]
    probs = [_num(p, f"{w}.probs", lo=0.0) for p in _list(o["probs"], f"{w}.probs")]
    if len(ids) != len(probs) or len(set(ids)) != len(ids):
        _fail(w, "types and probs must have the same length and types must be unique")
    if abs(sum(probs) - 1.0) > 1e-9:
        _fail(w, f"probs sum to {sum(probs)}, not 1")
    _num(o["rho"], f"{w}.rho", positive=True)
    _num(o["rate_per_s"], f"{w}.rate_per_s", positive=True)
    _str(o["arrival"], f"{w}.arrival", choices=_ARRIVALS)
    _num(o["cv_a"], f"{w}.cv_a", positive=True)
    if o["arrival"] == "poisson" and o["cv_a"] != 1.0:
        _fail(w, "arrival 'poisson' needs cv_a == 1.0")
    if types is not None:
        want = rate_for_load(o["rho"], o, types)
        if abs(o["rate_per_s"] - want) > 1e-3 * want:
            _fail(w, f"rate_per_s {o['rate_per_s']} != rate_for_load {want}")
    return o


def validate_policyspec(o) -> dict:
    """Accepts a PolicySpec or its json dict. Rules: edgeserving needs k 1/1; static needs f_mhz; governor needs
    f_mhz null; greenllm needs lane_cap; occupancy needs params.f_base_mhz and params.alpha_p."""
    if isinstance(o, PolicySpec):
        o = o.to_json()
    w = "policyspec"
    _keys(o, _POLICY_KEYS, w)
    _str(o["name"], f"{w}.name")
    ks, kl = (_int(o[k], f"{w}.{k}", lo=1, null=True) for k in ("k_short", "k_long"))
    _str(o["order"], f"{w}.order", choices=ORDERS)
    _str(o["f_rule"], f"{w}.f_rule", choices=F_RULES)
    f = _int(o["f_mhz"], f"{w}.f_mhz", lo=1, null=True)
    _dict(o["params"], f"{w}.params")
    lane = o["lane_cap"]
    if lane is not None:
        _keys(lane, CLASSES, f"{w}.lane_cap")
        for c in CLASSES:
            _int(lane[c], f"{w}.lane_cap.{c}", lo=1)
    if o["order"] == "edgeserving":
        if ks != 1 or kl != 1:
            _fail(w, f"edgeserving needs k_short == k_long == 1, got {ks}/{kl}")
        if "C" in o["params"]:
            _num(o["params"]["C"], f"{w}.params.C", positive=True)
    if o["f_rule"] == "static" and f is None:
        _fail(w, "f_rule 'static' needs f_mhz")
    if o["f_rule"] == "governor" and f is not None:
        _fail(w, "f_rule 'governor' needs f_mhz null")
    if o["f_rule"] == "greenllm" and lane is None:
        _fail(w, "f_rule 'greenllm' needs lane_cap")
    if o["f_rule"] == "occupancy":
        _num(o["params"].get("f_base_mhz"), f"{w}.params.f_base_mhz", positive=True)
        _num(o["params"].get("alpha_p"), f"{w}.params.alpha_p", positive=True)
    if "exit_rule" in o["params"]:
        rule = _str(o["params"]["exit_rule"], f"{w}.params.exit_rule", choices=EXIT_RULES)
        if "tau_stat" in o["params"]:
            _str(o["params"]["tau_stat"], f"{w}.params.tau_stat", choices=_STATS)
        if "cap_margin" in o["params"]:
            m = _num(o["params"]["cap_margin"], f"{w}.params.cap_margin")
            if m < 0 or rule != "coexit":
                _fail(w, "params.cap_margin must be >= 0 and is read only by exit_rule 'coexit'")
        if (rule == "edgeserving") != (o["order"] == "edgeserving"):
            _fail(w, f"exit_rule {rule!r} and order {o['order']!r} disagree (edgeserving needs both)")
        if o["f_rule"] != "static":
            _fail(w, "an exit_rule needs f_rule 'static'")
        if rule != "edgeserving" and (ks is not None or kl is not None or lane is not None):
            _fail(w, f"exit_rule {rule!r} reads no concurrency limit: k_short, k_long and lane_cap must be null")
    return o


def _ckpt_pairs(v, where):
    prev_c, prev_t = None, None
    for pair in _list(v, where):
        _list(pair, where, min_len=2)
        if len(pair) != 2:
            _fail(where, f"ckpt entry must be [ckpt, t_ns], got {pair!r}")
        c, t = _int(pair[0], where), _int(pair[1], where)
        if prev_c is not None and (c <= prev_c or t < prev_t):
            _fail(where, "ckpt entries must have increasing ckpt and non-decreasing time")
        prev_c, prev_t = c, t


def validate_request_row(o) -> dict:
    """K8: exactly REQUEST_KEYS, typed, with the per-outcome rules (expired/miss/error/closed)."""
    w = "request_row"
    _keys(o, REQUEST_KEYS, w, optional=REQUEST_OPTIONAL_KEYS)
    _cond_id(o["condition_id"], f"{w}.condition_id")
    _str(o["req_id"], f"{w}.req_id")
    fam, ex = parse_type_id(o["type_id"])
    if o["family"] != fam or o["exit"] != ex:
        _fail(w, f"family/exit {o['family']!r}/{o['exit']!r} disagree with type_id {o['type_id']!r}")
    _str(o["klass"], f"{w}.klass", choices=CLASSES)
    target = default_checkpoints(fam, ex)[-1]
    if o["target"] != target:
        _fail(w, f"target {o['target']!r} != {target}")
    for k in ("t_arrival_ns", "t_done_ns"):
        _int(o[k], f"{w}.{k}")
    for k in ("t_arrival_actual_ns", "t_deadline_ns", "t_admit_ns", "t_srv_recv_ns", "t_start_ns", "t_end_ns"):
        _int(o[k], f"{w}.{k}", null=True)
    _str(o["server"], f"{w}.server")
    _str(o["policy"], f"{w}.policy")
    if o["policy_spec"] is not None:
        validate_policyspec(o["policy_spec"])
    if o["f_mhz"] != "governor":
        _int(o["f_mhz"], f"{w}.f_mhz", lo=1, null=True)
    _int(o["n_in_service_at_admit"], f"{w}.n_in_service_at_admit", lo=0, null=True)
    _int(o["queue_len_at_arrival"], f"{w}.queue_len_at_arrival", lo=0)
    reached = _int(o["reached"], f"{w}.reached", null=True)
    comp = _num(o["completion"], f"{w}.completion", lo=0.0, hi=1.0)
    outcome = _str(o["outcome"], f"{w}.outcome", choices=OUTCOMES)
    _int(o["tokens_done"], f"{w}.tokens_done", null=True)
    _ckpt_pairs(o["ckpt"], f"{w}.ckpt")
    _bool(o["is_warmup"], f"{w}.is_warmup")
    _int(o["seed"], f"{w}.seed")
    _str(o["mode"], f"{w}.mode", choices=("open", "closed"))
    _str(o["err"], f"{w}.err", null=True)
    if (outcome == "error") != (o["err"] is not None):
        _fail(w, "err must be set exactly when outcome is 'error'")
    exit_policy = (o["mode"] == "open" and o["policy_spec"] is not None
                   and "exit_rule" in o["policy_spec"].get("params", {}))
    if ("req_type_id" in o) != exit_policy:
        _fail(w, "req_type_id is present exactly on open rows of an exit policy (params.exit_rule)")
    if "req_type_id" in o:
        if parse_type_id(o["req_type_id"])[0] != fam:
            _fail(w, f"req_type_id {o['req_type_id']!r} is not of family {fam!r}")
        if o["t_admit_ns"] is None and o["req_type_id"] != o["type_id"]:
            _fail(w, "a row that never ran keeps type_id == req_type_id")
    if reached is not None and abs(comp - completion(fam, ex, reached)) > 1e-3:
        _fail(w, f"completion {comp} != completion({fam}, {ex}, {reached})")
    if outcome == "complete" and (reached != target or comp != 1.0):
        _fail(w, "outcome 'complete' needs reached == target and completion 1.0")
    if outcome == "truncated" and (reached is None or reached >= target):
        _fail(w, "outcome 'truncated' needs 0 < reached < target")
    if outcome in ("miss", "expired_in_queue", "error") and (reached is not None or comp != 0.0):
        _fail(w, f"outcome {outcome!r} needs reached null and completion 0.0")
    if outcome == "expired_in_queue":
        late = [k for k in ("t_admit_ns", "t_srv_recv_ns", "t_start_ns", "t_end_ns", "n_in_service_at_admit")
                if o[k] is not None]
        if late or o["ckpt"] != [] or o["t_deadline_ns"] is None or o["t_done_ns"] != o["t_deadline_ns"]:
            _fail(w, "expired_in_queue needs t_admit..t_end and n_in_service null, ckpt [], t_done == t_deadline")
    if o["mode"] == "closed":
        if o["t_deadline_ns"] is not None:
            _fail(w, "closed-loop rows have t_deadline_ns null")
        if o["t_admit_ns"] is not None and o["t_arrival_ns"] != o["t_admit_ns"]:
            _fail(w, "closed-loop rows have t_arrival_ns == t_admit_ns")
    return o


def validate_ready(o) -> dict:
    w = "ready"
    _keys(o, ("op", "family", "pid", "exits_warmed", "warmup", "input_source", "versions", "mem", "t_ready_ns"), w)
    if o["op"] != "ready":
        _fail(w, f"op must be 'ready', got {o['op']!r}")
    fam = _str(o["family"], f"{w}.family", choices=FAMILIES)
    _int(o["pid"], f"{w}.pid", lo=0)
    exits = [_int(e, f"{w}.exits_warmed") for e in _list(o["exits_warmed"], f"{w}.exits_warmed", min_len=1)]
    for e in exits:
        _check_family_exit(fam, e)
    wu = o["warmup"]
    _keys(wu, ("load_s", "first_infer_s", "n_warm", "cv_last5", "act_mb", "inductor_cache_files_at_start"),
          f"{w}.warmup")
    _num(wu["load_s"], f"{w}.warmup.load_s", lo=0.0)
    _int(wu["inductor_cache_files_at_start"], f"{w}.warmup.inductor_cache_files_at_start", lo=0)
    for key in ("first_infer_s", "n_warm", "cv_last5", "act_mb"):
        d = _dict(wu[key], f"{w}.warmup.{key}")
        if sorted(d) != sorted(str(e) for e in exits):
            _fail(w, f"warmup.{key} keys {sorted(d)} != exits_warmed {exits}")
        for k, x in d.items():
            _num(x, f"{w}.warmup.{key}[{k}]", lo=0.0)
    _str(o["input_source"], f"{w}.input_source", choices=_PREPROCESS + ("dataset", "synthetic"))
    v = o["versions"]
    _keys(v, ("python", "torch", "transformers", "cuda", "compile"), f"{w}.versions")
    _str(v["python"], f"{w}.versions.python")
    for k in ("torch", "transformers", "cuda"):
        _str(v[k], f"{w}.versions.{k}", null=True)
    _str(v["compile"], f"{w}.versions.compile", choices=_COMPILE)
    mem = o["mem"]
    _keys(mem, ("torch_reserved_mb", "rss_mb"), f"{w}.mem")
    for k in mem:
        _num(mem[k], f"{w}.mem.{k}", lo=0.0)
    _int(o["t_ready_ns"], f"{w}.t_ready_ns")
    return o


def validate_run(o) -> dict:
    w = "run"
    _keys(o, ("op", "req_id", "type_id", "exit", "input_idx", "deadline_ns", "t_admit_ns"), w)
    if o["op"] != "run":
        _fail(w, f"op must be 'run', got {o['op']!r}")
    _str(o["req_id"], f"{w}.req_id")
    fam, ex = parse_type_id(o["type_id"])
    if o["exit"] != ex:
        _fail(w, f"exit {o['exit']!r} disagrees with type_id")
    _int(o["input_idx"], f"{w}.input_idx", lo=0)
    _int(o["deadline_ns"], f"{w}.deadline_ns", null=True)
    _int(o["t_admit_ns"], f"{w}.t_admit_ns")
    return o


def validate_done(o) -> dict:
    w = "done"
    _keys(o, ("op", "req_id", "type_id", "t_recv_ns", "t_start_ns", "t_end_ns", "reached", "target", "completion",
              "outcome", "ckpt", "tokens_done", "err"), w)
    if o["op"] != "done":
        _fail(w, f"op must be 'done', got {o['op']!r}")
    _str(o["req_id"], f"{w}.req_id")
    fam, ex = parse_type_id(o["type_id"])
    for k in ("t_recv_ns", "t_start_ns", "t_end_ns"):
        _int(o[k], f"{w}.{k}", null=True)
    ts = [o[k] for k in ("t_recv_ns", "t_start_ns", "t_end_ns") if o[k] is not None]
    if ts != sorted(ts):
        _fail(w, "t_recv_ns <= t_start_ns <= t_end_ns violated")
    reached = _int(o["reached"], f"{w}.reached", null=True)
    target = default_checkpoints(fam, ex)[-1]
    if o["target"] != target:
        _fail(w, f"target {o['target']!r} != {target}")
    comp = _num(o["completion"], f"{w}.completion", lo=0.0, hi=1.0)
    outcome = _str(o["outcome"], f"{w}.outcome", choices=("complete", "truncated", "miss", "error"))
    _ckpt_pairs(o["ckpt"], f"{w}.ckpt")
    _int(o["tokens_done"], f"{w}.tokens_done", null=True)
    _str(o["err"], f"{w}.err", null=True)
    if (outcome == "error") != (o["err"] is not None):
        _fail(w, "err must be set exactly when outcome is 'error'")
    if reached is not None and abs(comp - completion(fam, ex, reached)) > 1e-3:
        _fail(w, f"completion {comp} != completion({fam}, {ex}, {reached})")
    if outcome == "complete" and (reached != target or comp != 1.0):
        _fail(w, "outcome 'complete' needs reached == target and completion 1.0")
    if outcome == "truncated" and (reached is None or reached >= target):
        _fail(w, "outcome 'truncated' needs reached < target")
    if outcome in ("miss", "error") and (reached is not None or comp != 0.0):
        _fail(w, f"outcome {outcome!r} needs reached null and completion 0.0")
    return o


def validate_server_msg(o) -> dict:
    """Dispatch on op: ready, run, done, ping, pong, stop."""
    _dict(o, "server_msg")
    op = o.get("op")
    if op == "ready":
        return validate_ready(o)
    if op == "run":
        return validate_run(o)
    if op == "done":
        return validate_done(o)
    if op == "ping" or op == "stop":
        _keys(o, ("op",), op)
        return o
    if op == "pong":
        _keys(o, ("op", "t_ns"), "pong")
        _int(o["t_ns"], "pong.t_ns")
        return o
    _fail("server_msg", f"unknown op {op!r}")


def validate_root_msg(o) -> dict:
    """K17 root-helper traffic: requests {"op": hello|write|read|run|stop ...} and replies {"ok": ...}."""
    w = "root_msg"
    _dict(o, w)
    if "op" in o:
        op = o["op"]
        if op == "hello":
            _keys(o, ("op", "pid", "uid", "fake_root"), w)
            _int(o["pid"], f"{w}.pid", lo=0)
            _int(o["uid"], f"{w}.uid")
            _str(o["fake_root"], f"{w}.fake_root", null=True)
        elif op == "write":
            _keys(o, ("op", "path", "value"), w)
            _str(o["path"], f"{w}.path")
            if not isinstance(o["value"], (str, int, float)) or isinstance(o["value"], bool):
                _fail(w, f"write value must be str or number, got {o['value']!r}")
        elif op == "read":
            _keys(o, ("op", "path"), w)
            _str(o["path"], f"{w}.path")
        elif op == "run":
            _keys(o, ("op", "argv", "timeout_s"), w)
            for a in _list(o["argv"], f"{w}.argv", min_len=1):
                _str(a, f"{w}.argv")
            _num(o["timeout_s"], f"{w}.timeout_s", positive=True)
        elif op == "stop":
            _keys(o, ("op",), w)
        else:
            _fail(w, f"unknown op {op!r}")
        return o
    if "ok" not in o:
        _fail(w, "needs 'op' (request) or 'ok' (reply)")
    ok = _bool(o["ok"], f"{w}.ok")
    if not ok:
        _keys(o, ("ok", "err"), w)
        _str(o["err"], f"{w}.err")
    elif "readback" in o:
        _keys(o, ("ok", "readback"), w)
        if not isinstance(o["readback"], str):
            _fail(w, "readback must be a string")
    elif "value" in o:
        _keys(o, ("ok", "value"), w)
        if not isinstance(o["value"], str):
            _fail(w, "value must be a string")
    elif "rc" in o:
        _keys(o, ("ok", "rc", "out"), w)
        _int(o["rc"], f"{w}.rc")
        if not isinstance(o["out"], str):
            _fail(w, "out must be a string")
    else:
        _keys(o, ("ok",), w)
    return o


def validate_plan(o) -> dict:
    """K15. freeze is required for kind 'open' (path + content_sha256) and null for 'closed'. mode 'open'
    conditions need kind 'open'. sha256 must equal plan_sha256(plan)."""
    w = "plan"
    _keys(o, ("schema", "kind", "session", "freeze", "conditions", "estimated_wall_s", "sha256"), w,
          optional=("order_seed",))
    _schema(o, "plan", w)
    kind = _str(o["kind"], f"{w}.kind", choices=("open", "closed"))
    if "order_seed" in o:
        _int(o["order_seed"], f"{w}.order_seed", lo=0)
        if kind != "open":
            _fail(w, "order_seed is written only by open plans")
    s = o["session"]
    _keys(s, ("families", "exits", "emc_mhz", "settle_s", "idle_window_s", "n_inputs", "preprocess"), f"{w}.session")
    fams = _list(s["families"], f"{w}.session.families", min_len=1)
    for f in fams:
        _str(f, f"{w}.session.families", choices=FAMILIES)
    exits = _dict(s["exits"], f"{w}.session.exits")
    if sorted(exits) != sorted(fams):
        _fail(w, f"session.exits keys {sorted(exits)} != session.families {sorted(fams)}")
    for f, es in exits.items():
        for e in _list(es, f"{w}.session.exits[{f}]", min_len=1):
            _check_family_exit(f, e)
    emc = _num(s["emc_mhz"], f"{w}.session.emc_mhz", null=True)
    if emc is not None and emc not in EMC_LOCKABLE_MHZ:
        _fail(w, f"session.emc_mhz {emc} not in {EMC_LOCKABLE_MHZ}")
    _num(s["settle_s"], f"{w}.session.settle_s", lo=0.0)
    _num(s["idle_window_s"], f"{w}.session.idle_window_s", lo=0.0)
    _int(s["n_inputs"], f"{w}.session.n_inputs", lo=1)
    _str(s["preprocess"], f"{w}.session.preprocess", choices=_PREPROCESS)

    def known(tid, where):
        fam, ex = parse_type_id(tid)
        if fam not in exits or ex not in exits[fam]:
            _fail(where, f"{tid} is not covered by session.exits")

    fr = o["freeze"]
    if kind == "open":
        _keys(fr, ("path", "content_sha256"), f"{w}.freeze")
        _str(fr["path"], f"{w}.freeze.path")
        _hex(fr["content_sha256"], f"{w}.freeze.content_sha256")
    elif fr is not None:
        _fail(w, "freeze must be null for a closed plan")
    ids = set()
    for c in _list(o["conditions"], f"{w}.conditions", min_len=1):
        _dict(c, f"{w}.conditions[]")
        mode = _str(c.get("mode"), f"{w}.conditions[].mode", choices=_MODES)
        cw = f"{w}.conditions[{c.get('condition_id')}]"
        base = ("condition_id", "mode", "arm", "gpu_mhz")
        if mode == "idle":
            _keys(c, base + ("duration_s",), cw)
        elif mode == "closed":
            _keys(c, base + ("cell", "duration_s", "n_warmup", "split", "repeat"), cw)
        else:
            _keys(c, base + ("policy_key", "freeze_condition_id", "policy", "workload", "alpha", "n_requests",
                             "n_warmup", "seed", "repeat"), cw, optional=("confirm_of", "deadlines_s"))
        cid = _cond_id(c["condition_id"], f"{cw}.condition_id")
        if cid in ids:
            _fail(cw, "duplicate condition_id")
        ids.add(cid)
        arm = _str(c["arm"], f"{cw}.arm", choices=ARMS)
        if arm == "pinned":
            _int(c["gpu_mhz"], f"{cw}.gpu_mhz", lo=1)
        elif c["gpu_mhz"] is not None:
            _fail(cw, f"arm {arm!r} needs gpu_mhz null")
        if mode in ("idle", "closed"):
            _num(c["duration_s"], f"{cw}.duration_s", positive=True)
        if mode == "closed":
            for t in _list(c["cell"], f"{cw}.cell", min_len=1):
                known(t, cw)
            _int(c["n_warmup"], f"{cw}.n_warmup", lo=0)
            _str(c["split"], f"{cw}.split", choices=_SPLITS)
            _int(c["repeat"], f"{cw}.repeat", lo=0)
        if mode == "open":
            if kind != "open":
                _fail(cw, "open conditions need plan kind 'open'")
            _str(c["policy_key"], f"{cw}.policy_key")
            _str(c["freeze_condition_id"], f"{cw}.freeze_condition_id")
            validate_policyspec(c["policy"])
            validate_workload(c["workload"])
            for t in c["workload"]["types"]:
                known(t, cw)
            _num(c["alpha"], f"{cw}.alpha", positive=True, null=True)
            _int(c["n_requests"], f"{cw}.n_requests", lo=1)
            _int(c["n_warmup"], f"{cw}.n_warmup", lo=0)
            _int(c["seed"], f"{cw}.seed")
            _int(c["repeat"], f"{cw}.repeat", lo=0)
            if "confirm_of" in c:
                _str(c["confirm_of"], f"{cw}.confirm_of")
                if c["repeat"] not in CONFIRM_REPEATS:
                    _fail(cw, f"a confirm condition needs repeat in {CONFIRM_REPEATS}, got {c['repeat']}")
            wfams = sorted({parse_type_id(t)[0] for t in c["workload"]["types"]})
            if "deadlines_s" in c:
                dl = _dict(c["deadlines_s"], f"{cw}.deadlines_s")
                if sorted(dl) != wfams:
                    _fail(cw, f"deadlines_s keys {sorted(dl)} != workload families {wfams}")
                for f, v in dl.items():
                    _num(v, f"{cw}.deadlines_s.{f}", positive=True)
                if c["alpha"] is not None:
                    _fail(cw, "deadlines_s needs alpha null")
            if "exit_rule" in c["policy"].get("params", {}):
                for f in wfams:
                    if f not in EXIT_MENU:
                        _fail(cw, f"family {f!r} of an exit policy is not in EXIT_MENU")
                    if not set(EXIT_MENU[f]) <= set(exits.get(f, ())):
                        _fail(cw, f"session.exits[{f}] must load the exits {EXIT_MENU[f]} of the exit menu")
    _num(o["estimated_wall_s"], f"{w}.estimated_wall_s", lo=0.0)
    _hex(o["sha256"], f"{w}.sha256")
    if o["sha256"] != plan_sha256(o):
        _fail(w, f"sha256 {o['sha256']} != plan_sha256 {plan_sha256(o)}")
    return o


def validate_freeze(o) -> dict:
    """K16. conditions[].choices[key] and grid[i] are PolicySpec json; predictions[key] has "per_type" and
    "energy_per_req_j"; content_sha256 must equal content_sha256(o)."""
    w = "freeze"
    _keys(o, ("schema", "created_utc", "created_wall_ns", "git", "inputs", "slowdown_model", "sim", "constraint",
              "f_grid_mhz", "conditions", "runtime_estimate_s", "content_sha256"), w)
    _schema(o, "freeze", w)
    if not re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", str(o["created_utc"])):
        _fail(w, f"created_utc must be YYYY-MM-DDTHH:MM:SSZ, got {o['created_utc']!r}")
    _int(o["created_wall_ns"], f"{w}.created_wall_ns", lo=1)
    g = o["git"]
    _keys(g, ("commit", "dirty", "diff_sha256"), f"{w}.git")
    _str(g["commit"], f"{w}.git.commit")
    _bool(g["dirty"], f"{w}.git.dirty")
    _hex(g["diff_sha256"], f"{w}.git.diff_sha256", null=True)
    inp = o["inputs"]
    _keys(inp, ("types_sha256", "corun_sha256", "solo_sha256", "run1_plan_sha256", "slowdown_sha256", "power_sha256"),
          f"{w}.inputs", optional=("run1b_plan_sha256",))
    for k, v in inp.items():
        _hex(v, f"{w}.inputs.{k}")
    sm = o["slowdown_model"]
    _keys(sm, ("selected", "heldout_mape", "target_met"), f"{w}.slowdown_model")
    _str(sm["selected"], f"{w}.slowdown_model.selected", choices=_SLOWDOWN_NAMES)
    for k, v in _dict(sm["heldout_mape"], f"{w}.slowdown_model.heldout_mape").items():
        _str(k, f"{w}.slowdown_model.heldout_mape key", choices=_SLOWDOWN_NAMES)
        _num(v, f"{w}.slowdown_model.heldout_mape[{k}]", lo=0.0)
    _bool(sm["target_met"], f"{w}.slowdown_model.target_met")
    sim = o["sim"]
    _keys(sim, ("n_requests", "n_warmup", "n_reps", "seed"), f"{w}.sim")
    for k in ("n_requests", "n_reps"):
        _int(sim[k], f"{w}.sim.{k}", lo=1)
    _int(sim["n_warmup"], f"{w}.sim.n_warmup", lo=0)
    _int(sim["seed"], f"{w}.sim.seed")
    _keys(o["constraint"], ("headline", "sensitivity"), f"{w}.constraint")
    grid = [_int(f, f"{w}.f_grid_mhz", lo=1) for f in _list(o["f_grid_mhz"], f"{w}.f_grid_mhz", min_len=1)]
    if grid != sorted(set(grid)):
        _fail(w, "f_grid_mhz must be strictly ascending")
    ids = set()
    for c in _list(o["conditions"], f"{w}.conditions", min_len=1):
        cw = f"{w}.conditions[{c.get('condition_id') if isinstance(c, dict) else '?'}]"
        _keys(c, ("condition_id", "workload", "rho", "arrival", "cv_a", "alpha", "choices", "grid", "predictions",
                  "game", "compute_s"), cw)
        cid = _str(c["condition_id"], f"{cw}.condition_id")
        if cid in ids:
            _fail(cw, "duplicate condition_id")
        ids.add(cid)
        validate_workload(c["workload"])
        wl = c["workload"]
        if (c["rho"], c["arrival"], c["cv_a"]) != (wl["rho"], wl["arrival"], wl["cv_a"]):
            _fail(cw, "rho/arrival/cv_a must repeat the workload's values")
        _num(c["alpha"], f"{cw}.alpha", positive=True, null=True)
        choices = _dict(c["choices"], f"{cw}.choices")
        if not choices:
            _fail(cw, "choices is empty")
        for k, spec in choices.items():
            validate_policyspec(spec)
        for spec in _list(c["grid"], f"{cw}.grid", min_len=1):
            validate_policyspec(spec)
        for k, p in _dict(c["predictions"], f"{cw}.predictions").items():
            _dict(p, f"{cw}.predictions[{k}]")
            if "per_type" not in p or "energy_per_req_j" not in p:
                _fail(cw, f"predictions[{k}] needs per_type and energy_per_req_j")
        gm = c["game"]
        _keys(gm, ("player_cost", "equilibria", "worst", "optimum", "poa_E", "poa_R"), f"{cw}.game",
              optional=("optimum_constrained", "poa_eps", "tie_tol"))
        new_keys = [k for k in ("optimum_constrained", "poa_eps", "tie_tol") if k in gm]
        if new_keys and len(new_keys) != 3:
            _fail(f"{cw}.game", "optimum_constrained, poa_eps and tie_tol come together (D32-8)")
        _str(gm["player_cost"], f"{cw}.game.player_cost")
        for e in _list(gm["equilibria"], f"{cw}.game.equilibria"):
            ew = f"{cw}.game.equilibria[]"
            _keys(e, ("k_short", "k_long"), ew, optional=("start", "converged", "iters", "cycle"))
            if "start" in e:
                _keys(e, ("start", "k_short", "k_long", "converged", "iters", "cycle"), ew)
                _bool(e["converged"], f"{ew}.converged")
            _int(e["k_short"], f"{ew}.k_short", lo=1)
            _int(e["k_long"], f"{ew}.k_long", lo=1)
        _dict(gm["worst"], f"{cw}.game.worst")
        _dict(gm["optimum"], f"{cw}.game.optimum")
        for k in ("poa_E", "poa_R"):
            _num(gm[k], f"{cw}.game.{k}", null=True)
        if new_keys:
            oc = gm["optimum_constrained"]
            if oc is not None:
                _keys(oc, ("k_short", "k_long"), f"{cw}.game.optimum_constrained")
                _int(oc["k_short"], f"{cw}.game.optimum_constrained.k_short", lo=1)
                _int(oc["k_long"], f"{cw}.game.optimum_constrained.k_long", lo=1)
            _num(gm["poa_eps"], f"{cw}.game.poa_eps", positive=True, null=True)
            _num(gm["tie_tol"], f"{cw}.game.tie_tol", lo=0.0)
            if oc is None and gm["poa_eps"] is not None:
                _fail(f"{cw}.game", "poa_eps must be null when optimum_constrained is null")
        for k, v in _dict(c["compute_s"], f"{cw}.compute_s").items():
            _num(v, f"{cw}.compute_s[{k}]", lo=0.0)
    _num(o["runtime_estimate_s"], f"{w}.runtime_estimate_s", lo=0.0)
    _hex(o["content_sha256"], f"{w}.content_sha256")
    if o["content_sha256"] != content_sha256(o):
        _fail(w, f"content_sha256 {o['content_sha256']} != recomputed {content_sha256(o)}")
    return o


def _validate_window(v, where):
    _keys(v, ("t0_ns", "t1_ns", "gpu_mhz"), where)
    t0, t1 = _int(v["t0_ns"], f"{where}.t0_ns"), _int(v["t1_ns"], f"{where}.t1_ns")
    if t1 <= t0:
        _fail(where, "t1_ns must be > t0_ns")
    _int(v["gpu_mhz"], f"{where}.gpu_mhz", lo=1, null=True)


def validate_session(o) -> dict:
    """K9 session.json. Servers start one at a time: t_spawn <= t_ready, and t_ready(i) <= t_spawn(i+1)."""
    w = "session"
    _keys(o, ("schema", "session_id", "plan_sha256", "plan_kind", "freeze_content_sha256", "t_wall_start_ns",
              "t_mono_start_ns", "t_wall_end_ns", "git", "host", "board", "idle", "servers", "conditions",
              "exit_code"), w)
    _schema(o, "session", w)
    _str(o["session_id"], f"{w}.session_id")
    _hex(o["plan_sha256"], f"{w}.plan_sha256")
    kind = _str(o["plan_kind"], f"{w}.plan_kind", choices=("open", "closed"))
    _hex(o["freeze_content_sha256"], f"{w}.freeze_content_sha256", null=(kind == "closed"))
    if kind == "closed" and o["freeze_content_sha256"] is not None:
        _fail(w, "closed sessions have freeze_content_sha256 null")
    _int(o["t_wall_start_ns"], f"{w}.t_wall_start_ns", lo=1)
    _int(o["t_mono_start_ns"], f"{w}.t_mono_start_ns")
    _int(o["t_wall_end_ns"], f"{w}.t_wall_end_ns", null=True)
    g = o["git"]
    _keys(g, ("commit", "dirty", "diff_sha256"), f"{w}.git")
    _str(g["commit"], f"{w}.git.commit")
    _bool(g["dirty"], f"{w}.git.dirty")
    _hex(g["diff_sha256"], f"{w}.git.diff_sha256", null=True)
    h = o["host"]
    _keys(h, ("python", "numpy", "platform", "boot_id"), f"{w}.host")
    for k in h:
        _str(h[k], f"{w}.host.{k}")
    b = o["board"]
    _keys(b, ("dry", "snapshot_start", "snapshot_end", "jetson_clocks_store"), f"{w}.board")
    _bool(b["dry"], f"{w}.board.dry")
    _dict(b["snapshot_start"], f"{w}.board.snapshot_start")
    _dict(b["snapshot_end"], f"{w}.board.snapshot_end")
    _str(b["jetson_clocks_store"], f"{w}.board.jetson_clocks_store", null=True)
    idle = o["idle"]
    _keys(idle, ("pre", "resident"), f"{w}.idle")
    _validate_window(idle["pre"], f"{w}.idle.pre")
    _validate_window(idle["resident"], f"{w}.idle.resident")
    spawns = []
    for fam, sv in _dict(o["servers"], f"{w}.servers").items():
        _str(fam, f"{w}.servers key", choices=FAMILIES)
        sw = f"{w}.servers[{fam}]"
        _keys(sv, ("pid", "t_spawn_ns", "t_ready_ns", "t_spawn_wall_ns", "ready"), sw)
        _int(sv["pid"], f"{sw}.pid", lo=0)
        ts, tr = _int(sv["t_spawn_ns"], f"{sw}.t_spawn_ns"), _int(sv["t_ready_ns"], f"{sw}.t_ready_ns")
        _int(sv["t_spawn_wall_ns"], f"{sw}.t_spawn_wall_ns")
        if tr < ts:
            _fail(sw, "t_ready_ns < t_spawn_ns")
        if _dict(sv["ready"], f"{sw}.ready"):
            validate_ready(sv["ready"])
        spawns.append((ts, tr, fam))
    spawns.sort()
    for (_, tr, fam), (ts2, _, fam2) in zip(spawns, spawns[1:]):
        if tr > ts2:
            _fail(w, f"server {fam2} was spawned before server {fam} was ready (servers must start one at a time)")
    conds = [_cond_id(c, f"{w}.conditions[]") for c in _list(o["conditions"], f"{w}.conditions")]
    if len(set(conds)) != len(conds):
        _fail(w, "duplicate condition ids")
    _int(o["exit_code"], f"{w}.exit_code")
    return o


def validate_manifest(o) -> dict:
    """K9 manifest.json. valid False needs invalid_reason; open needs policy_spec + workload; closed needs
    cell + split."""
    w = "manifest"
    _keys(o, ("schema", "condition_id", "mode", "arm", "gpu_mhz", "emc_mhz", "t_wall_start_ns", "t_mono_start_ns",
              "t_mono_end_ns", "settle_s", "valid", "invalid_reason", "board_readback", "n_requests", "n_warmup",
              "seed",
              "policy_key", "policy_spec", "freeze_condition_id", "workload", "alpha", "cell", "split", "repeat"), w)
    _schema(o, "manifest", w)
    _cond_id(o["condition_id"], f"{w}.condition_id")
    mode = _str(o["mode"], f"{w}.mode", choices=_MODES)
    arm = _str(o["arm"], f"{w}.arm", choices=ARMS)
    if arm == "pinned":
        _int(o["gpu_mhz"], f"{w}.gpu_mhz", lo=1)
    elif o["gpu_mhz"] is not None:
        _fail(w, f"arm {arm!r} needs gpu_mhz null")
    _num(o["emc_mhz"], f"{w}.emc_mhz", null=True)
    _int(o["t_wall_start_ns"], f"{w}.t_wall_start_ns", lo=1)
    t0 = _int(o["t_mono_start_ns"], f"{w}.t_mono_start_ns")
    t1 = _int(o["t_mono_end_ns"], f"{w}.t_mono_end_ns", null=True)
    if t1 is not None and t1 < t0:
        _fail(w, "t_mono_end_ns < t_mono_start_ns")
    _num(o["settle_s"], f"{w}.settle_s", lo=0.0)
    valid = _bool(o["valid"], f"{w}.valid")
    _str(o["invalid_reason"], f"{w}.invalid_reason", null=True)
    if valid != (o["invalid_reason"] is None):
        _fail(w, "invalid_reason must be set exactly when valid is false")
    _dict(o["board_readback"], f"{w}.board_readback", null=True)
    _int(o["n_requests"], f"{w}.n_requests", lo=0, null=True)
    _int(o["n_warmup"], f"{w}.n_warmup", lo=0, null=True)
    _int(o["seed"], f"{w}.seed", null=True)
    _int(o["repeat"], f"{w}.repeat", lo=0, null=True)
    _num(o["alpha"], f"{w}.alpha", positive=True, null=True)
    if mode == "open":
        _str(o["policy_key"], f"{w}.policy_key")
        _str(o["freeze_condition_id"], f"{w}.freeze_condition_id")
        validate_policyspec(o["policy_spec"])
        validate_workload(o["workload"])
        for k in ("n_requests", "n_warmup", "seed"):
            if o[k] is None:
                _fail(w, f"open manifest needs {k}")
    elif mode == "closed":
        for t in _list(o["cell"], f"{w}.cell", min_len=1):
            parse_type_id(t)
        _str(o["split"], f"{w}.split", choices=_SPLITS)
    return o


def validate_simresult(o) -> dict:
    """K14. Per type: rates in [0,1] and complete+truncation+miss+expired == 1 when n > 0."""
    w = "simresult"
    _keys(o, ("schema", "spec_name", "per_type", "energy_per_req_j", "energy_ci", "mean_power_w", "throughput",
              "mean_in_system", "utilization", "unstable", "tagged", "reps", "trace", "runtime_s"), w)
    _schema(o, "simresult", w)
    _str(o["spec_name"], f"{w}.spec_name")
    for tid, st in _dict(o["per_type"], f"{w}.per_type").items():
        parse_type_id(tid)
        sw = f"{w}.per_type[{tid}]"
        _keys(st, _TYPE_STATS_KEYS, sw)
        _int(st["n"], f"{sw}.n", lo=0)
        for k in _TYPE_STATS_KEYS[1:5]:
            _num(st[k], f"{sw}.{k}", lo=0.0)
        for k in _TYPE_STATS_KEYS[5:]:
            _num(st[k], f"{sw}.{k}", lo=0.0, hi=1.0)
        total = st["complete_rate"] + st["truncation_rate"] + st["miss_rate"] + st["expired_rate"]
        if st["n"] > 0 and abs(total - 1.0) > 1e-6:
            _fail(sw, f"outcome rates sum to {total}, not 1")
    _num(o["energy_per_req_j"], f"{w}.energy_per_req_j", lo=0.0)
    ci = _list(o["energy_ci"], f"{w}.energy_ci")
    if len(ci) != 2 or not all(_is_num(x) for x in ci) or ci[0] > ci[1]:
        _fail(w, f"energy_ci must be [lo, hi], got {ci!r}")
    for k in ("mean_power_w", "throughput", "mean_in_system", "utilization", "runtime_s"):
        _num(o[k], f"{w}.{k}", lo=0.0)
    _bool(o["unstable"], f"{w}.unstable")
    if o["tagged"] is not None:
        _keys(o["tagged"], ("klass", "k_dev") + _TYPE_STATS_KEYS, f"{w}.tagged")
    for r in _list(o["reps"], f"{w}.reps"):
        _keys(r, ("seed", "energy_per_req_j", "n_slope", "per_type", "tagged"), f"{w}.reps[]")
    if o["trace"] is not None:
        _list(o["trace"], f"{w}.trace")
    return o


def validate_slowdown(o) -> dict:
    w = "slowdown"
    _keys(o, ("schema", "name", "params", "residuals", "heldout_mape", "fit_rows_sha256"), w, optional=("selection",))
    _schema(o, "slowdown", w)
    _str(o["name"], f"{w}.name", choices=_SLOWDOWN_NAMES)
    if "selection" in o:
        sel = o["selection"]
        _keys(sel, ("method", "loco_mdape", "selected"), f"{w}.selection")
        _str(sel["method"], f"{w}.selection.method", choices=("loco",))
        for k, v in _dict(sel["loco_mdape"], f"{w}.selection.loco_mdape").items():
            _str(k, f"{w}.selection.loco_mdape key", choices=_SLOWDOWN_NAMES)
            _num(v, f"{w}.selection.loco_mdape[{k}]", lo=0.0)
        _str(sel["selected"], f"{w}.selection.selected", choices=_SLOWDOWN_NAMES)
        if sel["selected"] != o["name"]:
            _fail(w, f"selection.selected {sel['selected']!r} != name {o['name']!r}")
        if sel["selected"] not in sel["loco_mdape"]:
            _fail(w, "selection.selected is missing from selection.loco_mdape")
    _dict(o["params"], f"{w}.params")
    for k, xs in _dict(o["residuals"], f"{w}.residuals").items():
        if not re.fullmatch(r"[a-z]+\|\d+", k):
            _fail(w, f"residual pool key {k!r} is not '<family>|<n>'")
        for x in _list(xs, f"{w}.residuals[{k}]"):
            _num(x, f"{w}.residuals[{k}]", positive=True)
    for k, v in _dict(o["heldout_mape"], f"{w}.heldout_mape").items():
        _str(k, f"{w}.heldout_mape key", choices=_SLOWDOWN_NAMES)
        _num(v, f"{w}.heldout_mape[{k}]", lo=0.0)
    _hex(o["fit_rows_sha256"], f"{w}.fit_rows_sha256")
    return o


def validate_power(o) -> dict:
    w = "power"
    _keys(o, ("schema", "p_idle_w", "dp_w", "eta"), w)
    _schema(o, "power", w)
    for k in ("p_idle_w", "dp_w", "eta"):
        _level_map(o[k], f"{w}.{k}", positive=(k != "dp_w"), lo=0.0 if k == "dp_w" else None)
    levels = {int(f) for f in o["p_idle_w"]}
    for k in ("dp_w", "eta"):
        if {int(f) for f in o[k]} != levels:
            _fail(w, f"{k} levels differ from p_idle_w levels {sorted(levels)}")
    return o


def _stat(v, where):
    _keys(v, ("log_ratio_mean", "ci_lo", "ci_hi", "p", "p_holm", "n_pairs"), where)
    _num(v["log_ratio_mean"], f"{where}.log_ratio_mean")
    lo, hi = _num(v["ci_lo"], f"{where}.ci_lo"), _num(v["ci_hi"], f"{where}.ci_hi")
    if lo > hi:
        _fail(where, f"ci_lo {lo} > ci_hi {hi}")
    for k in ("p", "p_holm"):
        _num(v[k], f"{where}.{k}", lo=0.0, hi=1.0)
    _int(v["n_pairs"], f"{where}.n_pairs", lo=2)


def validate_open_summary(o) -> dict:
    """exp2.open_summary/3 (C2b + D32-4/5/6); /2 documents (without the /3 keys) are still accepted.
    counts: runs_valid, runs_invalid, runs_duplicate, runs_screen_flagged (ints >= 0).
    per_condition[<fcid>_<key>]: status feasible|borderline|infeasible (/3: bootstrap rule), wilson_lo_min (Wilson 95%
    lower bound, min over types), n_repeats, measured_key; /3 adds attain_boot_lo_min and energy_per_work_j (null ok).
    energy_diff[<fcid>_<key>]: paired difference of E_F - E_key per repeat (method "paired_t", n_pairs >= 2).
    poa_detail[fcid]: worst_key, optimum_key, ratio and complete_min_* (null when the key was not measured).
    /3 top level: primary (exactly PRIMARY_REFS), secondary (subset of SECONDARY_REFS), sim_accuracy."""
    w = "open_summary"
    v3 = isinstance(o, dict) and o.get("schema") == SCHEMAS["open_summary"]
    base = ("schema", "freeze_content_sha256", "checks", "counts", "per_condition", "mape", "regret", "oracle",
            "energy_diff", "poa_detail", "poa_measured")
    if isinstance(o, dict) and "schema" in o and not v3 and o["schema"] != OPEN_SUMMARY_V2:
        _schema(o, "open_summary", w)
    _keys(o, base + (("primary", "secondary", "sim_accuracy") if v3 else ()), w)
    _hex(o["freeze_content_sha256"], f"{w}.freeze_content_sha256")
    ck = o["checks"]
    _keys(ck, ("hash_ok", "wall_order_ok", "t_wall_start_ns", "freeze_created_wall_ns"), f"{w}.checks")
    _bool(ck["hash_ok"], f"{w}.checks.hash_ok")
    _bool(ck["wall_order_ok"], f"{w}.checks.wall_order_ok")
    _int(ck["t_wall_start_ns"], f"{w}.checks.t_wall_start_ns", lo=1)
    _int(ck["freeze_created_wall_ns"], f"{w}.checks.freeze_created_wall_ns", lo=1)
    cn = o["counts"]
    _keys(cn, ("runs_valid", "runs_invalid", "runs_duplicate", "runs_screen_flagged"), f"{w}.counts")
    for k, v in cn.items():
        _int(v, f"{w}.counts.{k}", lo=0)
    for cid, pc in _dict(o["per_condition"], f"{w}.per_condition").items():
        pw = f"{w}.per_condition[{cid}]"
        _keys(pc, ("energy", "energy_ci", "complete_rate_min", "feasible", "pred", "ape", "status", "wilson_lo_min",
                   "n_repeats", "measured_key") + (("attain_boot_lo_min", "energy_per_work_j") if v3 else ()), pw)
        _str(pc["status"], f"{pw}.status", choices=OPEN_STATUSES)
        _num(pc["wilson_lo_min"], f"{pw}.wilson_lo_min", lo=0.0, hi=1.0)
        _int(pc["n_repeats"], f"{pw}.n_repeats", lo=1)
        _str(pc["measured_key"], f"{pw}.measured_key")
        if v3:
            _num(pc["attain_boot_lo_min"], f"{pw}.attain_boot_lo_min", lo=0.0, hi=1.0)
            _num(pc["energy_per_work_j"], f"{pw}.energy_per_work_j", lo=0.0, null=True)
    _keys(o["mape"], ("completion", "energy", "median", "max"), f"{w}.mape")
    _keys(o["regret"], ("F", "grid_spread", "best_key"), f"{w}.regret")
    _dict(o["oracle"], f"{w}.oracle", null=True)
    for k, d in _dict(o["energy_diff"], f"{w}.energy_diff").items():
        _keys(d, ("diff_j", "ci", "n_pairs", "method"), f"{w}.energy_diff[{k}]")
        _int(d["n_pairs"], f"{w}.energy_diff[{k}].n_pairs", lo=2)
        _str(d["method"], f"{w}.energy_diff[{k}].method", choices=("paired_t",))
    for fcid, d in _dict(o["poa_detail"], f"{w}.poa_detail").items():
        _keys(d, ("worst_key", "optimum_key", "ratio", "complete_min_worst", "complete_min_optimum"),
              f"{w}.poa_detail[{fcid}]")
        _str(d["worst_key"], f"{w}.poa_detail[{fcid}].worst_key")
        _str(d["optimum_key"], f"{w}.poa_detail[{fcid}].optimum_key")
        _num(d["ratio"], f"{w}.poa_detail[{fcid}].ratio", positive=True, null=True)
        for k in ("complete_min_worst", "complete_min_optimum"):
            _num(d[k], f"{w}.poa_detail[{fcid}].{k}", lo=0.0, hi=1.0, null=True)
    if v3:
        prim = _dict(o["primary"], f"{w}.primary")
        if sorted(prim) != sorted(PRIMARY_REFS):
            _fail(w, f"primary keys {sorted(prim)} != {sorted(PRIMARY_REFS)}")
        for ref, st in prim.items():
            _stat(st, f"{w}.primary[{ref}]")
        for ref, st in _dict(o["secondary"], f"{w}.secondary").items():
            _str(ref, f"{w}.secondary key", choices=SECONDARY_REFS)
            _stat(st, f"{w}.secondary[{ref}]")
        for fcid, d in _dict(o["sim_accuracy"], f"{w}.sim_accuracy").items():
            sw = f"{w}.sim_accuracy[{fcid}]"
            _keys(d, ("abs_err_pp_max", "abs_err_pp_median", "class_agreement", "spearman_energy",
                      "energy_ape_median"), sw)
            _num(d["abs_err_pp_max"], f"{sw}.abs_err_pp_max", lo=0.0)
            _num(d["abs_err_pp_median"], f"{sw}.abs_err_pp_median", lo=0.0)
            _num(d["class_agreement"], f"{sw}.class_agreement", lo=0.0, hi=1.0)
            _num(d["spearman_energy"], f"{sw}.spearman_energy", lo=-1.0, hi=1.0, null=True)
            _num(d["energy_ape_median"], f"{sw}.energy_ape_median", lo=0.0, null=True)
    return o


def validate_backtest(o) -> dict:
    """exp2.backtest/1 (D32-6): predicted versus measured per-tenant median latency and board energy on held-out
    Run 1 cells. ape is percent: 100 * |pred - meas| / meas (checked to 1e-3). summary medians are null for an
    empty list."""
    w = "backtest"
    _keys(o, ("schema", "cells", "energy", "summary"), w)
    _schema(o, "backtest", w)

    def ape_ok(row, pred, meas, where):
        want = 100.0 * abs(row[pred] - row[meas]) / row[meas]
        if abs(row["ape"] - want) > 1e-3 * max(1.0, want):
            _fail(where, f"ape {row['ape']} != 100*|pred-meas|/meas = {want}")

    for r in _list(o["cells"], f"{w}.cells"):
        rw = f"{w}.cells[]"
        _keys(r, ("cell_id", "f_mhz", "type_id", "pred_tau_p50_s", "meas_tau_p50_s", "ape"), rw)
        _str(r["cell_id"], f"{rw}.cell_id")
        _int(r["f_mhz"], f"{rw}.f_mhz", lo=1)
        parse_type_id(r["type_id"])
        _num(r["pred_tau_p50_s"], f"{rw}.pred_tau_p50_s", positive=True)
        _num(r["meas_tau_p50_s"], f"{rw}.meas_tau_p50_s", positive=True)
        _num(r["ape"], f"{rw}.ape", lo=0.0)
        ape_ok(r, "pred_tau_p50_s", "meas_tau_p50_s", rw)
    for r in _list(o["energy"], f"{w}.energy"):
        rw = f"{w}.energy[]"
        _keys(r, ("cell_id", "f_mhz", "pred_j", "meas_j", "ape"), rw)
        _str(r["cell_id"], f"{rw}.cell_id")
        _int(r["f_mhz"], f"{rw}.f_mhz", lo=1)
        _num(r["pred_j"], f"{rw}.pred_j", positive=True)
        _num(r["meas_j"], f"{rw}.meas_j", positive=True)
        _num(r["ape"], f"{rw}.ape", lo=0.0)
        ape_ok(r, "pred_j", "meas_j", rw)
    _keys(o["summary"], ("tau_mdape", "energy_mdape"), f"{w}.summary")
    for k in ("tau_mdape", "energy_mdape"):
        _num(o["summary"][k], f"{w}.summary.{k}", lo=0.0, null=True)
    return o


def validate_pilot(o) -> dict:
    w = "pilot"
    _keys(o, ("schema", "gates", "dry"), w)
    _schema(o, "pilot", w)
    _bool(o["dry"], f"{w}.dry")
    for gid, g in _dict(o["gates"], f"{w}.gates").items():
        _keys(g, ("value", "threshold", "pass", "step"), f"{w}.gates[{gid}]")
        _bool(g["pass"], f"{w}.gates[{gid}].pass")
        _int(g["step"], f"{w}.gates[{gid}].step", lo=0)
    return o


def validate_fake_spec(o) -> dict:
    """seg_ms has exactly one entry per default checkpoint of the type."""
    w = "fake_spec"
    _keys(o, ("schema", "family", "clock", "load_s", "first_infer_s", "types"), w)
    _schema(o, "fake_spec", w)
    fam = _str(o["family"], f"{w}.family", choices=FAMILIES)
    _str(o["clock"], f"{w}.clock", choices=("sleep", "virtual"))
    _num(o["load_s"], f"{w}.load_s", lo=0.0)
    _num(o["first_infer_s"], f"{w}.first_infer_s", lo=0.0)
    for e, t in _dict(o["types"], f"{w}.types").items():
        ex = _int_key(e, w)
        _check_family_exit(fam, ex)
        _keys(t, ("seg_ms", "head_ms"), f"{w}.types[{e}]")
        segs = _list(t["seg_ms"], f"{w}.types[{e}].seg_ms")
        for x in segs:
            _num(x, f"{w}.types[{e}].seg_ms", positive=True)
        if len(segs) != len(default_checkpoints(fam, ex)):
            _fail(w, f"types[{e}].seg_ms has {len(segs)} entries, needs one per checkpoint "
                     f"({len(default_checkpoints(fam, ex))})")
        _num(t["head_ms"], f"{w}.types[{e}].head_ms", lo=0.0)
    return o


def validate_telemetry_meta(o) -> dict:
    w = "telemetry_meta"
    _keys(o, ("schema", "vdd_in", "target_hz", "achieved_hz", "clocks_target_hz", "devfreq", "emc_source",
              "thermal_zones", "boot_id", "t_mono_start_ns", "t_wall_start_ns", "n_rows"), w)
    _schema(o, "telemetry_meta", w)
    v = o["vdd_in"]
    _keys(v, ("hwmon", "channel", "label", "update_interval_ms", "samples"), f"{w}.vdd_in")
    _str(v["hwmon"], f"{w}.vdd_in.hwmon")
    _int(v["channel"], f"{w}.vdd_in.channel", lo=1)
    if v["label"] != "VDD_IN":
        _fail(w, f"vdd_in.label must be 'VDD_IN', got {v['label']!r}")
    _num(v["update_interval_ms"], f"{w}.vdd_in.update_interval_ms", null=True)
    _int(v["samples"], f"{w}.vdd_in.samples", lo=1, null=True)
    _num(o["target_hz"], f"{w}.target_hz", positive=True)
    _num(o["achieved_hz"], f"{w}.achieved_hz", lo=0.0)
    _num(o["clocks_target_hz"], f"{w}.clocks_target_hz", positive=True)
    _str(o["devfreq"], f"{w}.devfreq", null=True)
    _str(o["emc_source"], f"{w}.emc_source", null=True)
    for z in _list(o["thermal_zones"], f"{w}.thermal_zones"):
        _str(z, f"{w}.thermal_zones[]")
    _str(o["boot_id"], f"{w}.boot_id", null=True)
    _int(o["t_mono_start_ns"], f"{w}.t_mono_start_ns")
    _int(o["t_wall_start_ns"], f"{w}.t_wall_start_ns")
    _int(o["n_rows"], f"{w}.n_rows", lo=0)
    return o


VALIDATORS = {
    SCHEMAS["types"]: validate_types, SCHEMAS["plan"]: validate_plan, SCHEMAS["freeze"]: validate_freeze,
    SCHEMAS["session"]: validate_session, SCHEMAS["manifest"]: validate_manifest,
    SCHEMAS["simresult"]: validate_simresult, SCHEMAS["slowdown"]: validate_slowdown,
    SCHEMAS["power"]: validate_power, SCHEMAS["open_summary"]: validate_open_summary,
    SCHEMAS["pilot"]: validate_pilot, SCHEMAS["fake_spec"]: validate_fake_spec,
    SCHEMAS["telemetry_meta"]: validate_telemetry_meta,
    SCHEMAS["backtest"]: validate_backtest,
}


def validate_schema(o) -> dict:
    """Dispatch on o["schema"] to the matching validate_*; unknown or missing schema is a ContractError."""
    _dict(o, "object")
    sid = o.get("schema")
    if sid not in VALIDATORS:
        _fail("object", f"unknown or missing schema {sid!r}; known: {sorted(VALIDATORS)}")
    return VALIDATORS[sid](o)


# --------------------------------------------------------------------------------------------- csv and jsonl readers
def validate_csv_header(kind: str, header) -> None:
    """Exact column list for `kind` (RUN1_CSV_COLS kinds, power, freq_switch, open_conditions) or, for "clocks",
    CLOCKS_CSV_FIXED_COLS, then CLOCKS_CSV_OPTIONAL_COLS (the whole block or none), then sorted unique
    temp_<zone>_mc columns."""
    header = list(header)
    if kind == "clocks":
        fixed = list(CLOCKS_CSV_FIXED_COLS)
        temps = header[len(fixed):]
        if header[:len(fixed)] != fixed:
            _fail("clocks.csv", f"header must start with {fixed}, got {header}")
        opt = list(CLOCKS_CSV_OPTIONAL_COLS)
        if temps[:len(opt)] == opt:
            temps = temps[len(opt):]
        if not all(_TEMP_COL.fullmatch(c) for c in temps) or temps != sorted(set(temps)):
            _fail("clocks.csv", f"temperature columns must be sorted unique temp_<zone>_mc, got {temps}")
        return
    cols = {**RUN1_CSV_COLS, "power": POWER_CSV_COLS, "freq_switch": FREQ_SWITCH_COLS,
            "open_conditions": OPEN_CONDITIONS_COLS, "screen": SCREEN_CSV_COLS}.get(kind)
    if cols is None:
        _fail("csv", f"unknown kind {kind!r}; known: {CSV_KINDS}")
    if header != list(cols):
        _fail(f"{kind}.csv", f"header {header} != {list(cols)}")


def _csv_cell(kind, col, raw, where):
    if raw is None or raw == "":
        return None
    try:
        if col in _CSV_STR_COLS:
            return raw
        if kind == "open_conditions" and col == "f_mhz":
            try:
                return int(raw)
            except ValueError:
                return raw          # "governor"
        if col in _CSV_INT_COLS or (kind == "clocks" and col.startswith("temp_")):
            try:
                return int(raw)
            except ValueError:
                x = float(raw)
                if not x.is_integer():
                    raise
                return int(x)
        return float(raw)
    except ValueError:
        _fail(where, f"column {col!r}: cannot parse {raw!r}")


def read_csv(path, kind: str) -> list[dict]:
    """Read a CSV of a known kind: header checked exactly, cells typed (int / float / str), empty cell -> None."""
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        header = next(reader, None)
        if header is None:
            _fail(f"{kind}.csv", f"{path} is empty (no header)")
        validate_csv_header(kind, header)
        rows = []
        for n, cells in enumerate(reader, start=2):
            if not cells:
                continue
            where = f"{path}:{n}"
            if len(cells) != len(header):
                _fail(where, f"{len(cells)} cells for {len(header)} columns")
            rows.append({c: _csv_cell(kind, c, v, where) for c, v in zip(header, cells)})
    return rows


def read_requests(path) -> list[dict]:
    """Read requests.jsonl; every row goes through validate_request_row. Blank lines are skipped."""
    rows = []
    with open(path, encoding="utf-8") as fh:
        for n, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except ValueError as e:
                _fail(f"{path}:{n}", f"invalid JSON: {e}")
            try:
                validate_request_row(row)
            except ContractError as e:
                raise ContractError(f"{path}:{n}: {e}") from e
            rows.append(row)
    return rows


def load_types(path) -> TypeTable:
    with open(path, encoding="utf-8") as fh:
        return TypeTable.from_json(json.load(fh))
