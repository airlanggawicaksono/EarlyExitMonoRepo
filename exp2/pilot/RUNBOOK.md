# Pilot RUNBOOK (steps 0-9)

Run on the Jetson only. Nothing here runs on the laptop except `--dry-run`.

```
laptop: git bundle -> scp (Tailscale) -> board: separate worktree -> pilot gates -> pilot_summary.json
```

## 0. Rules

| Rule | Detail |
|---|---|
| Password | Type it at the `getpass` prompt, or `read -s JETSON_SUDO_PW; export JETSON_SUDO_PW` in your shell. Never write it to a file, argv or log. |
| Board main checkout | Stays at `872e66d`. Never edit or pull in it. |
| Push | No push. |
| Output | `OUT=~/exp2_pilot_$(date +%Y%m%d)`. Measurements go to `$OUT/measurements/step<N>.json`. |

## 1. Deliver the code

```bash
# laptop
git bundle create exp2.bundle exp2-dev
scp exp2.bundle <board-tailscale-name>:~/
# board (main checkout untouched at 872e66d)
cd /home/earlyexp/EarlyExitMonoRepo
git fetch ~/exp2.bundle exp2-dev:exp2-dev
git worktree add ~/exp2_wt exp2-dev
cd ~/exp2_wt && . /home/earlyexp/EarlyExitMonoRepo/venv/bin/activate
python -m exp2.pilot.pilot --dry-run --out /tmp/exp2_dry     # rc 0 before any real step
```

## 2. Gates

Run all gates: `python -m exp2.pilot.pilot --out $OUT` (exit 0 pass, 1 gate failed, 2 error).
Run one gate: add `--steps N`. A gate with no measurement file fails (value null).

| Step | Gate id | Threshold (pass) | Command / measurement |
|---|---|---|---|
| 0 | `inventory` | 1.0 (all required checks: `/etc/nv_tegra_release`, GPU devfreq, INA3221, `tegrastats` without root; MPS binary only recorded) | `python -m exp2.pilot.pilot --steps 0 --out $OUT`; also `ls /usr/bin/nvidia-cuda-mps-control` |
| 1 | `power_mode_id` | mode id == 2 (MAXN_SUPER) | `python -m exp2.pilot.pilot --steps 1 --out $OUT` (sudo prompt); then `python -m exp2.board.clocks snapshot` for the jetson_clocks store |
| 2 | `coresident_min_mem_mb` | >= 500 MB MemAvailable (idle and load), all 4 servers ready; else workloads use 3 families | 4 servers, 10 min idle then 10 min load; write `step2.json` `{all_ready, min_mem_available_mb_idle, min_mem_available_mb_load}` |
| 3 | `warm_over_cold_load` | worst family warm/cold load time <= 0.5 | per family cold then warm cache `ready` message; `step3.json` `{family: {cold_load_s, warm_load_s}}`; also run the torch test on the board |
| 4 | `gpu_lock_hold` | >= 0.99 of samples at the locked GPU MHz | `python -m exp2.board.clocks set-gpu 1020`, then `python -m exp2.board.telemetry --out $OUT/tel4 --duration 60`; `step4.json` `{observed_mhz: [...]}` |
| 5 | `emc_lock_hold` | >= 0.99 of samples with sysfs AND tegrastats EMC within 1% of 3199 MHz | `python -m exp2.board.clocks lock-emc 3199`, telemetry 60 s; `step5.json` `{sysfs_mhz, tegrastats_mhz}` |
| 6 | `switch_latency_s` | p95 (nearest rank) <= 0.5 s | repeated `set-gpu` between 612 and 1020 with read-back; `step6.json` `{latencies_s}` |
| 7 | `ckpt_overhead_frac` | worst family median(with)/median(without) - 1 <= 0.10 | solo closed-loop per family with and without checkpoints (also record event vs spin); `step7.json` `{family: {with_s, without_s}}` |
| 8 | `yolo_solo_cv` | solo YOLO latency CV < 0.1 | solo YOLO closed-loop at f_max; `step8.json` `{latencies_s}` |
| 9 | `energy_vs_f_h2` | 1.0 = energy per request (window energy incl. idle / completed) is not minimal at f_max for at least one family at one planned load; else f fixed at f_max | open-loop at `planned_loads_rho` for each f in `f_levels_mhz` (`exp2/configs/pilot.json`); `step9.json` `{family: {load: {f_mhz: {window_energy_j, n_done}}}}` |

Thresholds live in `exp2/configs/pilot.json`.

## 3. After the run

```bash
python -m exp2.pilot.pilot --out $OUT      # writes $OUT/pilot_summary.json
cat $OUT/pilot_summary.json                 # 10 gates, each value/threshold/pass/step
```

Copy `$OUT` back with scp. Remove the worktree when done: `git worktree remove ~/exp2_wt`.

## 4. Exact command sequence on the board (research flow)

Password: every `sudo` step below asks at the `getpass` prompt (or reads `JETSON_SUDO_PW` from your shell). Do not type it into a file.

```bash
# A. deploy (section 1), then in ~/exp2_wt with the venv active
export PYTHONDONTWRITEBYTECODE=1; OUT=~/exp2_pilot_$(date +%Y%m%d); mkdir -p $OUT

# B. step 0 + 1: inventory, MAXN_SUPER (id 2) through the root helper
cat /etc/nv_tegra_release; ls /usr/bin/nvidia-cuda-mps-control; which tegrastats
python -m exp2.pilot.pilot --steps 0,1 --out $OUT            # sudo prompt; sets mode 2 and reads it back

# C. clock pin through the root helper (GPU 1020 MHz, EMC 3199 MHz) and snapshot
python -m exp2.board.clocks snapshot
python -m exp2.board.clocks apply-arm pinned --gpu 1020 --emc 3199
python -m exp2.board.clocks snapshot

# D. telemetry (steps 4-5 samples; run while pinned)
python -m exp2.board.telemetry --out $OUT/tel --hz 20 --duration 60

# E. torch tests on the board (laptop never runs them)
python -m pytest exp2/tests/test_fam_bert.py exp2/tests/test_fam_vision.py exp2/tests/test_fam_llama.py \
    exp2/tests/test_fam_yolo.py exp2/tests/test_serve_families.py -q -p no:cacheprovider

# F. solo smoke run: one solo condition per family (real adapter)
python -m exp2.model.plan_run1 --exits bert=5,23 vision=6,24 llama=3,15 yolo=1,5 --solo-levels 1020 \
    --heldout-level 714 --corun-levels 612,1020 --repeats 1 --heldout-frac 0.3 --duration 10 --seed 0 --out $OUT/plan_smoke.json
python -m exp2.serve.run_session --plan $OUT/plan_smoke.json --types exp2/configs/example_types.json \
    --out $OUT/session_smoke                                  # sudo prompt; sessions pin clocks themselves

# G. co-residency (step 2): all four servers resident for 10 min idle, then 10 min under load.
#    Take min MemAvailable from $OUT/tel (telemetry) and write measurements/step2.json (section 2).

# H. rest of the gates (steps 3, 6-9 measurements per section 2), then the summary
python -m exp2.pilot.pilot --out $OUT ; echo rc=$?

# I. Run 1 launch (only after every pilot gate passed; check estimated_wall_s in the plan first)
python -m exp2.model.plan_run1 --exits bert=5,23 vision=6,24 llama=3,15 yolo=1,5 --solo-levels 408,612,816,1020 \
    --heldout-level 714 --corun-levels 612,1020 --repeats 3 --heldout-frac 0.3 --duration 30 --seed 0 --out $OUT/plan_run1.json
python -m exp2.serve.run_session --plan $OUT/plan_run1.json --types <types.json> --out $OUT/session_run1
```

Note: `plan_run1 --out` is a file path. `--types` takes the real types file; the repo example is `exp2/configs/example_types.json`.
