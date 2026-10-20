# Run 2 analysis plan (frozen before Run 2, 2026-10-20)

This file is committed and pushed together with freeze.json before any Run 2 measurement. It fixes the design
decisions, hypotheses and success criteria of the validation (Run 2) and the confirmation re-measurement (Run 2b).
Everything below was decided before any Run 2 simulation result or measurement existed.

## Profiling data behind the freeze

All profiling sessions ran on one NVIDIA Jetson Orin Nano 8 GB (JetPack R36.5, MAXN_SUPER, EMC 3199 MHz locked,
GPU pinned per condition, CPU 1728 MHz) with board code at commit 131f1f1. Merged with exp2.analyze.closed_loop
(Run 1, 1b, 1d, 1e; minimum 10 measured requests per tenant per condition) into run1_csv/.

| Session | Plan file | Plan sha256 |
|---|---|---|
| Run 1 (closed-loop profiling, 215 conditions) | `plan_run1_3fam.json` | `b83eeb41b65b3117c6bfb276ba809655e1adb541810cc16384b4275cb3710beb` |
| Run 1b (co-run at 816 MHz, 79) | `plan_run1b_816.json` | `c37c105002b77965e656ac3a312b21ac7b3803d1af14ea58bd1a06fb93495513` |
| Run 1c bert (cold/warm start) | `plan_run1c_bert.json` | `d2b39d5e6b6a448a4d1ec4dd071048dc717b440ce012de5655e0b0fbfcf06c2e` |
| Run 1c llama | `plan_run1c_llama.json` | `5bd29641932b3e85cca865320cdf2d64f32be426351b5b8295693d7be742d565` |
| Run 1c yolo | `plan_run1c_yolo.json` | `7d0e938e9aa14b0df2474fbaa7f0e5aa48ccd88d76eb9e45a53e556615fbc859` |
| Run 1d (LLaMA co-run cells, longer windows, 99) | `plan_run1d_llama.json` | `83d893b841df04bb3cac2ccfc5bdb6ee738e64219d3ff6fd8e95bc794053c195` |
| Run 1e (solo references + warm idle, 21) | `plan_run1e_ref.json` | `813f95048c009aa5f148092905bedfec12354d984d577b0a25b9c2559e53a8ae` |

## Design decisions (D32)

| ID | Item | Decision |
|---|---|---|
| D32-1 | F safety margin | A policy is feasible for F only if, for every type, the lower one-sided 95% bound of its predicted complete rate over the R_sim simulation replications (mean - t_{0.95,R-1} * sd / sqrt(R)) is at least 0.95. R_sim = 10 replications. An infeasible measured F counts as a failure of H4, never as "regret undefined". |
| D32-2 | Run 2 size and mixes | N = 500 measured requests per run after W = 50 warm-up requests. Mix A (encoder+detector regime, every Run 1 cell has c <= 1.03): bert:5, bert:23, yolo:1, yolo:5 at 0.25 each. Mix B (decoder regime, cells with LLaMA have c > 1): llama:3 0.10, llama:15 0.10, bert:5 0.20, bert:23 0.20, yolo:1 0.20, yolo:5 0.20. Conditions: A_r030, A_r060, B_r030, B_r060 (Poisson, cv_a 1) and B_r060_b (bursty, cv_a 2), alpha 5. Expected requests per type per run >= 50. Grid trimming (fixed before any simulation, independent of simulator rankings): single-run grid points use GRID_LEVELS_MHZ = (612, 1020); F and every frequency baseline still choose from the co-run levels {612, 816, 1020}. Grid = K^2 x {FCFS, STF} x 2 levels = 36 points before aliasing. |
| D32-3 | H1 | H1 becomes a descriptive profiling result: report c(A,f) per cell with its interval over repeats; no pass/fail. Introduction and Appendix A use the measured range (no gain without the decoder, gain with it). |
| D32-4 | Primary endpoint | Primary: paired log energy ratio log(E_F / E_ref) per condition and repeat, ref in {default_pinned (= corun at 1020 MHz), serial_stf}, pooled over the 5 conditions x 3 paired repeats (15 pairs) with a t interval. Two primary tests, Holm-adjusted at alpha 0.05. Smallest effect of interest: 5% energy (|log ratio| = 0.0513). Secondary: F versus each other 3-repeat baseline (default under the governor, serial, solo_profile_freq), Holm within the secondary family. Report the minimum detectable effect from the Run 1/1b/1e repeat variance. Report energy per completed work (board energy divided by the summed reached-checkpoint depth relative to target) next to energy per request. |
| D32-5 | Winner's curse | After Run 2, for every condition where the best measured feasible grid key differs from F's key, that grid key is re-measured with 2 more repeats (Run 2b, arrival seeds of repeats 1 and 2). Regret is computed on equal repeat counts (3 versus 3). |
| D32-6 | Intervals and simulator accuracy | Attainment interval: bootstrap that resamples runs (repeats) and, inside each run, moving blocks of 50 consecutive requests (2000 resamples, fixed seed); feasible if the 2.5% quantile of every type's complete rate is >= 0.95, borderline if only the point estimate is, infeasible otherwise. Wilson stays as a secondary column. Simulator accuracy: absolute error in percentage points of the per-type complete rate, feasibility-class agreement, Spearman rank correlation of predicted versus measured energy over the measured grid keys per condition, energy APE as secondary; the governor default is excluded from simulator-accuracy metrics. Back-test: before the freeze, simulate the held-out Run 1 cells in closed loop with the fitted slowdown and power models and report predicted versus measured per-tenant median latency and board energy. |
| D32-7 | Slowdown model selection | Leave-one-cell-out cross-validation over the training cells (each cell with all its levels left out together): select the simplest model whose LOCO median absolute percentage error of s_p50 is <= 10% and below M0. Held-out cells are only reported, with per-cell errors (no bootstrap over 6 cells). AM1 survivorship sensitivity: refit with MIN_MEASURED_REQ_PER_TENANT = 5 and report the change. Session offset: compare per-type solo medians across Run 1, Run 1b and Run 1e at shared levels; report; if any offset exceeds 1%, report a sensitivity of the Run 1d slowdowns scaled by that offset. |
| D32-8 | Classes and game | Class split: sort types by solo median service time at f_max and split at the largest ratio between consecutive types (Run 1: 77 ms -> 640 ms, so short = bert:5, yolo:1, yolo:5, bert:23 and long = llama:3, llama:15). Game: enumerate all 9 profiles of K x K; a profile is a pure Nash equilibrium if no class lowers its cost by more than the fixed tolerance tau = 0.005 (absolute shortfall) by a unilateral threshold change; report all equilibria. PoA_eps = energy of the worst equilibrium divided by the lowest energy over profiles that meet the 95% constraint (constrained optimum); undefined if no profile meets it. Best-response dynamics are kept as a secondary description. |
| D32-9 | Close design todos | W = 50, R_sim = 10, grid 36 points, probe margin = D6 formula, H5: PoA_eps >= 1.05 at rho 0.6 and larger at rho 0.6 than at 0.3, screening flag Tj > 85 C (report and flag; sensitivity without flagged runs), checkpoint-overhead gate from Run 1 solo data. Warm idle power: Run 1e (idle at 612/816/1020 at the end of a warm session). LLaMA recompilation stationarity: report the latency trend of llama requests within conditions (slope test) from Run 1. Default arm named "OS governor with memory clock locked at 3199 MHz"; no stock-default arm. Third-party timestamp: the freeze commit includes exp2/frozen/run2/ANALYSIS_PLAN.md (this decision table + success criteria), pushed to GitHub with a tag and a GitHub release before Run 2. |
| D32-10 | Reporting (after Run 2) | Limitations subsection (one board, untrained exits, EMC fixed, one alpha); swarm framing: admission reads a shared count (environment signal); dataset claim only with released per-request data. |

## Answers to open design questions (D32-11)
| Question | Decision |
|---|---|
| MDE power level | MDE at 80% power, two-sided alpha 0.025 (the first Holm step for the two primary tests), from the repeat variance of energy per request in Run 1/1b/1e. |
| Feasibility rule of the constrained optimum (PoA_eps denominator) | The same one-sided 95% lower bound over the R_sim replications as F (D32-1), as implemented in exp2/model/game.py. |
| Simulator accuracy pass rule | Pre-registered, per Run 2 condition, then counted over the 5 conditions: median absolute error of per-type complete rate <= 2.0 percentage points, feasibility-class agreement >= 0.80, Spearman rank correlation of energy over the measured grid >= 0.60. The simulator is called accurate if at least 4 of 5 conditions meet all three. Energy APE is reported only. |
| Regret candidates | Measured feasible or borderline points (point estimate of every type's complete rate >= 0.95), as in the analyzer; the bootstrap status is reported next to it. |
| H4 and H5 aggregation | H4 is supported if F is measured feasible (point estimate) and its regret is <= 10% in at least 4 of the 5 conditions. H5 is evaluated per mix on the Poisson conditions only (rho 0.3 and 0.6). |
| Tie tolerance for PoA_R (no deadlines) | A relative tolerance of 1% on the mean response time. |

## Hypotheses and success criteria (summary)

| ID | Statement | Test |
|---|---|---|
| Primary | F uses less board energy per request than the default pinned at 1020 MHz and than serial-STF | Paired log energy ratio over 5 conditions x 3 repeats (15 pairs), t interval, Holm over the two primary tests at alpha 0.05, smallest effect of interest 5% |
| H1 | Co-run capacity depends on the mix (descriptive) | c(A,f) per profiling cell with intervals over repeats |
| H2 | The lowest energy per request is below f_max in some condition | Cheapest feasible or borderline measured grid point |
| H3 | The best policy changes with load and variability | No single grid point is best in every condition |
| H4 | F approaches the best measured policy without run-time telemetry | F measured feasible and regret <= 10% in at least 4 of 5 conditions (regret after Run 2b, equal repeats) |
| H5 | The price of anarchy grows with load | Per mix, Poisson conditions: PoA_eps >= 1.05 at rho 0.6 and larger at rho 0.6 than at rho 0.3 |
| Simulator | Predictions match measurements | Per condition: median abs error of per-type complete rate <= 2.0 pp, feasibility-class agreement >= 0.80, Spearman of energy over the measured grid >= 0.60; accurate if at least 4 of 5 conditions meet all three; governor default excluded |
