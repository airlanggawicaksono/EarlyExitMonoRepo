# Addendum 1 to the Run 2 analysis plan (exploratory checks of Proposition 2)

Written on 2026-10-20 after Run 2 started on the board (20:42 local time) and before any Run 2 measurement was
opened or analysed. Only the run status (alive, number of finished runs) had been read. This addendum does not
change ANALYSIS_PLAN.md: the primary endpoint, H1-H5, the simulator accuracy rule and every confirmatory test stay as
frozen. Everything below is exploratory and will be reported as such.

## Why

A threshold swarm model of the requests (a response-threshold admission rule whose deterministic limit is the class
limit k, with an age-depth structured population of requests) yields work and energy balance identities
(Proposition 2) that can be checked on Run 2 data without new board runs. The derivation and its assumptions are
documented with the paper. Two findings from the frozen Run 1 data motivate the checks: the measured co-run energy
ratio gamma(A,f) is below 1 for all 60 set x level combinations (0.911-0.978, median of three repeats), and the frozen
power model saturates (U_A/eta >= 1.75 for every set), so the simulator credits co-running with energy savings in 38 of
60 sets that the measurements do not show.

## Exploratory checks (no new board runs)

| ID | Check | Expected | A failure means |
|---|---|---|---|
| E1 | R_c = (measured credited + wasted work) / sum_A T_A c(A,f) from Run 1, per run | about 1 | capacity does not transfer from closed-loop profiling to open-loop serving |
| E2 | R_P = measured energy / [P_idle T_0 + sum_A T_A P_Run1(A,f)], per run | about 1 | power does not transfer, or open-loop idle power differs |
| E3 | Serial runs: busy fraction equals the processed work rate (capacity 1) | equal within noise | open-loop solo service differs from the closed-loop profile |
| E4 | Corollary 2b at equal f: eps(policy) - eps(serial) - (1/lambda) sum_i dP_i (w_i(policy) - w_i(serial)) | >= 0 and at most the co-run time fraction x max e/lambda (realistically below 1% of eps) | the measured gamma does not transfer |
| E5 | Simulator energy bias: measured co-run minus serial energy at equal f exceeds the frozen predicted difference | positive sign, small (below 1% in mix A) | the frozen power model is right for open loop |
| E6 | A_r060 at 612 MHz: depth shortfall of co-run policies exceeds that of serial policies | yes | the capacity mechanism does not dominate |
| E7 | A_r060 at 612 MHz: busy fraction >= credited work rate (c* = 1 for serial) | always | assumption (P) or c* = 1 is violated |
| E8 | Power-model sensitivity: recompute the energy predictions of every measured key with P(A,f) taken directly from the Run 1 co-run cells (no saturation), and report the change of the simulator energy accuracy metrics | rank fidelity of energy improves in mix A | the saturated power model is not the cause of energy mis-ranking |
| E9 | Mix B at 612 MHz: truncation of bert requests rises with the fraction of time a triple is in service (bert in triples at 612 MHz has s x tau_p50 / D = 1.05-1.12) | positive association | the triple stretch mechanism is not active |

Note on E5 and E8: they concern the confirmatory criterion "Spearman rank correlation of energy >= 0.60". Energy
differences between grid points in mix A are only 0.1-0.8%, so the energy ranking there can flip by noise or by this
bias. These checks explain the criterion's outcome; they do not change it.
