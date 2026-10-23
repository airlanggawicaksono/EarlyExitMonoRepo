# Board scripts (Jetson Orin Nano, Exp2)

Run on the board as user early-experiment from the worktree ~/exp2_wt (branch exp2-dev). Deploy by `git push` on the
laptop and `git -C ~/exp2_wt pull --ff-only https://github.com/airlanggawicaksono/EarlyExitMonoRepo.git exp2-dev` on
the board; never pull while a session runs. Data goes to ~/exp2_pilot_20261020/.

| Script | Use |
|---|---|
| launch.sh PLAN OUT NAME [LIMIT_MB] [TYPES] [ARGS..] | Start run_session as a daemon with a memory watchdog (kills exp2.serve below LIMIT_MB MemAvailable). Run 2 passes the frozen types file. ARGS go to run_session; Run 3: `launch.sh PLAN OUT NAME 250 exp2/frozen/run2/types.json --slowdown exp2/frozen/run2/slowdown.json` (OUT must be new, launch.sh deletes it). |
| r32_fit.sh | Merge the profiling sessions (Run 1, 1b, 1d, 1e), fit types, slowdown, power, back-test, write the Run 2 conditions. |
| make_run2_conditions.py TYPES OUT | Run 2 conditions (mixes A and B, rho 0.3 and 0.6, bursty B), rates from rate_for_load. |
| freeze_run.sh CONDITIONS OUT [WORKERS] | Build freeze.json with exp2.model.freeze_build and log the elapsed time. |
| run1c.sh | Run 1c: cold then warm server start per family with its own compile caches. |
| run1d_after_1c.sh, run1e_after_1d.sh | Chain Run 1d and Run 1e after the previous stage (kept for the record of how the 2026-10-20 sessions were started). |

The plans these sessions used are in exp2/frozen/run2/plans/.
