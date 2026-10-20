#!/bin/bash
# usage: freeze_run.sh CONDITIONS_JSON OUT_DIR [WORKERS]   (R32 freeze build on the board, 2026-10-20)
set -u
B=/home/earlyexp/exp2_pilot_20261020; A=$B/analysis_r32; COND=$1; OUT=$2; W=${3:-1}
cd /home/earlyexp/exp2_wt
export PYTHONDONTWRITEBYTECODE=1
t0=$(date +%s); /home/earlyexp/EarlyExitMonoRepo/venv/bin/python -m exp2.model.freeze_build \
  --run1-csv $A/run1_csv --types $A/types.json --slowdown $A/slowdown.json --power $A/power.json \
  --conditions $COND --run1-plan $B/plan_run1_3fam.json --run1b-plan $B/plan_run1b_816.json \
  --seed 20261020 --workers $W --out $OUT > $OUT.log 2>&1
rc=$?; echo "exit $rc $(date +%T) elapsed_s $(( $(date +%s) - t0 ))" >> $OUT.log
