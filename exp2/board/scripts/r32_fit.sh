#!/bin/bash
# R32 board analysis (2026-10-20): merge Run 1, 1b, 1d, 1e; fit types, slowdown, power; back-test; Run 2 conditions.
set -e
B=/home/earlyexp/exp2_pilot_20261020; A=$B/analysis_r32; PY=/home/earlyexp/EarlyExitMonoRepo/venv/bin/python
export PYTHONDONTWRITEBYTECODE=1
cd /home/earlyexp/exp2_wt
mkdir -p $A
S="$B/session_run1 $B/session_run1b $B/session_run1d $B/session_run1e"
$PY -m exp2.analyze.closed_loop --sessions $S --out $A/run1_csv 2> $A/closed_loop.log
$PY -m exp2.analyze.closed_loop --sessions $S --out $A/run1_csv_min5 --min-requests 5 2> $A/closed_loop_min5.log
$PY -m exp2.model.build_types --run1-csv $A/run1_csv --alpha 5 --out $A/types.json > $A/build_types.log 2>&1
$PY -m exp2.model.slowdown --run1-csv $A/run1_csv --types $A/types.json --out $A/slowdown.json > $A/slowdown.log 2>&1
$PY -m exp2.model.power --run1-csv $A/run1_csv --types $A/types.json --slowdown $A/slowdown.json --out $A/power.json > $A/power.log 2>&1
$PY -m exp2.model.backtest --run1-csv $A/run1_csv --types $A/types.json --slowdown $A/slowdown.json --power $A/power.json --out $A > $A/backtest.log 2>&1
PYTHONPATH=/home/earlyexp/exp2_wt $PY exp2/board/scripts/make_run2_conditions.py $A/types.json $A/run2_conditions.json > $A/conditions.log 2>&1
echo "$(date +%T) r32 fit done" >> $A/done.log
