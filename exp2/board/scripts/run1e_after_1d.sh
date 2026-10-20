#!/bin/bash
# Start Run 1e (solo references at 612/816/1020 + warm idle power) after Run 1d has finished (2026-10-20).
B=/home/earlyexp/exp2_pilot_20261020
until [ -f $B/session_run1d/session.json ] && ! kill -0 "$(cat $B/run1d.pid)" 2>/dev/null; do sleep 20; done
sleep 60
bash /home/earlyexp/exp2_wt/exp2/board/scripts/launch.sh $B/plan_run1e_ref.json $B/session_run1e run1e 250
echo "$(date +%T) run1e launched" >> $B/run1c.log
