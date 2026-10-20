#!/bin/bash
# Run 1c (2026-10-20): cold then warm server start per family, one server per session, own compile caches.
# The shared cache used by Run 1, Run 1b and Run 2 is not touched.
set -u
B=/home/earlyexp/exp2_pilot_20261020; C=$B/run1c_cache
for fam in bert llama yolo; do
  rm -rf "$C/$fam"; mkdir -p "$C/$fam/inductor" "$C/$fam/triton" "$C/$fam/cuda"
  for state in cold warm; do
    if [ "$state" = cold ]; then sync; sudo -n sysctl -q -w vm.drop_caches=3; fi
    TORCHINDUCTOR_CACHE_DIR=$C/$fam/inductor TRITON_CACHE_DIR=$C/$fam/triton CUDA_CACHE_PATH=$C/$fam/cuda \
      bash /home/earlyexp/exp2_wt/exp2/board/scripts/launch.sh $B/plan_run1c_$fam.json $B/session_run1c_${fam}_$state run1c_${fam}_$state 250
    pid=$(cat $B/run1c_${fam}_$state.pid)
    while kill -0 "$pid" 2>/dev/null; do sleep 5; done
    echo "$(date +%T) $fam $state done: inductor files $(find $C/$fam/inductor -type f | wc -l), triton files $(find $C/$fam/triton -type f | wc -l)" >> $B/run1c.log
    sleep 60
  done
done
echo "$(date +%T) run1c all done" >> $B/run1c.log
