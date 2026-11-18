#!/usr/bin/env bash
# 循环播放最新 checkpoint，模拟"训练过程中的实时可视化"
cd "$HOME/InstinctMJ" || exit 1
RUN='g1AcpTrack40'
RUNS=(logs/instinct_rl/${RUN}*/)

while true; do
  CKPT=$(ls -t "${RUNS[@]}"*/model_*.pt 2>/dev/null | head -1)
  if [ -z "$CKPT" ]; then
    sleep 15
    continue
  fi
  RUNDIR=$(dirname "$CKPT")
  echo "[live_view] playing $CKPT"
  MUJOCO_GL=glfw .venv/bin/instinct-play Instinct-ACP-Track-G1-v1 \
    --load-run "$RUNDIR" \
    --checkpoint-file "$CKPT" \
    --num-envs 1 --viewer native --max-steps 2000 \
    > /tmp/live_view.log 2>&1
  sleep 3
done
