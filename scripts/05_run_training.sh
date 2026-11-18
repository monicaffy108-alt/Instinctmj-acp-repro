#!/usr/bin/env bash
# Stage-1 接触条件化 tracker 训练（单卡适配版）
#
# 用法:
#   ./05_run_training.sh                       # 默认 512 envs / 5000 iters
#   NUM_ENVS=768 ./05_run_training.sh          # 调环境数（显存够才敢加）
#   RESUME_RUN=<run目录名> ./05_run_training.sh  # 续训
#
# 必须在 InstinctMJ 仓库根目录下执行
set -euo pipefail

NUM_ENVS="${NUM_ENVS:-512}"
MAX_ITERS="${MAX_ITERS:-5000}"
NCONMAX="${NCONMAX:-64}"          # 关键：默认值过大，吃显存又拖速度
SAVE_INTERVAL="${SAVE_INTERVAL:-100}"
EXP_NAME="${EXP_NAME:-g1AcpTrack}"
RESUME_RUN="${RESUME_RUN:-}"      # 例如 2026-11-15_21-41-47_g1AcpTrackV1_supermimic
RESUME_CKPT="${RESUME_CKPT:-}"

ARGS=(
  Instinct-ACP-Track-G1-v1
  --num-envs "$NUM_ENVS"
  --env.sim.nconmax "$NCONMAX"
  --agent.max-iterations "$MAX_ITERS"
  --agent.save-interval "$SAVE_INTERVAL"
  --agent.experiment-name "$EXP_NAME"
)

if [[ -n "$RESUME_RUN" ]]; then
  ARGS+=(--agent.load-run "$RESUME_RUN")
  ARGS+=(--agent.load-checkpoint "${RESUME_CKPT:-model_5000.pt}")
fi

echo ">>> 启动: envs=$NUM_ENVS iters=$MAX_ITERS nconmax=$NCONMAX exp=$EXP_NAME"
.venv/bin/instinct-train "${ARGS[@]}"
