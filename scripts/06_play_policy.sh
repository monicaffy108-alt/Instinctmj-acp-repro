#!/usr/bin/env bash
# 播放训练出的策略（MuJoCo 窗口，相机跟随机器人）
#
# 用法:
#   ./06_play_policy.sh <run目录名> [checkpoint文件名]
#   例: ./06_play_policy.sh 2026-11-15_21-41-47_g1AcpTrackV1_supermimic model_5000.pt
set -euo pipefail

RUN="${1:?请给出 run 目录名（logs/instinct_rl/<实验名>/ 下的子目录）}"
CKPT="${2:-model_5000.pt}"
RUN_DIR="$PWD/logs/instinct_rl/$RUN"

# 允许只给 run 子目录名，自动补全实验名前缀
if [[ ! -d "$RUN_DIR" ]]; then
  RUN_DIR=$(ls -d "$PWD"/logs/instinct_rl/*/"$RUN" 2>/dev/null | head -1)
fi
[[ -d "$RUN_DIR" ]] || { echo "找不到 run: $RUN"; exit 1; }
[[ -f "$RUN_DIR/$CKPT" ]] || { echo "找不到 checkpoint: $RUN_DIR/$CKPT"; exit 1; }

echo ">>> 播放 $RUN_DIR/$CKPT"
# 注意：--load-run 用相对仓库根目录的路径，--checkpoint-file 用绝对路径
.venv/bin/instinct-play Instinct-ACP-Track-G1-v1 \
  --load-run "${RUN_DIR#$PWD/}" \
  --checkpoint-file "$RUN_DIR/$CKPT" \
  --num-envs 1 \
  --viewer native
