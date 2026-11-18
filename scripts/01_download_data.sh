#!/usr/bin/env bash
# 下载 Unitree LAFAN1_Retargeting_Dataset (G1 部分, 30fps CSV)
# 许可证/来源见 https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset
set -euo pipefail

DEST="${1:-$HOME/Datasets/LAFAN1_Retargeting}"

echo ">>> 目标目录: $DEST"
mkdir -p "$DEST"

if command -v hf >/dev/null 2>&1; then
  HF_CLI=hf
elif command -v huggingface-cli >/dev/null 2>&1; then
  HF_CLI=huggingface-cli
else
  echo "未找到 hf / huggingface-cli，正在安装..."
  pip install -U "huggingface_hub[cli]"
  HF_CLI=hf
fi

$HF_CLI download lvhaidong/LAFAN1_Retargeting_Dataset \
  --repo-type dataset \
  --include "g1/*.csv" \
  --local-dir "$DEST"

echo ">>> 完成，CSV 数量: $(ls "$DEST"/g1/*.csv 2>/dev/null | wc -l)"
