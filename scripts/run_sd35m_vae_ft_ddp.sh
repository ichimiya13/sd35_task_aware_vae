#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/run_sd35m_vae_ft_ddp.sh center_patch_weakkl
#   bash scripts/run_sd35m_vae_ft_ddp.sh center_patch_covgram
#   NPROC_PER_NODE=4 bash scripts/run_sd35m_vae_ft_ddp.sh base_covgram

TARGET="${1:-center_patch_weakkl}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
MASTER_PORT="${MASTER_PORT:-29517}"

case "$TARGET" in
  base_weakkl)
    CONFIG="configs/vae/train/sd35m_vae_ft_full_base_weakkl_ddp.yaml"
    ;;
  center_patch_weakkl|center_weakkl)
    CONFIG="configs/vae/train/sd35m_vae_ft_full_center_patch_only_weakkl_ddp.yaml"
    ;;
  base_covgram)
    CONFIG="configs/vae/train/sd35m_vae_ft_full_base_weakkl_covgram_ddp.yaml"
    ;;
  center_patch_covgram|center_covgram)
    CONFIG="configs/vae/train/sd35m_vae_ft_full_center_patch_only_weakkl_covgram_ddp.yaml"
    ;;
  *)
    echo "Unknown target: $TARGET" >&2
    echo "Valid targets: base_weakkl, center_patch_weakkl, base_covgram, center_patch_covgram" >&2
    exit 2
    ;;
esac

echo "[run] torchrun nproc=$NPROC_PER_NODE config=$CONFIG"
torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" --master_port="$MASTER_PORT" \
  -m src.scripts.train_vae_from_config --config "$CONFIG"
