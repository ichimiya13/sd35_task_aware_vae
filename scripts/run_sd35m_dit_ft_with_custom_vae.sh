#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: bash scripts/run_sd35m_dit_ft_with_custom_vae.sh [base|center_patch]"
  echo "Optional env: NPROC_PER_NODE=8 MASTER_PORT=29501"
  exit 1
fi

variant="$1"
case "$variant" in
  base)
    CONFIG="configs/sd3/train/sd35m_dit_full_ft_from_fullvae_base_a100x8_main.yaml"
    ;;
  center|center_patch|center_patch_only)
    CONFIG="configs/sd3/train/sd35m_dit_full_ft_from_fullvae_center_patch_only_a100x8_main.yaml"
    ;;
  *)
    echo "Unknown variant: $variant"
    echo "Expected one of: base, center_patch"
    exit 1
    ;;
esac

NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
MASTER_PORT="${MASTER_PORT:-29501}"

if [[ "$NPROC_PER_NODE" -le 1 ]]; then
  python -m src.scripts.train_sd3_finetune_from_config --config "$CONFIG"
else
  torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" --master_port="$MASTER_PORT" \
    -m src.scripts.train_sd3_finetune_from_config --config "$CONFIG"
fi
