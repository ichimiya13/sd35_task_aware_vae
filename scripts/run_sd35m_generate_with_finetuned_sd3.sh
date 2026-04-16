#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

TARGET="${1:-center_patch}"
GPU_ID="${GPU_ID:-0}"
CONFIG=""

case "$TARGET" in
  base)
    CONFIG="configs/eval/sd35m_generate_neutral_count_trained_fullvae_base_dit_ft_a100x8_main.yaml"
    ;;
  center_patch|center)
    CONFIG="configs/eval/sd35m_generate_neutral_count_trained_fullvae_center_patch_only_dit_ft_a100x8_main.yaml"
    ;;
  *)
    echo "Usage: $0 [base|center_patch]" >&2
    exit 1
    ;;
esac

CUDA_VISIBLE_DEVICES="$GPU_ID" python -m src.scripts.run_sd3_generate_aug_from_config --config "$CONFIG"
