# DiT fine-tuning with full fine-tuned SD3.5 VAE

## Recommended order

1. Start from the existing neutral DiT checkpoint trained with `sd35m_dit_full_ft_a100x8_main.yaml`.
2. Keep the custom VAE frozen and fine-tune only the transformer.
3. Compare `full_base` and `full_center_patch_only` under the same seeds / prompts.
4. After selecting the better VAE, move on to label-conditioned DiT fine-tuning if needed.

## Configs

- Base VAE adaptation:
  - `configs/sd3/train/sd35m_dit_full_ft_from_fullvae_base_a100x8_main.yaml`
- Center-patch VAE adaptation:
  - `configs/sd3/train/sd35m_dit_full_ft_from_fullvae_center_patch_only_a100x8_main.yaml`

Both configs:
- initialize the transformer from `outputs/checkpoints/sd3/sd35m_dit_full_ft_a100x8_main/best/transformer`
- load the calibrated custom VAE from `outputs/checkpoints/vae/.../best/vae_latent_calibrated`
- freeze the VAE and train only the transformer
- keep the prompt mode neutral for a clean compatibility / quality comparison

## Launch

### 8 GPU DDP

```bash
bash scripts/run_sd35m_dit_ft_with_custom_vae.sh base
bash scripts/run_sd35m_dit_ft_with_custom_vae.sh center_patch
```

### Single GPU debug

```bash
NPROC_PER_NODE=1 bash scripts/run_sd35m_dit_ft_with_custom_vae.sh center_patch
```

## Notes

- These configs are intended for *VAE comparison* first, not for final class-conditioned generation.
- If `vae_latent_calibrated` does not exist, change `vae.checkpoint` to `.../best/vae`.
- Preview generation keeps a neutral prompt and uses guidance scale 3.5 for a more stable sanity check.
