# Generate images with fine-tuned SD3.5 + fine-tuned VAE

This guide uses the **full fine-tuned VAE** (`best/vae_latent_calibrated`) together with the
corresponding **DiT fine-tuned checkpoint**.

## Configs

- Base full-FT VAE + DiT FT:
  - `configs/eval/sd35m_generate_neutral_count_trained_fullvae_base_dit_ft_a100x8_main.yaml`
- Center-patch full-FT VAE + DiT FT:
  - `configs/eval/sd35m_generate_neutral_count_trained_fullvae_center_patch_only_dit_ft_a100x8_main.yaml`

Both configs use:

- `generation.mode: text2img`
- `prompt.mode: neutral_count`
- `num_images: 64`
- `guidance_scale: 3.5`
- `num_inference_steps: 40`
- `use_negative_prompt: false`

## Run directly

```bash
python -m src.scripts.run_sd3_generate_aug_from_config \
  --config configs/eval/sd35m_generate_neutral_count_trained_fullvae_base_dit_ft_a100x8_main.yaml
```

```bash
python -m src.scripts.run_sd3_generate_aug_from_config \
  --config configs/eval/sd35m_generate_neutral_count_trained_fullvae_center_patch_only_dit_ft_a100x8_main.yaml
```

## Convenience script

```bash
bash scripts/run_sd35m_generate_with_finetuned_sd3.sh base
```

```bash
bash scripts/run_sd35m_generate_with_finetuned_sd3.sh center_patch
```

You can choose the GPU with:

```bash
GPU_ID=1 bash scripts/run_sd35m_generate_with_finetuned_sd3.sh center_patch
```

## Outputs

Generated images are written under:

- `outputs/aug/sd3/sd35m_generate_neutral_count_trained_fullvae_base_dit_ft_a100x8_main/`
- `outputs/aug/sd3/sd35m_generate_neutral_count_trained_fullvae_center_patch_only_dit_ft_a100x8_main/`

Each run also writes:

- `config_used.yaml`
- `metadata.csv`
- `summary.json`
- `generated_split.yaml`

## Editing generation settings

The main fields to change are:

- `generation.num_images`
- `generation.batch_size`
- `model.num_inference_steps`
- `model.guidance_scale`
- `prompt.neutral_prompt`
