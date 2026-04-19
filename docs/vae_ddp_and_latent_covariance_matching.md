# VAE DDP and latent covariance / Gram matching

## DDP execution

```bash
bash scripts/run_sd35m_vae_ft_ddp.sh center_patch_weakkl
bash scripts/run_sd35m_vae_ft_ddp.sh center_patch_covgram
```

Available targets:

- `base_weakkl`
- `center_patch_weakkl`
- `base_covgram`
- `center_patch_covgram`

By default the script launches 8 processes. Override with:

```bash
NPROC_PER_NODE=4 MASTER_PORT=29521 bash scripts/run_sd35m_vae_ft_ddp.sh center_patch_covgram
```

The DDP configs use local `batch_size: 1` and `grad_accum_steps: 1`. With 8 GPUs this gives effective batch size 8, matching the previous single-GPU `batch_size: 2, grad_accum_steps: 4` update size while reducing wall-clock time.

## Latent distribution matching loss

Enable it under `loss.latent_distribution`:

```yaml
loss:
  kl:
    weight: 2.0e-07
  latent_distribution:
    weight: 0.05
    type: mean_covariance  # mean_covariance | covariance | mean_gram | gram | mean_correlation | correlation
    max_tokens: 4096
    include_mean: true
    mean_weight: 1.0
    matrix_weight: 1.0
    posterior: mode
    reference:
      checkpoint: null
      subfolder: vae
      model_repo_id: stabilityai/stable-diffusion-3.5-medium
      eval_mode: true
```

The loss compares the current scaled SD3 latents against the official SD3.5 VAE scaled latents on the same batch. It constrains the aggregated channel statistics seen by the DiT without forcing every sample latent to exactly match the official encoder.

Suggested order:

1. `center_patch_weakkl`
2. `center_patch_covgram`
3. only then tune `latent_distribution.weight` if the logged weighted term is too small or too large relative to `recon_term`.
