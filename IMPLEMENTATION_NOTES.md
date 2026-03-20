# SD3.5 実装メモ

## 追加した主なファイル

- `src/sd35_task_aware_vae/sd3/*`
  - SD3.5 pipeline factory
  - VAE factory
  - latent codec
  - reverse restoration
  - sampling / prompts / runtime
- `src/sd35_task_aware_vae/vae/*`
  - SD3.5 VAE fine-tuning trainer
  - reconstruction / KL / teacher feature loss
- `src/sd35_task_aware_vae/evaluation/*`
  - teacher consistency 指標
  - restore evaluation 集計
  - semantic filter
- `src/scripts/run_sd3_restore_eval_from_config.py`
- `src/scripts/run_sd3_generate_aug_from_config.py`
- `src/scripts/export_recon_dataset_from_config.py`
- `src/scripts/train_vae_from_config.py`
  - backend dispatch 化
- `src/scripts/reconstruct_from_config.py`
  - backend dispatch 化

## 互換性

- teacher / dataset / labels は既存 repo の流れを維持
- 旧 SDXL 用 script は `*_legacy_sdxl_*` として残し、dispatch で呼び分け

## Custom VAE checkpoint の想定

`sd3/vae_factory.py` は次を読めます。

1. official `repo_id/subfolder=vae`
2. `save_pretrained()` で保存した directory
3. `.pt/.pth/.bin/.safetensors` の state dict

## まず試すコマンド

```bash
python -m src.scripts.train_vae_from_config \
  --config configs/vae/train/sd35m_vae_ft_base.yaml

python -m src.scripts.run_sd3_restore_eval_from_config \
  --config configs/eval/sd35m_restore_eval_official_vae.yaml

python -m src.scripts.run_sd3_generate_aug_from_config \
  --config configs/eval/sd35m_generate_aug_official_vae.yaml
```

## 既知の注意

- この環境では `diffusers` が未導入だったため、SD3.5 の end-to-end 実行そのものまでは検証していません。
- ただし、全 Python ファイルの compile check と軽い pytest は通しています。
