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


## 2026-03 update: SD3.5 transformer fine-tuning

- `src/scripts/train_sd3_finetune_from_config.py` を追加
- `src/sd35_task_aware_vae/sd3/finetune.py` で以下を実装
  - DiT full fine-tuning
  - DiT LoRA fine-tuning
  - optional VAE joint training
  - YAML 駆動の train target 切り替え
  - wandb logging
  - `runtime.gpu_ids` による GPU 選択
- 追加 config:
  - `configs/sd3/train/sd35m_dit_full_ft.yaml`
  - `configs/sd3/train/sd35m_dit_lora_ft.yaml`
  - `configs/sd3/train/sd35m_joint_dit_vae_ft.yaml`
- `train_vae_from_config.py` と `src/sd35_task_aware_vae/vae/trainer.py` にも wandb logging を追加

## 2026-03 update: VAE loss expansion / prompt conditioning / latent-stat calibration

- `vae/losses.py`
  - edge / gradient loss (`sobel_l1`, `laplacian_l1` など)
  - weighted reconstruction loss
  - optional LPIPS loss
  - `charbonnier` reconstruction
- `vae/trainer.py`
  - nested YAML loss config (`recon`, `kl`, `edge`, `weighted_recon`, `feature`, `logit`, `lpips`, `noise_feature`)
  - `vae.posterior.train` / `vae.posterior.eval` をサポート
  - latent mean/std 推定と `scaling_factor`, `shift_factor` 再設定の保存導線を追加
- `sd3/prompts.py`
  - `template_file` による prompt template 読み込み
  - class-wise alias / class prompt を YAML で編集可能
  - multi-label prompt strategy（top-k / random-one など）
  - class-wise text2img 用 `build_class_prompt_entries()` を追加
- `run_sd3_generate_aug_from_config.py`
  - dataset-driven generation に加えて `prompt.mode=class_text2img` をサポート
  - class-conditioned text2img でラベル YAML も保存可能
- `evaluate_teacher_from_config.py`
  - per-class CSV に `ap` / `auroc` を追加
  - metrics JSON に macro AP / AUROC を追加
