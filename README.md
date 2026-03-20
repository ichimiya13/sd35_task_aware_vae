# sd35_task_aware_vae

SD3.5 Medium / Large を基盤に、UWF 向け task-aware VAE を評価・学習するための研究用リポジトリです。

## できること

- 既存の UWF dataset / label schema / ConvNeXt-Large teacher を再利用
- official SD3.5 VAE / custom SD3.5 VAE の差し替え
- SD3.5 reverse restoration 評価
- SD3.5 text2img / img2img augmentation 生成
- baseline / task-aware loss による SD3.5 VAE fine-tuning

## 主なエントリポイント

```bash
python -m src.scripts.train_teacher_from_config --config configs/classifier/train/teacher_convnextl_proposed.yaml
python -m src.scripts.evaluate_teacher_from_config --config configs/classifier/eval/eval_teacher_proposed.yaml
python -m src.scripts.train_vae_from_config --config configs/vae/train/sd35m_vae_ft_base.yaml
python -m src.scripts.run_sd3_restore_eval_from_config --config configs/eval/sd35m_restore_eval_official_vae.yaml
python -m src.scripts.run_sd3_generate_aug_from_config --config configs/eval/sd35m_generate_aug_official_vae.yaml
python -m src.scripts.reconstruct_from_config --config configs/vae/reconstruct/recon_pretrained_val_4ch.yaml
```

## ディレクトリ

- `src/sd35_task_aware_vae/datasets`: dataset
- `src/sd35_task_aware_vae/labels`: label schema / masking
- `src/sd35_task_aware_vae/teacher_classifier`: ConvNeXt-Large teacher
- `src/sd35_task_aware_vae/vae`: VAE fine-tuning
- `src/sd35_task_aware_vae/sd3`: SD3.5 pipeline / latent codec / restore
- `src/sd35_task_aware_vae/evaluation`: teacher-based evaluation utilities

## 注意

- reverse restoration は diffusers の `StableDiffusion3Pipeline` / `StableDiffusion3Img2ImgPipeline` を前提にしています。
- 4bit transformer 量子化を使う場合は `bitsandbytes` が必要です。
- teacher の checkpoint path と UWF dataset path は各 config に合わせて更新してください。
