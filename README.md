# sd35_task_aware_vae

SD3.5 Medium / Large を基盤に、UWF 向け task-aware VAE と SD3.5 transformer を評価・学習するための研究用リポジトリです。

## できること

- 既存の UWF dataset / label schema / ConvNeXt-Large teacher を再利用
- official SD3.5 VAE / custom SD3.5 VAE の差し替え
- SD3.5 reverse restoration 評価
- SD3.5 text2img / img2img augmentation 生成
- baseline / task-aware loss による SD3.5 VAE fine-tuning
- SD3.5 Medium / Large の DiT 部分フル fine-tuning
- SD3.5 Medium / Large の DiT 部分 LoRA fine-tuning
- 1 本の YAML 駆動 trainer で transformer / LoRA / VAE / joint 学習を切り替え
- VAE / SD3.5 trainer の wandb ロギング

## 主なエントリポイント

```bash
python -m src.scripts.train_teacher_from_config --config configs/classifier/train/teacher_convnextl_proposed.yaml
python -m src.scripts.evaluate_teacher_from_config --config configs/classifier/eval/eval_teacher_proposed.yaml
python -m src.scripts.train_vae_from_config --config configs/vae/train/sd35m_vae_ft_base.yaml
python -m src.scripts.train_sd3_finetune_from_config --config configs/sd3/train/sd35m_dit_full_ft.yaml
python -m src.scripts.train_sd3_finetune_from_config --config configs/sd3/train/sd35m_dit_lora_ft.yaml
python -m src.scripts.train_sd3_finetune_from_config --config configs/sd3/train/sd35m_joint_dit_vae_ft.yaml
python -m src.scripts.run_sd3_restore_eval_from_config --config configs/eval/sd35m_restore_eval_official_vae.yaml
python -m src.scripts.run_sd3_generate_aug_from_config --config configs/eval/sd35m_generate_aug_official_vae.yaml
python -m src.scripts.reconstruct_from_config --config configs/vae/reconstruct/sd35m_recon_official_val.yaml
```

## 追加した SD3.5 学習系 config

- `configs/sd3/train/sd35m_dit_full_ft.yaml`
  - DiT full fine-tuning
- `configs/sd3/train/sd35m_dit_lora_ft.yaml`
  - DiT LoRA fine-tuning
- `configs/sd3/train/sd35m_joint_dit_vae_ft.yaml`
  - DiT + VAE joint fine-tuning のたたき台

`train_targets.transformer.mode` を `full / lora / frozen` で切り替え、
`train_targets.vae.enabled` で VAE 学習をオンにできます。

## GPU 指定

各 YAML に

```yaml
runtime:
  gpu_ids: [3]
```

のように書くと、その番号が `CUDA_VISIBLE_DEVICES` に反映されます。
単 GPU 実行では、指定した物理 GPU がプロセス内では `cuda:0` として見えます。

## wandb

VAE trainer と SD3.5 trainer の両方で、次のような設定をそのまま使えます。

```yaml
wandb:
  enabled: true
  project: uwf-sd35
  name: sd35m_dit_full_ft
  mode: online   # offline / disabled も可
  log_interval_steps: 10
```

## ディレクトリ

- `src/sd35_task_aware_vae/datasets`: dataset
- `src/sd35_task_aware_vae/labels`: label schema / masking
- `src/sd35_task_aware_vae/teacher_classifier`: ConvNeXt-Large teacher
- `src/sd35_task_aware_vae/vae`: VAE fine-tuning
- `src/sd35_task_aware_vae/sd3`: SD3.5 pipeline / latent codec / restore / finetuning
- `src/sd35_task_aware_vae/evaluation`: teacher-based evaluation utilities
- `configs/sd3/train`: SD3.5 transformer / LoRA / joint fine-tuning configs

## 注意

- reverse restoration と finetuning は diffusers の SD3 系実装を前提にしています。
- LoRA を使う場合は `peft`、wandb を使う場合は `wandb` が必要です。
- teacher の checkpoint path と UWF dataset path は各 config に合わせて更新してください。
- static prompt cache を使うと、neutral / explicit prompt のときは text encoder を CPU へ退避してメモリを節約できます。
