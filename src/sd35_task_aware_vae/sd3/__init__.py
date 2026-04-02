from __future__ import annotations

__all__ = [
    "build_sd3_img2img_pipeline",
    "build_sd3_text2img_pipeline",
    "build_sd3_vae",
    "decode_from_latents",
    "encode_to_latents",
    "reverse_restore_batch",
    "train_sd35_system_from_config",
]



def __getattr__(name: str):
    if name in {"decode_from_latents", "encode_to_latents"}:
        from src.sd35_task_aware_vae.sd3.latent_codec import decode_from_latents, encode_to_latents

        return {"decode_from_latents": decode_from_latents, "encode_to_latents": encode_to_latents}[name]
    if name in {"build_sd3_img2img_pipeline", "build_sd3_text2img_pipeline"}:
        from src.sd35_task_aware_vae.sd3.pipeline_factory import build_sd3_img2img_pipeline, build_sd3_text2img_pipeline

        return {
            "build_sd3_img2img_pipeline": build_sd3_img2img_pipeline,
            "build_sd3_text2img_pipeline": build_sd3_text2img_pipeline,
        }[name]
    if name == "reverse_restore_batch":
        from src.sd35_task_aware_vae.sd3.restore import reverse_restore_batch

        return reverse_restore_batch
    if name == "build_sd3_vae":
        from src.sd35_task_aware_vae.sd3.vae_factory import build_sd3_vae

        return build_sd3_vae
    if name == "train_sd35_system_from_config":
        from src.sd35_task_aware_vae.sd3.finetune import train_sd35_system_from_config

        return train_sd35_system_from_config
    raise AttributeError(name)
