import pytest


@pytest.mark.skipif(__import__("importlib").util.find_spec("diffusers") is None, reason="diffusers not installed")
def test_sd3_module_imports():
    from src.sd35_task_aware_vae.sd3.pipeline_factory import build_sd3_text2img_pipeline  # noqa: F401
