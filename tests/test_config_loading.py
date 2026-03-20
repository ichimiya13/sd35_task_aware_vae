from src.sd35_task_aware_vae.utils.config import load_yaml


def test_load_yaml_mapping(tmp_path):
    path = tmp_path / "cfg.yaml"
    path.write_text("a: 1\n", encoding="utf-8")
    cfg = load_yaml(path)
    assert cfg["a"] == 1
