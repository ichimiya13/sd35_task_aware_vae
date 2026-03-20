from src.sd35_task_aware_vae.labels.schema import load_label_schema


def test_proposed_schema_loads():
    classes, label_groups, group_reduce, mask_cfg = load_label_schema("configs/labels/proposed_schema.yaml")
    assert len(classes) > 0
    assert isinstance(label_groups, dict)
    assert isinstance(mask_cfg, dict)
    assert group_reduce in {"any", "all", "or", "and", "max", "min"}
