from src.sd35_task_aware_vae.sd3.prompts import build_class_prompt_entries, build_neutral_prompt_entries


def test_build_neutral_prompt_entries_count():
    entries = build_neutral_prompt_entries(
        {
            "neutral_prompts": ["prompt a", "prompt b"],
            "neutral_prompt_strategy": "cycle",
            "use_negative_prompt": False,
        },
        num_images=3,
    )
    assert len(entries) == 3
    assert [item["prompt"] for item in entries] == ["prompt a", "prompt b", "prompt a"]
    assert all(item["negative_prompt"] == "" for item in entries)



def test_build_class_prompt_entries_per_class_counts_and_target_prompt():
    class_names = ["PDR", "DME", "ERM"]
    entries = build_class_prompt_entries(
        class_names,
        {
            "class_targets": ["PDR", ["PDR", "DME"]],
            "target_prompts": {
                "PDR": "prompt pdr",
                "PDR__DME": "prompt combo",
            },
            "use_negative_prompt": False,
        },
        num_images_per_target={"PDR": 2, "PDR__DME": 1},
    )
    assert len(entries) == 3
    assert [item["target_id"] for item in entries] == ["PDR", "PDR", "PDR__DME"]
    assert [item["prompt"] for item in entries] == ["prompt pdr", "prompt pdr", "prompt combo"]
    assert [item["sample_index"] for item in entries] == [0, 1, 0]
    assert entries[0]["label_vector"] == [1.0, 0.0, 0.0]
    assert entries[2]["label_vector"] == [1.0, 1.0, 0.0]
