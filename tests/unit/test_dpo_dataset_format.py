from typing import List, Dict

from src.training.train_dpo import FixedEmploymentActDPOTrainer


def test_preprocess_preference_data_columns():
    # Create a trainer object without running __init__
    obj = FixedEmploymentActDPOTrainer.__new__(FixedEmploymentActDPOTrainer)  # type: ignore

    # Monkeypatch minimal attributes
    obj.model_name = "meta-llama/Llama-3.1-8B-Instruct"
    obj._format_prompt = lambda p: p  # no-op for test

    data: List[Dict] = [
        {"prompt": "What is annual leave entitlement?", "chosen": "... EA-2022-60E", "rejected": "..."}
    ]

    ds = obj.preprocess_preference_data(data)
    cols = set(ds.column_names)
    assert {"prompt", "chosen", "rejected"}.issubset(cols)

