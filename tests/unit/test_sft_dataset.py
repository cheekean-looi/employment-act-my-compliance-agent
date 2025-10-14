import json
from pathlib import Path

from src.training.make_sft_dataset import ProductionSFTGenerator
from src.training.validate_dataset import SFTDatasetValidator


def _write_chunks(tmp_path: Path) -> Path:
    chunks = [
        {"section_id": "EA-60E", "text": "Employees are entitled to annual leave of at least 8 days."},
        {"section_id": "EA-2022-13", "text": "Termination requires notice in writing and certain procedures."},
        {"section_id": "EA-2022-60E(1)", "text": "Annual leave increases by service length."},
        {"section_id": "EA-14", "text": "Dismissal must be with just cause or excuse."},
    ]
    p = tmp_path / "chunks.jsonl"
    with p.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c) + "\n")
    return p


def test_sft_generation_and_split(tmp_path: Path):
    chunks_path = _write_chunks(tmp_path)
    gen = ProductionSFTGenerator(chunks_path, seed=123)
    examples = gen.generate_dataset(target_size=12)
    # Target attainment bound (allow some shortfall but require non-trivial size)
    assert len(examples) >= 8 and len(examples) <= 12
    # All examples have canonical citations and outputs mention section
    for ex in examples:
        assert ex["citations"], "missing citations"
        assert ex["citations"][0].startswith("EA-"), "non-canonical citation format"
        assert "section" in ex["output"].lower()

    # Save and validate split
    out = tmp_path / "sft_out"
    gen.save_dataset(out, examples)
    train = list((out / "sft_dataset_train.jsonl").read_text().splitlines())
    eval_ = list((out / "sft_dataset_eval.jsonl").read_text().splitlines())
    assert len(train) > 0 and len(eval_) > 0

    # Section-isolated split: no overlap between train and eval sections
    train_secs = set()
    for line in train:
        train_secs.update(json.loads(line)["citations"])
    eval_secs = set()
    for line in eval_:
        eval_secs.update(json.loads(line)["citations"])
    assert not (train_secs & eval_secs)

    # Schema validation
    validator = SFTDatasetValidator.from_chunks_file(chunks_path)
    result = validator.validate_train_eval_split(out / "sft_dataset_train.jsonl", out / "sft_dataset_eval.jsonl")
    assert result.is_valid

