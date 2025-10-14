import json
from pathlib import Path

from src.training.make_pref_pairs import FixedPreferencePairGenerator


def _write_chunks(tmp_path: Path) -> Path:
    chunks = [
        {"section_id": "EA-60E", "text": "Annual leave of 8 days at least."},
        {"section_id": "EA-2022-13", "text": "Termination requires notice in writing."},
        {"section_id": "EA-2022-60E(1)", "text": "Annual leave increases by service length."},
        {"section_id": "EA-14", "text": "Dismissal must be with just cause or excuse."},
    ]
    p = tmp_path / "chunks.jsonl"
    with p.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c) + "\n")
    return p


def test_pref_pairs_grounding_and_split(tmp_path: Path):
    chunks_path = _write_chunks(tmp_path)
    gen = FixedPreferencePairGenerator(chunks_path)

    # Family map should be non-empty with canonicalization
    assert len(gen.section_families) > 0

    pairs = gen.generate_preference_pairs(target_size=12)
    assert len(pairs) >= 10

    # At least one chosen response should be correctly grounded
    grounded = sum(1 for p in pairs if p["chosen_grounding"]["correctly_grounded"]) 
    assert grounded >= 1

    # Save and check split hygiene by section
    out = tmp_path / "pairs.jsonl"
    split = gen.save_preference_pairs(pairs, out)
    train = [json.loads(l) for l in (out.parent / f"{out.stem}_train.jsonl").read_text().splitlines()]
    eval_ = [json.loads(l) for l in (out.parent / f"{out.stem}_eval.jsonl").read_text().splitlines()]

    train_secs = set(p["source_section"] for p in train)
    eval_secs = set(p["source_section"] for p in eval_)
    assert not (train_secs & eval_secs)

