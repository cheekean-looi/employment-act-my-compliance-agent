import re
from pathlib import Path

from src.training.citation_utils import CanonicalCitationValidator


def test_normalize_legacy_to_canonical():
    v = CanonicalCitationValidator()
    assert v.normalize_section_id("EA-60E") == "EA-2022-60E"
    assert v.normalize_section_id("EA-60E(1)") == "EA-2022-60E(1)"
    assert v.normalize_section_id("EA-2022-60E(1)") == "EA-2022-60E(1)"


def test_extract_citations_regex():
    v = CanonicalCitationValidator()
    text = "Under EA-2022-60E(1) and EA-2022-13, employees have rights."
    found = v.extract_section_ids(text)
    assert "EA-2022-60E(1)" in found and "EA-2022-13" in found


def test_citation_metrics_em_iou():
    v = CanonicalCitationValidator(valid_sections={"EA-2022-60E", "EA-2022-13"})
    predicted = {"EA-2022-60E", "EA-2022-13"}
    gold = {"EA-60E"}  # legacy form
    em, iou = v.compute_citation_metrics(predicted, gold)
    assert em == 1.0  # exact match present after normalization
    assert 0.0 <= iou <= 1.0

