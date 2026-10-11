import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def metric_module():
    spec = importlib.util.spec_from_file_location("public_metrics", ROOT / "scripts/metric_helpers.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_anls_preserves_paper_half_similarity_rule():
    scorer = metric_module()
    assert scorer.anls("ab", ["ac"]) == .5
    assert scorer.anls("ab", ["cd"]) == 0


def test_local_vqa_not_mistaken_for_official_leave_one_out():
    scorer = metric_module()
    assert scorer._score_prediction("textvqa", "cat", ["cat"] * 3 + ["dog"] * 7)["vqa_soft_score"] == 1


def test_released_runs_are_only_current_main_evidence():
    manifest = json.loads((ROOT / "results/run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["run_counts"] == {"E1": 32, "E2": 2, "E3": 10, "E4": 24}
    assert len({r["run_id"] for r in manifest["runs"]}) == 68


def test_dataset_revisions_and_license_not_invented():
    registry = json.loads((ROOT / "data/dataset_registry.json").read_text(encoding="utf-8"))
    assert all(d["revision"] == "unknown_not_recorded" for d in registry["datasets"])
    assert all(not d["raw_redistributed"] for d in registry["datasets"])
