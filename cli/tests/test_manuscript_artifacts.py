"""Frozen publication data stays connected to proposals and released assets."""

import importlib.util
import json
from pathlib import Path

import pytest


FIGURES = Path(__file__).parents[2] / "figures"


@pytest.fixture
def artifacts(monkeypatch):
    monkeypatch.syspath_prepend(str(FIGURES))
    spec = importlib.util.spec_from_file_location("manuscript_artifact_audit", FIGURES / "audit.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_publication_curves_and_policies_replay_from_all_scored_proposals(artifacts):
    report = artifacts.audit()
    assert report["molecular_proposals"] == 500
    assert report["reaction_proposals"] == 250
    assert report["released_figure_assets"] == 8
    assert report["all_asset_and_data_hashes_match"]
    assert report["maximum_display_rounding_difference"] < 5e-9


def test_frozen_input_resolver_rejects_modified_data(artifacts, monkeypatch, tmp_path):
    import _paths

    frozen = json.loads(_paths.MANIFEST.read_text())
    record = frozen["data_files"]["data/elote115_search_and_energy.json"]
    altered = tmp_path / record["path"]
    altered.parent.mkdir(parents=True)
    altered.write_text('{"altered": true}')
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(frozen))
    monkeypatch.setattr(_paths, "ROOT", tmp_path)
    monkeypatch.setattr(_paths, "MANIFEST", manifest)
    with pytest.raises(ValueError, match="Frozen manuscript data changed"):
        _paths.data_path("data/elote115_search_and_energy.json")


def test_historical_dependency_keeps_its_actual_identity(artifacts):
    import _paths

    manifest = json.loads(_paths.MANIFEST.read_text())
    for sha256, record in manifest["historical_source_files"].items():
        old = _paths.data_path(record["logical_path"], sha256)
        current = _paths.data_path(record["logical_path"])
        assert old != current
        # Only the reporting cohort changed; Figure 2's per-case results did not.
        assert (
            json.loads(old.read_text())["molecules"] == json.loads(current.read_text())["molecules"]
        )
        with pytest.raises(ValueError, match="Historical source identity"):
            _paths.data_path("data/reaction_elote_delta_a100_summary.json", sha256)
