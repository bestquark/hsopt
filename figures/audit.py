"""Audit the published plot inputs against every frozen scientific proposal."""

import gzip
import hashlib
import json
from pathlib import Path
from dataclasses import asdict

import numpy as np

from _paths import MANIFEST, ROOT, data_path


def read_campaign(manifest_path: Path):
    manifest = json.loads(manifest_path.read_text())
    path = manifest_path.parent / "trajectories.jsonl.gz"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest["files"][path.name]
    with gzip.open(path, "rt") as stream:
        header = json.loads(next(stream))
        rows = [json.loads(line) for line in stream]
    assert len(rows) == manifest["round_records"]
    assert (
        hashlib.sha256(header["suite_source_utf8"].encode()).hexdigest() == header["suite_sha256"]
    )
    for row in rows:
        assert (
            hashlib.sha256(row["candidate_source_utf8"].encode()).hexdigest()
            == row["candidate_sha256"]
        )
        assert len(row["result"]["case_metrics"]) == header["case_count"]
        if row["decision"]["accepted"]:
            assert row["result"]["passes_chemical_accuracy"]
            assert all(m["passes_chemical_accuracy"] for m in row["result"]["case_metrics"])
    grouped = {}
    for seed in header["seeds"]:
        selected = sorted(
            (r for r in rows if r["seed"] == seed["seed"]), key=lambda r: r["iteration"]
        )
        assert [r["iteration"] for r in selected] == list(range(header["budget"] + 1))
        grouped[seed["seed"]] = selected
    return manifest, header, grouped


def audit():
    manifest = json.loads(MANIFEST.read_text())
    for name in manifest["data_files"]:
        data_path(name)
    for row in manifest["figures"].values():
        assert hashlib.sha256((ROOT / row["path"]).read_bytes()).hexdigest() == row["sha256"]
    from plot_baseline_initialization_vs_qubits import load_records, DEFAULT_DATA
    baseline_parameters = load_records(DEFAULT_DATA)
    baseline_export = json.loads(data_path("data/elote115_baseline_initialization.json").read_text())
    assert baseline_export["records"] == [asdict(row) for row in baseline_parameters]
    assert baseline_export["source_sha256"] == hashlib.sha256(DEFAULT_DATA.read_bytes()).hexdigest()
    source = json.loads(data_path("data/elote115_search_and_energy.json").read_text())
    t = source["trajectory"]
    _, mh, molecular = read_campaign(ROOT / manifest["scientific_trajectories"]["molecular"])
    assert set(molecular) == set(t["seed_ids"])
    case_ids = t["case_ids"]
    arrays = []
    for seed in t["seed_ids"]:
        rows = molecular[seed]
        baseline = {m["case_id"]: m["t95"] for m in rows[0]["result"]["case_metrics"]}
        history = []
        incumbent = None
        for row in rows:
            if row["decision"]["accepted"]:
                incumbent = {m["case_id"]: m for m in row["result"]["case_metrics"]}
            assert set(incumbent) == set(case_ids)
            history.append([incumbent[c]["t95"] / baseline[c] for c in case_ids])
        arrays.append(np.array(history).T)
    arrays = np.array(arrays)
    seed_means = arrays.mean(axis=1)
    molecule_means = arrays.mean(axis=0)
    # The published view came from rounded dashboard percentages. These are
    # display-rounding checks (six decimals in percent, i.e. 5e-9 in ratio),
    # not scientific feasibility tolerances.
    np.testing.assert_allclose(seed_means, t["seed_mean_relative_t95"], rtol=0, atol=5e-9)
    np.testing.assert_allclose(molecule_means, t["molecule_mean_relative_t95"], rtol=0, atol=5e-9)
    np.testing.assert_allclose(seed_means.min(axis=0), t["min_relative_t95"], rtol=0, atol=5e-9)
    np.testing.assert_allclose(seed_means.max(axis=0), t["max_relative_t95"], rtol=0, atol=5e-9)
    accepted = [r for rows in molecular.values() for r in rows if r["decision"]["accepted"]]
    best = min(accepted, key=lambda r: r["result"]["score"])
    assert best["candidate_sha256"] == source["provenance"]["policy_sha256"]

    _, rh, reactions = read_campaign(ROOT / manifest["scientific_trajectories"]["reactions"])
    summary = json.loads(data_path("data/reaction_elote_delta_a100_summary.json").read_text())
    delta_rows = [r for rows in reactions.values() for r in rows if r["decision"]["accepted"]]
    delta = min(delta_rows, key=lambda r: r["result"]["score"])
    assert delta["candidate_sha256"] == summary["ranked_policies"][0]["candidate_sha256"]
    for rows in reactions.values():
        assert rows[0]["candidate_sha256"] == summary["transferred_policy_sha256"]
        assert {m["case_id"]: m["t95"] for m in rows[0]["result"]["case_metrics"]} == summary[
            "elote_t95"
        ]
    assert {m["case_id"]: m["t95"] for m in delta["result"]["case_metrics"]} == summary[
        "ranked_policies"
    ][0]["t95"]
    energy = json.loads(data_path("data/reaction_elote_delta_energy_summary.json").read_text())
    for ranked in summary["ranked_policies"]:
        proposal = next(
            r for r in delta_rows if r["candidate_sha256"] == ranked["candidate_sha256"]
        )
        assert {m["case_id"]: m["t95"] for m in proposal["result"]["case_metrics"]} == ranked["t95"]
    transferred = next(iter(reactions.values()))[0]
    assert set(delta["measurements"]) == {r["key"] for r in energy["rows"]}
    for row in energy["rows"]:
        for label, proposal in (("elote", transferred), ("delta_elote", delta)):
            receipt = proposal["measurements"][row["key"]]
            assert receipt["absolute_difference_error_mha"] == row[label]["signed_error_mha"]
            assert receipt["sum_absolute_species_error_mha"] == row[label]["sumabs_error_mha"]
            assert receipt["passes_chemical_accuracy"] == row[label]["passes_accuracy"]
            assert receipt["backend"] == energy["backend"]
            assert receipt["evaluator_sha256"] == energy["evaluator_sha256"]
    for path, expected in manifest["processing_source_files"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == expected, path
    return {
        "molecular_proposals": mh["seed_count"] * mh["budget"],
        "reaction_proposals": rh["seed_count"] * rh["budget"],
        "molecular_selected_policy_sha256": best["candidate_sha256"],
        "reaction_selected_policy_sha256": delta["candidate_sha256"],
        "published_mean_relative_t95_at_100": t["mean_relative_t95"][-1],
        "raw_replayed_mean_relative_t95_at_100": float(seed_means[:, -1].mean()),
        "maximum_display_rounding_difference": float(
            max(
                np.max(np.abs(seed_means - t["seed_mean_relative_t95"])),
                np.max(np.abs(molecule_means - t["molecule_mean_relative_t95"])),
            )
        ),
        "figure6_verified_starting_hamiltonians": len(baseline_parameters),
        "released_figure_assets": len(manifest["figures"]),
        "all_asset_and_data_hashes_match": True,
    }


if __name__ == "__main__":
    print(json.dumps(audit(), indent=2))
