from __future__ import annotations

import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
INPUT_ROOT = REPO_ROOT / "experiments" / "diff_energy" / "inputs"
REACTION_MANIFEST = INPUT_ROOT / "reactions.json"
PROBLEM_METADATA = INPUT_ROOT / "problem_metadata.json"
CACHE_CHECKSUMS = INPUT_ROOT / "problem_cache.sha256"

REACTION_KEYS = [
    "hcn_hnc",
    "hco_hoc_cation",
    "hno_noh",
    "h2co_hcoh",
    "acetylene_vinylidene",
    "h2_cl2_hcl",
    "n2_h2_nh3",
    "co_h2o_co2_h2",
    "h2_f2_hf",
    "co_h2_ch4_h2o",
]
SPECIES = {
    "acetylene",
    "ch4",
    "cl2",
    "co",
    "co2",
    "f2",
    "h2",
    "h2co",
    "h2o",
    "hcl",
    "hcn",
    "hco_plus",
    "hcoh",
    "hf",
    "hnc",
    "hno",
    "hoc_plus",
    "n2",
    "nh3",
    "noh",
    "vinylidene",
}
CHEMICAL_ACCURACY_HARTREE = 1.0 / 627.5094740631


def test_reaction_inputs_are_minimal_and_baseline_free() -> None:
    manifest = json.loads(REACTION_MANIFEST.read_text(encoding="utf-8"))
    metadata = json.loads(PROBLEM_METADATA.read_text(encoding="utf-8"))
    assert manifest["schema"] == 1
    assert set(manifest) == {
        "schema",
        "reaction_keys",
        "reaction_target_hartree",
        "reactions",
    }
    assert manifest["reaction_keys"] == REACTION_KEYS
    assert set(manifest["reactions"]) == set(REACTION_KEYS)
    assert manifest["reaction_target_hartree"] == CHEMICAL_ACCURACY_HARTREE
    assert set(metadata["species"]) == SPECIES
    assert metadata["phase_encoding"]["method"] == (
        "centered_nonidentity_pauli_one_norm"
    )
    assert all("target_time" not in row for row in metadata["species"].values())


def test_external_cache_manifest_is_exact_and_minimal() -> None:
    entries: list[tuple[str, str]] = []
    for line in CACHE_CHECKSUMS.read_text(encoding="utf-8").splitlines():
        digest, relative = line.split(maxsplit=1)
        entries.append((digest, relative))
        assert len(digest) == 64
        int(digest, 16)
        assert relative.startswith("problem_cache/")
        assert Path(relative).suffix in {".json", ".npz"}

    paths = [relative for _digest, relative in entries]
    assert len(entries) == 42
    assert len(set(paths)) == 42
    assert {Path(path).stem for path in paths} == SPECIES
    assert all(
        {Path(path).suffix for path in paths if Path(path).stem == species} == {".json", ".npz"}
        for species in SPECIES
    )
    assert not (INPUT_ROOT / "problem_cache").exists()
