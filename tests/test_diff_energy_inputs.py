from __future__ import annotations

import hashlib
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
INPUT_ROOT = REPO_ROOT / "experiments" / "diff_energy" / "inputs"
SEED_MANIFEST = INPUT_ROOT / "seed_manifest.json"
SEED_VALIDATION = INPUT_ROOT / "seed_validation.json"
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


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_seed_inputs_are_normalized_and_validated() -> None:
    assert _sha256(SEED_MANIFEST) == (
        "9fe91f88d3af88295d4d23ac5a6612d5a7e3eec97cc0c234a3c602001e708b1b"
    )
    assert _sha256(SEED_VALIDATION) == (
        "e2bee43bbf79fd5cfa8877c0be166a115443528ca39f2c53ec596b6239555fcc"
    )

    manifest = json.loads(SEED_MANIFEST.read_text(encoding="utf-8"))
    validation = json.loads(SEED_VALIDATION.read_text(encoding="utf-8"))
    assert manifest["initial_pf_generator"] == "experiments/generate_initial_pf.py"
    assert manifest["reaction_keys"] == REACTION_KEYS
    assert set(manifest["reactions"]) == set(REACTION_KEYS)
    assert manifest["reaction_target_mha"] / 1000.0 == CHEMICAL_ACCURACY_HARTREE
    assert [row["reaction_key"] for row in validation] == REACTION_KEYS
    assert len(validation) == 10

    for row in validation:
        assert row["reaction_error_target_hartree"] == CHEMICAL_ACCURACY_HARTREE
        assert row["reaction_error_target_mha"] / 1000.0 == CHEMICAL_ACCURACY_HARTREE
        assert row["formula_order"] == 1
        assert row["n_steps"] == 512
        assert row["validation_passes"] is True
        assert set(row["policies"]) == {"equal_share", "reaction_aware"}
        assert all(
            policy["full_space_constraint_passes"] is True for policy in row["policies"].values()
        )


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
