"""Bounded chemistry checks and independent fermionic-convention regression."""

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from hsopt.campaign.setup import prepare_reactions
from hsopt.data import reactions as generation
from hsopt.data.acquire import _checksums, _sha256

REPO = Path(__file__).resolve().parents[2]
INPUTS = REPO / "baseline"
FIXTURE = Path(__file__).parent / "fixtures/reaction_h2_integrals.json"


def _h2_integrals():
    payload = json.loads(FIXTURE.read_text())
    return {key: np.asarray(payload[key]) for key in ("one_body", "two_body")} | {
        "constant": payload["constant"]
    }


def _annihilators(n):
    """Independent occupation-basis operators, qubit zero leftmost."""
    result = []
    for p in range(n):
        operator = np.zeros((2**n, 2**n))
        for state in range(2**n):
            mask = 1 << (n - 1 - p)
            if state & mask:
                preceding = sum((state >> (n - 1 - q)) & 1 for q in range(p))
                operator[state ^ mask, state] = (-1) ** preceding
        result.append(operator)
    return result


def _physical_matrix(integrals):
    """Direct chemist-integral formula, with annihilators in physical order."""
    h1, eri = integrals["one_body"], integrals["two_body"]
    n = h1.shape[0]
    annihilate = _annihilators(2 * n)
    create = [op.T for op in annihilate]
    one, interaction = np.zeros((2 ** (2 * n),) * 2), np.zeros((2 ** (2 * n),) * 2)
    for p, q in np.ndindex(h1.shape):
        for spin in range(2):
            one += h1[p, q] * create[2 * p + spin] @ annihilate[2 * q + spin]
    for p, q, r, s in np.ndindex(eri.shape):
        for sigma in range(2):
            for tau in range(2):
                interaction += (
                    0.5
                    * eri[p, q, r, s]
                    * create[2 * p + sigma]
                    @ create[2 * r + tau]
                    @ annihilate[2 * s + tau]
                    @ annihilate[2 * q + sigma]
                )
    core = integrals["constant"] * np.eye(one.shape[0])
    return core + one + interaction, core + one - 2 * interaction


def _spin_matrix(one, two, constant):
    annihilate = _annihilators(one.shape[0])
    create = [op.T for op in annihilate]
    matrix = constant * np.eye(2 ** one.shape[0])
    for p, q in np.ndindex(one.shape):
        matrix += one[p, q] * create[p] @ annihilate[q]
    for p, q, r, s in np.ndindex(two.shape):
        if two[p, q, r, s]:
            matrix += two[p, q, r, s] * create[p] @ create[q] @ annihilate[r] @ annihilate[s]
    return matrix


def _ground_energy(matrix, electrons):
    indices = [i for i in range(matrix.shape[0]) if i.bit_count() == electrons]
    return float(np.linalg.eigvalsh(matrix[np.ix_(indices, indices)])[0])


def test_fixed_species_cover_ten_balanced_reactions():
    definitions, reactions = generation.load_definitions(
        INPUTS / "reactions.json", INPUTS / "reactions.json"
    )
    assert len(definitions["species"]) == 21
    assert len(reactions["reaction_keys"]) == 10
    for reaction in reactions["reactions"].values():
        balance, charge = Counter(), 0
        for term in reaction["terms"]:
            spec = definitions["species"][term["species"]]
            for line in spec["geometry"].splitlines():
                atom, *coordinates = line.split()
                assert len(coordinates) == 3
                assert np.isfinite([float(x) for x in coordinates]).all()
                balance[atom] += term["coefficient"]
            charge += spec["charge"] * term["coefficient"]
        assert all(value == 0 for value in balance.values())
        assert charge == 0


@pytest.mark.parametrize("convention", generation.CONVENTIONS)
def test_two_electron_mapping_matches_independent_occupation_formula(convention):
    integrals = _h2_integrals()
    physical, historical = _physical_matrix(integrals)
    one, two = generation.spatial_to_spin_integrals(
        integrals["one_body"], integrals["two_body"], convention=convention
    )
    matrix = _spin_matrix(one, two, integrals["constant"])
    np.testing.assert_allclose(
        matrix, physical if convention == "physical" else historical, atol=1e-13
    )
    expected = json.loads(FIXTURE.read_text())[
        "physical_ground_energy" if convention == "physical" else "historical_ground_energy"
    ]
    assert _ground_energy(matrix, 2) == pytest.approx(expected, abs=1e-12)


def test_jordan_wigner_labels_and_coefficients_reconstruct_physical_matrix():
    pytest.importorskip("openfermion")
    integrals = _h2_integrals()
    physical, _ = _physical_matrix(integrals)
    identity, terms = generation.pauli_decomposition(integrals)
    paulis = {
        "I": np.eye(2),
        "X": np.array([[0, 1], [1, 0]]),
        "Y": np.array([[0, -1j], [1j, 0]]),
        "Z": np.diag([1, -1]),
    }
    matrix = identity * np.eye(16, dtype=complex)
    for term in terms:
        operator = np.ones((1, 1))
        for char in term["label"]:
            operator = np.kron(operator, paulis[char])
        matrix += term["coefficient"] * operator
    np.testing.assert_allclose(matrix, physical, atol=1e-12)
    assert [term["label"] for term in terms] == sorted(term["label"] for term in terms)


def test_pyscf_h2_integrals_match_fci_energy():
    pyscf = pytest.importorskip("pyscf")
    pytest.importorskip("openfermion")
    definitions, _ = generation.load_definitions(
        INPUTS / "reactions.json", INPUTS / "reactions.json"
    )
    integrals = generation.active_space_integrals(definitions["species"]["h2"], threads=1)
    matrix, _ = _physical_matrix(integrals)
    # FCI is an independent PySCF energy check, including the core constant.
    energy, _ = pyscf.fci.direct_spin1.kernel(
        integrals["one_body"], integrals["two_body"], 2, 2, ecore=integrals["constant"]
    )
    assert _ground_energy(matrix, 2) == pytest.approx(energy, abs=1e-10)
    assert energy == pytest.approx(
        json.loads(FIXTURE.read_text())["physical_ground_energy"], abs=1e-10
    )


def _fake_chemistry(monkeypatch):
    def integrals(spec, **kwargs):
        n = spec["active_orbitals"]
        return {
            "one_body": np.eye(n),
            "two_body": np.zeros((n,) * 4),
            "constant": 0.0,
            "mo_active": np.eye(n),
            "rhf_energy_hartree": -1.0,
            "frozen_core_orbitals": 0,
        }

    def decompose(integrals, **kwargs):
        n = 2 * integrals["one_body"].shape[0]
        return 0.0, [{"label": "Z" + "I" * (n - 1), "coefficient": 1.0}]

    monkeypatch.setattr(generation, "active_space_integrals", integrals)
    monkeypatch.setattr(generation, "pauli_decomposition", decompose)
    monkeypatch.setattr(generation.importlib.metadata, "version", lambda name: "test")


def test_generated_checksums_and_metadata_feed_all_reaction_preparation(tmp_path, monkeypatch):
    _fake_chemistry(monkeypatch)
    output = tmp_path / "raw"
    result = generation.generate_reactions(
        INPUTS / "reactions.json", INPUTS / "reactions.json", output
    )
    assert result["species_generated"] == 21
    assert result["hamiltonian_convention"] == "physical"
    assert result["complete_reaction_set"]
    checksums = _checksums(output / "problem_cache.sha256")
    assert len(checksums) == 42
    assert all(_sha256(output / name) == digest for name, digest in checksums.items())
    manifest = prepare_reactions(
        reaction_manifest=INPUTS / "reactions.json",
        cache_checksums=output / "problem_cache.sha256",
        cache_root=output / "problem_cache",
        output_root=tmp_path / "prepared",
    )
    assert manifest["selected_instances"] == 10
    h2 = json.loads((tmp_path / "prepared/h2_cl2_hcl/hamiltonians/h2.json").read_text())
    assert h2["source"]["generation"]["hamiltonian_convention"] == "physical"
    metadata = json.loads((output / "problem_metadata.json").read_text())
    assert len(metadata["species"]) == 21
    provenance = json.loads((output / "provenance.json").read_text())
    assert provenance["source"]["species_sha256"] == _sha256(INPUTS / "reactions.json")
    with np.load(output / "problem_cache/h2.npz", allow_pickle=False) as arrays:
        assert "hamiltonian_real" not in arrays


def test_generation_failure_leaves_no_partial_output(tmp_path, monkeypatch):
    _fake_chemistry(monkeypatch)

    def fail(spec, **kwargs):
        raise RuntimeError("RHF did not converge")

    monkeypatch.setattr(generation, "active_space_integrals", fail)
    output = tmp_path / "raw"
    with pytest.raises(RuntimeError, match="did not converge"):
        generation.generate_reactions(INPUTS / "reactions.json", INPUTS / "reactions.json", output)
    assert list(tmp_path.iterdir()) == []


def test_existing_campaign_inputs_are_never_replaced(tmp_path):
    original = tmp_path / "inputs"
    original.mkdir()
    sentinel = original / "sealed.json"
    sentinel.write_text("original campaign")
    with pytest.raises(FileExistsError, match="fresh output"):
        generation.generate_reactions(
            INPUTS / "reactions.json", INPUTS / "reactions.json", original
        )
    assert sentinel.read_text() == "original campaign"


def test_partial_generation_is_labeled_and_historical_mode_explicit(tmp_path, monkeypatch):
    _fake_chemistry(monkeypatch)
    output = tmp_path / "historical"
    result = generation.generate_reactions(
        INPUTS / "reactions.json",
        INPUTS / "reactions.json",
        output,
        convention="historical-replay",
        species=["h2"],
    )
    assert result["species_generated"] == 1
    assert not result["complete_reaction_set"]
    metadata = json.loads((output / "problem_cache/h2.json").read_text())
    assert metadata["hamiltonian_convention"] == "historical-replay"


def test_unknown_species_is_rejected_before_calculation(tmp_path):
    with pytest.raises(ValueError, match="fixed registry"):
        generation.generate_reactions(
            INPUTS / "reactions.json",
            INPUTS / "reactions.json",
            tmp_path / "raw",
            species=["unknown"],
        )
    assert not (tmp_path / "raw").exists()
