"""Build cached exact-ground reaction references for a new shared suite.

The source selection must point to immutable species Hamiltonians and existing
exact references. This command never recalibrates a baseline or edits a source.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from hsopt.evaluation import compiled_unitary as judge


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_new(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != encoded:
            raise FileExistsError(f"existing sealed output differs: {path}")
        return
    path.write_text(encoded, encoding="utf-8")


def build(
    selection: Path, output: Path, species_reference_root: Path, *, include_held: bool
) -> dict:
    selection = selection.resolve()
    output = output.resolve()
    species_reference_root = species_reference_root.resolve()
    original = read(selection)
    cases = [*original["cases"]]
    if include_held:
        cases.extend(original.get("held", []))
    if len({row["id"] for row in cases}) != len(cases):
        raise ValueError("duplicate reaction case ID")
    if not cases:
        raise ValueError("empty reaction selection")
    converted = []
    provenance = []
    for row in cases:
        source = (selection.parent / row["source"]).resolve()
        old_reference = (selection.parent / row["reference"]).resolve()
        cached = read(old_reference)
        reaction = read(source)
        species = {str(item["species"]) for item in reaction["terms"]}
        if species != set(cached["species"]):
            raise ValueError(f"species reference mismatch for {row['id']}")
        ground_species = {}
        species_hashes = {}
        for name in species:
            path = species_reference_root / name / "reference.json"
            ground_species[name] = read(path)
            species_hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        ground = judge.assemble_exact_difference_reference(
            source, ground_species, backend="cached_exact_ground",
            energy_target="ground_branch",
        )
        reference = output / "references" / f"{row['id']}.json"
        write_new(reference, ground)
        converted.append({
            "id": row["id"], "source": str(source), "reference": str(reference),
            **({"calibration": str((selection.parent / row["calibration"]).resolve())}
               if "calibration" in row else {}),
            **({"calibration_root": str((selection.parent / row["calibration_root"]).resolve())}
               if "calibration_root" in row else {}),
        })
        provenance.append({
            "id": row["id"],
            "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "prior_reference_sha256": hashlib.sha256(old_reference.read_bytes()).hexdigest(),
            "species_ground_reference_sha256": species_hashes,
            "ground_reference_sha256": hashlib.sha256(reference.read_bytes()).hexdigest(),
        })
    result = {"cases": converted, "energy_target": "ground_branch"}
    write_new(output / "selection.json", result)
    write_new(output / "provenance.json", {
        "schema": "hsopt-ground-reaction-reference-v1",
        "selection_sha256": hashlib.sha256(selection.read_bytes()).hexdigest(),
        "included_held": include_held,
        "cases": provenance,
    })
    return {"cases": len(converted), "selection": str(output / "selection.json")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--species-reference-root", required=True, type=Path)
    parser.add_argument("--include-held", action="store_true")
    args = parser.parse_args()
    print(json.dumps(build(
        args.selection, args.output, args.species_reference_root,
        include_held=args.include_held,
    )))


if __name__ == "__main__":
    main()
