"""Materialize the default 115-Hamiltonian shared ELOTE selection externally."""

import argparse
from pathlib import Path
from hsopt.campaign.setup import prepare_hamlib
from hsopt.data.acquire import download_hamlib


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    if args.output.resolve().is_relative_to(root):
        parser.error("generated inputs belong outside the Git checkout")
    selection = root / "experiments/baseline/molecules.json"
    checksums = selection
    if args.download:
        download_hamlib(selection, checksums, args.archive_root)
    manifest = prepare_hamlib(
        active_manifest=selection,
        checksum_manifest=checksums,
        data_root=args.archive_root,
        output_root=args.output,
        max_qubits=16,
    )
    print(f"Prepared {manifest['selected_instances']} Hamiltonian inputs")


if __name__ == "__main__":
    main()
