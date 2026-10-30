"""Extract the exact coefficient samples behind the approved Figure 1 inset."""

import argparse
import hashlib
import io
import json
from pathlib import Path
import re
import urllib.request
import zipfile

import h5py
import numpy as np

from _coefficient_data import INSTANCES, ZERO_THRESHOLD
from _paths import ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    registry = json.loads((ROOT / "baseline/molecules.json").read_text())
    arrays, metadata = {}, {"schema": "hsopt-schematic-coefficients-v1", "instances": {}}
    pattern = re.compile(r"^\s*\(?([^\[\)]+)\)?\s*\[")
    for instance in INSTANCES:
        relative = f"{instance.collection}/{instance.archive}"
        path = args.archive_root / relative
        expected = registry["archive_checksums"][relative]
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            with urllib.request.urlopen(registry["base_url"] + "/" + relative) as source:
                payload = source.read()
            assert hashlib.sha256(payload).hexdigest() == expected, relative
            path.write_bytes(payload)
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, relative
        with zipfile.ZipFile(path) as archive:
            names = [name for name in archive.namelist() if name.endswith(".hdf5")]
            assert len(names) == 1
            with h5py.File(io.BytesIO(archive.read(names[0])), "r") as handle:
                text = handle[instance.dataset][()].decode()
        values = []
        for raw in text.splitlines():
            line = raw.strip()
            if line.endswith("+"):
                line = line[:-1].strip()
            if not line or line.endswith("[]"):
                continue
            match = pattern.match(line)
            if match:
                magnitude = abs(float(complex(match.group(1).strip()).real))
                if magnitude > ZERO_THRESHOLD:
                    values.append(magnitude)
        key = instance.archive.removesuffix(".zip")
        arrays[key] = np.asarray(values, dtype=float)
        assert len(values) > 0
        metadata["instances"][key] = {
            "archive": relative,
            "archive_sha256": expected,
            "dataset": instance.dataset,
            "dataset_utf8_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "n_qubits": instance.nqubits,
            "zero_threshold": ZERO_THRESHOLD,
            "identity_excluded": True,
            "coefficients": len(values),
            "array_sha256": hashlib.sha256(arrays[key].tobytes()).hexdigest(),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **arrays, metadata_json=json.dumps(metadata, sort_keys=True))
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
