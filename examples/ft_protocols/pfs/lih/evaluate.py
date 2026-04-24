from __future__ import annotations

import json
import os
import sys
from pathlib import Path


BENCHMARK_ROOT = Path(os.environ.get("TROVOLUT_BENCHMARK_ROOT", Path(__file__).resolve().parent)).resolve()
FT_PROTOCOL_ROOT = BENCHMARK_ROOT.parents[1]
if str(FT_PROTOCOL_ROOT) not in sys.path:
    sys.path.insert(0, str(FT_PROTOCOL_ROOT))

from pfs.evaluate import evaluate  # noqa: E402


def main() -> int:
    source = Path(os.environ.get("TROVOLUT_SOURCE_FILE", BENCHMARK_ROOT / "simple_pf.py")).resolve()
    print(json.dumps(evaluate("lih", source), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
