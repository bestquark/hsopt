from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from hsopt.campaign.queue import (
    enqueue,
    queue_status,
)
from hsopt.evaluation.compiled_unitary import (
    DIFFERENCE_ERROR_CONTRACT_CANCELLATION,
    DIFFERENCE_ERROR_CONTRACT_INDIVIDUAL,
)


REPO = Path(__file__).resolve().parents[2]
ABSOLUTE = REPO / "cli" / "tests" / "fixtures" / "abs_energy_h2"
DIFFERENCE = REPO / "cli" / "tests" / "fixtures" / "diff_energy_h2_bond_stretch"


def test_concurrent_queue_insertion_remains_content_deduplicated(
    tmp_path: Path,
) -> None:
    def insert(_: int) -> dict[str, object]:
        return enqueue(
            tmp_path,
            kind="absolute",
            source=ABSOLUTE / "hamiltonian.json",
            policy=ABSOLUTE / "initial_pf.py",
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(insert, range(8)))

    assert len({str(result["task_id"]) for result in results}) == 1
    assert sum(result["status"] == "queued" for result in results) == 1
    assert sum(result["status"] == "exists" for result in results) == 7
    assert queue_status(tmp_path)["pending"] == 1


def test_difference_contracts_have_distinct_queue_identities(tmp_path: Path) -> None:
    queued = {
        contract: enqueue(
            tmp_path,
            kind="difference",
            source=DIFFERENCE / "reaction.json",
            policy=DIFFERENCE / "initial_pf.py",
            difference_error_contract=contract,
        )
        for contract in (
            DIFFERENCE_ERROR_CONTRACT_INDIVIDUAL,
            DIFFERENCE_ERROR_CONTRACT_CANCELLATION,
        )
    }

    assert (
        queued[DIFFERENCE_ERROR_CONTRACT_INDIVIDUAL]["task_id"]
        != queued[DIFFERENCE_ERROR_CONTRACT_CANCELLATION]["task_id"]
    )
    for contract, record in queued.items():
        task = tmp_path / "pending" / record["task_id"] / "task.json"
        assert json.loads(task.read_text())["difference_error_contract"] == contract
