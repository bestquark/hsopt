from __future__ import annotations

from hsopt.tgate_opt.common import discover_lanes, evaluate_command, resolve_lane
from hsopt.tgate_opt.t_autoresearch_cli import build_agent_prompt


def test_discovers_canonical_pfs_lanes() -> None:
    lanes = discover_lanes()
    assert {lane.key for lane in lanes} == {
        "pf:h2",
        "pf:bh",
        "pf:lih",
        "pf:dmds",
        "pf:n2",
        "pf:co",
        "pf:nh3",
        "pf:hcn",
    }
    assert all(lane.supported for lane in lanes)


def test_resolves_legacy_pfs_target_and_builds_commands() -> None:
    lane = resolve_lane(target="pfs/h2")
    assert lane.key == "pf:h2"

    old_layout_lane = resolve_lane(target="examples/ft_protocols/pfs/h2")
    assert old_layout_lane.key == "pf:h2"

    energy = evaluate_command(lane, evaluator="energy")
    assert energy[:3] == ["uv", "run", "python"]
    assert energy[3].endswith("examples/pfs/evaluate_energy.py")
    assert energy[-4:] == ["--molecule", "h2", "--script", str(lane.editable_script)]

    fidelity = evaluate_command(lane, evaluator="fidelity", epsilon=1e-3)
    assert fidelity[3].endswith("examples/pfs/evaluate_fidelity.py")
    assert fidelity[-2:] == ["--epsilon", "0.001"]


def test_agent_prompt_keeps_mutation_scope_on_simple_pf() -> None:
    lane = resolve_lane(target="examples/pfs/h2")
    prompt = build_agent_prompt(lane, evaluator="energy")

    assert "examples/pfs/h2/simple_pf.py" in prompt
    assert "Edit only `examples/pfs/h2/simple_pf.py`" in prompt
    assert "evaluate_energy.py" in prompt
