from __future__ import annotations

from pathlib import Path

import pytest

from hsopt.tgate_opt.common import REPO_ROOT, discover_lanes, evaluate_command, resolve_lane
from hsopt.tgate_opt.t_autoresearch_cli import build_agent_prompt
from hsopt.tgate_opt.t_gepa_cli import build_parser as build_gepa_parser


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


@pytest.mark.parametrize(
    "arguments",
    [
        ["dataset"],
        ["evaluate-prompt", "--prompt-file", "/tmp/seed.md"],
        ["optimize"],
    ],
)
def test_hsgepa_write_commands_require_explicit_destinations(arguments: list[str]) -> None:
    with pytest.raises(SystemExit, match="2"):
        build_gepa_parser().parse_args(arguments)


def test_hsgepa_accepts_caller_supplied_external_paths(tmp_path: Path) -> None:
    dataset = build_gepa_parser().parse_args(
        ["dataset", "--output", str(tmp_path / "dataset.json")]
    )
    assert dataset.output == str(tmp_path / "dataset.json")

    evaluation = build_gepa_parser().parse_args(
        [
            "evaluate-prompt",
            "--prompt-file",
            str(tmp_path / "seed.md"),
            "--run-root",
            str(tmp_path / "manual"),
        ]
    )
    assert evaluation.run_root == str(tmp_path / "manual")

    optimization = build_gepa_parser().parse_args(
        ["optimize", "--run-root", str(tmp_path / "optimization")]
    )
    assert optimization.run_root == str(tmp_path / "optimization")


def test_install_docs_separate_agent_skill_from_benchmark_runtime() -> None:
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    skill = (REPO_ROOT / "skills" / "hsopt" / "SKILL.md").read_text(encoding="utf-8")
    readme_words = " ".join(readme.split())
    skill_words = " ".join(skill.split())

    assert "skills and the benchmark runtime are separate prerequisites" in readme_words
    assert "does not clone `examples/pfs/`" in readme_words
    assert "Clone the Benchmark Runtime" in readme
    assert "installed agent skill and the benchmark runtime as separate pieces" in skill_words
    assert "Direct evaluator runs" in skill_words and "`uv sync`" in skill_words
