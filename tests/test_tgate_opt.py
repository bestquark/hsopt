from __future__ import annotations

from hsopt.tgate_opt.common import REPO_ROOT, discover_lanes


def test_clean_exact_branch_has_no_legacy_pfs_lanes() -> None:
    assert discover_lanes() == []


def test_agent_skill_contains_the_runtime_package() -> None:
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    cli_guide = (REPO_ROOT / "skills" / "hsopt" / "references" / "cli.md").read_text(
        encoding="utf-8"
    )
    pyproject = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    skill = (REPO_ROOT / "skills" / "hsopt" / "SKILL.md").read_text(encoding="utf-8")
    runtime = REPO_ROOT / "skills" / "hsopt"
    readme_words = " ".join(readme.split())
    skill_words = " ".join(skill.split())

    assert runtime.is_dir()
    assert (runtime / "SKILL.md").is_file()
    assert (runtime / "references" / "mutation.md").is_file()
    assert sorted(path.name for path in runtime.glob("*.py")) == ["__init__.py"]
    assert {
        "campaign",
        "data",
        "evaluation",
        "sandbox",
        "tgate_opt",
        "view",
    } <= {path.name for path in runtime.iterdir() if path.is_dir()}
    assert (runtime / "evaluation" / "compiled_unitary.py").is_file()
    assert (runtime / "sandbox" / "openshell.py").is_file()
    assert not (runtime / "paper_baseline.py").exists()
    generator = REPO_ROOT / "experiments" / "baseline" / "generate_initial_pf.py"
    generator_source = generator.read_text(encoding="utf-8")
    calibration_source = (runtime / "campaign" / "calibration.py").read_text(encoding="utf-8")
    assert "def calibrate_problem(" in calibration_source
    assert "def optimize_paper_error_allocation(" in calibration_source
    assert "class ProductFormula:" in generator_source
    assert "def product_formula(" in generator_source
    assert "def render(" in generator_source
    assert (REPO_ROOT / "tests").is_dir()
    assert not (REPO_ROOT / "src").exists()
    assert not (REPO_ROOT / "cli").exists()
    assert not (REPO_ROOT / "skills" / "hsopt" / "scripts").exists()
    assert "git clone https://github.com/bestquark/hsopt.git" in readme
    assert "skills/hsopt/" in readme_words
    assert 'module-root = "skills"' in pyproject
    assert "## OpenShell boundary and telemetry" in cli_guide
    assert "hsopt-openshell audit" in cli_guide
    assert "## Agent skill" in readme
    assert "repository runtime is ordinary package code" in skill_words
    assert len(skill.splitlines()) < 150
