from __future__ import annotations

from hsopt.tgate_opt.common import REPO_ROOT, discover_lanes


def test_clean_exact_branch_has_no_legacy_pfs_lanes() -> None:
    assert discover_lanes() == []


def test_agent_skill_is_separate_from_runtime_package() -> None:
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    cli_readme = (REPO_ROOT / "cli" / "README.md").read_text(encoding="utf-8")
    pyproject = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    skill = (REPO_ROOT / "skills" / "hsopt" / "SKILL.md").read_text(encoding="utf-8")
    runtime = REPO_ROOT / "cli" / "src" / "hsopt"
    readme_words = " ".join(readme.split())
    skill_words = " ".join(skill.split())

    assert runtime.is_dir()
    assert (runtime / "compiled_unitary.py").is_file()
    assert (runtime / "openshell_agent.py").is_file()
    assert (REPO_ROOT / "cli" / "tests").is_dir()
    assert not (REPO_ROOT / "src").exists()
    assert not (REPO_ROOT / "tests").exists()
    assert not (REPO_ROOT / "skills" / "hsopt" / "scripts").exists()
    assert "git clone https://github.com/bestquark/hsopt.git" in readme
    assert "cli/src/hsopt/" in readme_words
    assert 'module-root = "cli/src"' in pyproject
    assert "## OpenShell boundary and telemetry" in cli_readme
    assert "hsopt-openshell audit" in cli_readme
    assert "## Agent skill" in readme
    assert "repository runtime is ordinary package code" in skill_words
    assert len(skill.splitlines()) < 130
