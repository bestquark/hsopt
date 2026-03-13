"""EvaluatorScaffoldScreen — wizard to generate an evaluator.py file."""

from __future__ import annotations

from pathlib import Path

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.screen import Screen
from textual.widgets import (
    Button,
    Footer,
    Header,
    Input,
    Label,
    Select,
    TabbedContent,
    TabPane,
    TextArea,
)

from trovolut.tui.scaffold import RUNNER_MAP, generate_evaluator_scaffold
from trovolut.tui.widgets.constraint_editor import ConstraintEditor


class EvaluatorScaffoldScreen(Screen):
    """Wizard to generate an OpenEvolve-compatible evaluator file."""

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical(classes="wizard-container"):
            with TabbedContent():
                with TabPane("Backend", id="tab-backend"):
                    with Vertical(classes="wizard-tab"):
                        yield Label("Backend")
                        yield Select(
                            [
                                ("cpu", "cpu"),
                                ("gpu", "gpu"),
                                ("gpu-mp-emulator", "gpu-mp-emulator"),
                                ("qpu", "qpu"),
                            ],
                            id="eval-backend",
                            value="cpu",
                        )
                        yield Label("Runner (auto-selected)")
                        yield Input(
                            value=RUNNER_MAP["cpu"],
                            id="eval-runner-display",
                            disabled=True,
                        )
                        yield Label("GPUs (LegateRunner only)")
                        yield Input(value="1", id="eval-gpus")
                        yield Label("CPUs (LegateRunner only)")
                        yield Input(value="4", id="eval-cpus")
                        yield Label("Framebuffer MB (LegateRunner only)")
                        yield Input(value="256", id="eval-fbmem")

                with TabPane("Constraints", id="tab-constraints"):
                    with Vertical(classes="wizard-tab"):
                        yield ConstraintEditor(backend="cpu", id="eval-constraint-editor")

                with TabPane("Parameters", id="tab-params"):
                    with Vertical(classes="wizard-tab"):
                        yield Label("run_fn_name")
                        yield Input(value="run_dynamics", id="eval-run-fn")
                        yield Label("n_trials")
                        yield Input(value="10", id="eval-n-trials")
                        yield Label("timeout (seconds)")
                        yield Input(value="30", id="eval-timeout")

                with TabPane("Generate", id="tab-generate"):
                    with Vertical(classes="wizard-tab"):
                        yield Label("Output path")
                        yield Input(value="evaluator.py", id="eval-output-path")
                        yield Button("Preview", id="eval-preview-btn", variant="default")
                        yield TextArea(
                            "",
                            language="python",
                            read_only=True,
                            id="preview-area",
                        )
                        yield Button("Generate", id="generate-btn", variant="success")
        yield Footer()

    # ── events ────────────────────────────────────────────

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id == "eval-backend":
            backend = str(event.value)
            self.query_one("#eval-runner-display", Input).value = RUNNER_MAP.get(
                backend, "CPURunner"
            )
            try:
                self.query_one("#eval-constraint-editor", ConstraintEditor).backend = backend
            except Exception:
                pass

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "eval-preview-btn":
            code = self._build_scaffold()
            self.query_one("#preview-area", TextArea).text = code
        elif event.button.id == "generate-btn":
            self._write_file()

    # ── scaffold helpers ──────────────────────────────────

    def _build_scaffold(self) -> str:
        backend = str(self.query_one("#eval-backend", Select).value)
        editor = self.query_one("#eval-constraint-editor", ConstraintEditor)
        return generate_evaluator_scaffold(
            backend=backend,
            constraints_config=editor.constraints_config,
            run_fn_name=self.query_one("#eval-run-fn", Input).value or "run_dynamics",
            n_trials=int(self.query_one("#eval-n-trials", Input).value or 10),
            timeout=int(self.query_one("#eval-timeout", Input).value or 30),
            gpus=int(self.query_one("#eval-gpus", Input).value or 1),
            cpus=int(self.query_one("#eval-cpus", Input).value or 4),
            fbmem_mb=int(self.query_one("#eval-fbmem", Input).value or 256),
        )

    def _write_file(self) -> None:
        code = self._build_scaffold()
        out_path = Path(self.query_one("#eval-output-path", Input).value or "evaluator.py")
        try:
            out_path.write_text(code, encoding="utf-8")
            self.notify(f"Evaluator written to {out_path}", severity="information")
        except Exception as exc:
            self.notify(f"Error: {exc}", severity="error")
