"""WorkflowScaffoldScreen — wizard to generate an initial_program.py file."""

from __future__ import annotations

from pathlib import Path

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.screen import Screen
from textual.widgets import (
    Button,
    Checkbox,
    Footer,
    Header,
    Input,
    Label,
    Select,
    TabbedContent,
    TabPane,
    TextArea,
)

from trovolut.tui.scaffold import EVOLUTION_METHODS, generate_workflow_scaffold


class WorkflowScaffoldScreen(Screen):
    """Wizard to generate an initial-program (workflow) file."""

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical(classes="wizard-container"):
            with TabbedContent():
                with TabPane("Backend & Method", id="tab-bm"):
                    with Vertical(classes="wizard-tab"):
                        yield Label("Backend")
                        yield Select(
                            [
                                ("cpu", "cpu"),
                                ("gpu", "gpu"),
                                ("gpu-mp-emulator", "gpu-mp-emulator"),
                                ("qpu", "qpu"),
                            ],
                            id="wf-backend",
                            value="cpu",
                        )
                        yield Label("Evolution method")
                        yield Select(
                            self._method_options("cpu"),
                            id="wf-method",
                            value="TrotterEvolution",
                        )
                        yield Label("Trotter steps")
                        yield Input(value="50", id="wf-steps")
                        yield Label("Order (1, 2, 4, …)")
                        yield Select(
                            [("1", "1"), ("2", "2"), ("4", "4"), ("6", "6"), ("8", "8")],
                            id="wf-order",
                            value="2",
                        )

                with TabPane("Options", id="tab-options"):
                    with Vertical(classes="wizard-tab"):
                        yield Checkbox(
                            "Include API reference comments",
                            value=True,
                            id="wf-api-ref",
                        )
                        yield Checkbox(
                            "Include EVOLVE-BLOCK markers",
                            value=True,
                            id="wf-evo-markers",
                        )

                with TabPane("Generate", id="tab-wf-generate"):
                    with Vertical(classes="wizard-tab"):
                        yield Label("Output path")
                        yield Input(value="initial_program.py", id="wf-output-path")
                        yield Button("Preview", id="wf-preview-btn", variant="default")
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
        if event.select.id == "wf-backend":
            backend = str(event.value)
            method_select = self.query_one("#wf-method", Select)
            options = self._method_options(backend)
            method_select.set_options(options)
            if options:
                method_select.value = options[0][1]

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "wf-preview-btn":
            code = self._build_scaffold()
            self.query_one("#preview-area", TextArea).text = code
        elif event.button.id == "generate-btn":
            self._write_file()

    # ── helpers ───────────────────────────────────────────

    @staticmethod
    def _method_options(backend: str) -> list[tuple[str, str]]:
        methods = EVOLUTION_METHODS.get(backend, [])
        return [(m, m) for m in methods]

    def _build_scaffold(self) -> str:
        backend = str(self.query_one("#wf-backend", Select).value)
        method = str(self.query_one("#wf-method", Select).value)
        return generate_workflow_scaffold(
            backend=backend,
            evolution_method=method,
            trotter_steps=int(self.query_one("#wf-steps", Input).value or 50),
            order=int(str(self.query_one("#wf-order", Select).value)),
            include_api_reference=self.query_one("#wf-api-ref", Checkbox).value,
            include_evo_markers=self.query_one("#wf-evo-markers", Checkbox).value,
        )

    def _write_file(self) -> None:
        code = self._build_scaffold()
        out_path = Path(self.query_one("#wf-output-path", Input).value or "initial_program.py")
        try:
            out_path.write_text(code, encoding="utf-8")
            self.notify(f"Workflow written to {out_path}", severity="information")
        except Exception as exc:
            self.notify(f"Error: {exc}", severity="error")
