"""AutoresearchScreen — live dashboard for monitoring autoresearch optimization."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import Screen
from textual.timer import Timer
from textual.widgets import (
    Button,
    Footer,
    Header,
    Input,
    Label,
    RichLog,
    Select,
    Static,
)

from trovolut.tui.widgets.experiment_table import ExperimentTable
from trovolut.tui.widgets.score_chart import ScoreChart


class AutoresearchScreen(Screen):
    """Live dashboard for autoresearch experiment monitoring.

    Features:
    - Experiment results table (auto-updates during runs)
    - Best-score sparkline chart
    - Problem configuration summary
    - Controls to load history or start new runs
    - Activity log
    """

    _poll_timer: Optional[Timer] = None
    _history_path: Optional[Path] = None
    _displayed_count: int = 0
    _running: bool = False

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal(id="ar-container"):
            # ── Left column: table + chart ──
            with Vertical(id="ar-left-col"):
                yield Label("[bold]Experiment Results[/]")
                yield ExperimentTable(id="ar-table")
                yield ScoreChart(id="ar-chart")

            # ── Right column: config + controls + log ──
            with Vertical(id="ar-right-col"):
                yield Static("[bold]Problem Config[/]", id="ar-config-title")
                yield Static("No problem loaded", id="ar-config-summary")

                yield Label("[bold]Controls[/]")
                with Vertical(id="ar-controls"):
                    yield Label("Target script")
                    yield Input(placeholder="path/to/simulation.py", id="ar-target")
                    yield Label("Mode")
                    yield Select(
                        [("openevolve", "openevolve"), ("agent", "agent")],
                        id="ar-mode",
                        value="agent",
                    )
                    yield Label("Budget (seconds)")
                    yield Input(value="300", id="ar-budget")
                    with Horizontal(id="ar-btn-row"):
                        yield Button("Load History", id="ar-load-btn", variant="default")
                        yield Button("Start", id="ar-start-btn", variant="success")

                yield Label("[bold]Activity Log[/]")
                yield RichLog(id="ar-log", wrap=True, markup=True)
        yield Footer()

    def on_mount(self) -> None:
        self._log("Dashboard ready. Load a history file or start a new run.")

    # ── event handlers ────────────────────────────────────

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "ar-load-btn":
            self._load_history()
        elif event.button.id == "ar-start-btn":
            self._start_run()

    # ── load history ──────────────────────────────────────

    def _load_history(self) -> None:
        """Load history.json from the target script's output directory."""
        target = self.query_one("#ar-target", Input).value.strip()
        if not target:
            self._log("[red]Enter a target script path first[/]")
            return

        # Try common locations for history.json
        target_path = Path(target)
        candidates = [
            target_path.parent / "autoresearch_output" / "history.json",
            target_path.parent / "history.json",
            Path("autoresearch_output") / "history.json",
        ]

        history_path = None
        for p in candidates:
            if p.exists():
                history_path = p
                break

        if history_path is None:
            self._log(f"[red]No history.json found near {target}[/]")
            self._log(f"  Searched: {', '.join(str(p) for p in candidates)}")
            return

        self._history_path = history_path
        self._refresh_from_disk()
        self._log(f"[green]Loaded {len(self._get_history().records)} records from {history_path}[/]")

    def _get_history(self):
        """Load history from disk."""
        from trovolut.autoresearch.history import History
        if self._history_path and self._history_path.exists():
            return History.load(self._history_path)
        return History()

    def _refresh_from_disk(self) -> None:
        """Refresh the dashboard from history.json on disk."""
        history = self._get_history()
        records = history.records

        table = self.query_one("#ar-table", ExperimentTable)
        chart = self.query_one("#ar-chart", ScoreChart)

        if len(records) > self._displayed_count:
            table.update_records(records, append_only=True)
            chart.update_scores([r.combined_score for r in records])
            self._displayed_count = len(records)

            # Update config summary
            summary = history.summary()
            summary_text = (
                f"Experiments: {summary['n_experiments']}\n"
                f"Satisfied: {summary['n_satisfied']}\n"
                f"Best score: {summary.get('best_score', '—')}\n"
                f"Best iter: {summary.get('best_iteration', '—')}\n"
                f"Mean score: {summary.get('mean_score', '—')}"
            )
            self.query_one("#ar-config-summary", Static).update(summary_text)

    # ── start run ─────────────────────────────────────────

    def _start_run(self) -> None:
        """Start an autoresearch optimization in a background worker."""
        if self._running:
            self._log("[yellow]A run is already in progress[/]")
            return

        target = self.query_one("#ar-target", Input).value.strip()
        if not target:
            self._log("[red]Enter a target script path[/]")
            return

        if not Path(target).exists():
            self._log(f"[red]Target not found: {target}[/]")
            return

        mode = str(self.query_one("#ar-mode", Select).value)
        budget = float(self.query_one("#ar-budget", Input).value or 300)

        self._log(f"Starting autoresearch: [bold]{target}[/] mode={mode} budget={budget}s")

        # Reset display
        self.query_one("#ar-table", ExperimentTable).clear()
        self._displayed_count = 0

        # Determine output dir and history path
        output_dir = str(Path(target).parent / "autoresearch_output")
        self._history_path = Path(output_dir) / "history.json"

        self._running = True
        self.query_one("#ar-start-btn", Button).disabled = True

        # Start polling timer
        self._poll_timer = self.set_interval(2.0, self._poll_update)

        # Run in background thread
        self.run_worker(
            self._run_loop(target, mode, budget, output_dir),
            thread=True,
        )

    def _run_loop(self, target: str, mode: str, budget: float, output_dir: str):
        """Callable for the worker thread."""
        def _work() -> None:
            from trovolut.autoresearch import Problem, Loop

            problem = Problem(
                target=target,
                budget_seconds=budget,
                output_dir=output_dir,
            )

            loop = Loop(problem, mode=mode)
            loop.run(max_iterations=50)

        return _work

    def _poll_update(self) -> None:
        """Timer callback to refresh dashboard from disk."""
        if self._history_path and self._history_path.exists():
            old_count = self._displayed_count
            self._refresh_from_disk()
            if self._displayed_count > old_count:
                new = self._displayed_count - old_count
                self._log(f"+{new} new result(s) (total: {self._displayed_count})")

    def on_worker_state_changed(self, event) -> None:
        """Handle worker completion."""
        if hasattr(event, "worker") and event.worker.is_finished:
            self._running = False
            try:
                self.query_one("#ar-start-btn", Button).disabled = False
            except Exception:
                pass
            if self._poll_timer is not None:
                self._poll_timer.stop()
                self._poll_timer = None
            # Final refresh
            self._refresh_from_disk()
            self._log("[green bold]Run complete![/]")

    # ── logging ───────────────────────────────────────────

    def _log(self, message: str) -> None:
        """Write a message to the activity log."""
        try:
            log = self.query_one("#ar-log", RichLog)
            log.write(message)
        except Exception:
            pass
