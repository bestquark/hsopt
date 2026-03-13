"""TrovolutApp — main Textual application."""

from textual.app import App
from textual.binding import Binding


class TrovolutApp(App):
    """Trovolut interactive terminal interface."""

    TITLE = "trovolut"
    CSS_PATH = "tui.tcss"

    SCREENS = {
        "home": "trovolut.tui.screens.home.HomeScreen",
        "backends": "trovolut.tui.screens.backends.BackendExplorerScreen",
        "evaluator": "trovolut.tui.screens.evaluator.EvaluatorScaffoldScreen",
        "workflow": "trovolut.tui.screens.workflow.WorkflowScaffoldScreen",
        "autoresearch": "trovolut.tui.screens.autoresearch.AutoresearchScreen",
    }

    BINDINGS = [
        Binding("h", "switch_screen('home')", "Home", show=True),
        Binding("b", "switch_screen('backends')", "Backends", show=True),
        Binding("e", "switch_screen('evaluator')", "Evaluator", show=True),
        Binding("w", "switch_screen('workflow')", "Workflow", show=True),
        Binding("a", "switch_screen('autoresearch')", "AutoResearch", show=True),
        Binding("q", "quit", "Quit", show=True),
    ]

    def on_mount(self) -> None:
        self.push_screen("home")
