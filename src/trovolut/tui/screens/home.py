"""HomeScreen — dashboard with backend overview and quick actions."""

import platform
import sys

from textual.app import ComposeResult
from textual.screen import Screen
from textual.widgets import Button, Footer, Header, Static

from trovolut.tui.widgets.backend_table import BackendTable


class HomeScreen(Screen):
    """Landing screen for the Trovolut TUI."""

    def compose(self) -> ComposeResult:
        yield Header()
        with Static(id="home-container"):
            yield Static(self._banner(), id="banner")
            yield Static(self._system_info(), id="system-info")
            yield BackendTable()
            with Static(id="quick-actions"):
                yield Button("Backend Explorer", id="btn-backends", variant="primary")
                yield Button("Generate Evaluator", id="btn-evaluator", variant="primary")
                yield Button("Generate Program", id="btn-workflow", variant="primary")
                yield Button("AutoResearch", id="btn-autoresearch", variant="success")
                yield Button("Quit", id="btn-quit", variant="error")
        yield Footer()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        actions = {
            "btn-backends": "backends",
            "btn-evaluator": "evaluator",
            "btn-workflow": "workflow",
            "btn-autoresearch": "autoresearch",
        }
        screen = actions.get(event.button.id)
        if screen:
            self.app.switch_screen(screen)
        elif event.button.id == "btn-quit":
            self.app.exit()

    @staticmethod
    def _banner() -> str:
        from trovolut import __version__

        return f"trovolut v{__version__}"

    @staticmethod
    def _system_info() -> str:
        from trovolut import get_active_backend

        backend = get_active_backend() or "none"
        return (
            f"Python {sys.version.split()[0]}  |  "
            f"{platform.system()} {platform.machine()}  |  "
            f"Active backend: {backend}"
        )
