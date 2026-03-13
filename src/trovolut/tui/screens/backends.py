"""BackendExplorerScreen — browse and inspect backends."""

from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.screen import Screen
from textual.widgets import Button, Footer, Header, OptionList, Static
from textual.widgets.option_list import Option


class BackendExplorerScreen(Screen):
    """List all backends and show details for the selected one."""

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal(id="backend-explorer"):
            yield self._build_option_list()
            yield Static("Select a backend to view details.", id="backend-detail-panel")
        yield Footer()

    def on_mount(self) -> None:
        self.query_one(OptionList).focus()

    def on_option_list_option_highlighted(self, event: OptionList.OptionHighlighted) -> None:
        self._show_details(str(event.option.id))

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        self._show_details(str(event.option.id))

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "set-active-btn":
            backend_name = event.button.name
            try:
                import trovolut as trv
                trv.set_backend(backend_name)
                self.notify(f"Active backend set to '{backend_name}'", severity="information")
            except Exception as exc:
                self.notify(str(exc), severity="error")

    # ── helpers ──────────────────────────────────────────

    @staticmethod
    def _build_option_list() -> OptionList:
        from trovolut.backends import list_all

        options = []
        for name, info in list_all().items():
            label = f"{'*' if info['available'] else ' '} {name}"
            options.append(Option(label, id=name))
        ol = OptionList(*options, id="backend-list-panel")
        return ol

    def _show_details(self, backend: str) -> None:
        from trovolut.backends import _registry

        info = _registry._backends.get(backend)
        if info is None:
            return

        available = _registry.is_available(backend)
        missing = _registry.get_missing_packages(backend)

        lines = [
            f"[bold]{backend}[/bold]",
            "",
            f"Status:   {'[green]available[/]' if available else '[red]unavailable[/]'}",
            f"Platform: {info['platform']}",
            f"Module:   {info['module']}",
            "",
            f"Description:\n  {info['description']}",
            "",
            f"Required packages: {', '.join(info['required_packages']) or 'none'}",
        ]
        if missing:
            lines.append(f"Missing packages:  [red]{', '.join(missing)}[/]")
            lines.append(f"Install: [bold]uv pip install trovolut\\[{backend}][/bold]")

        panel = self.query_one("#backend-detail-panel", Static)
        # Remove any existing button from a previous selection
        for btn in panel.query(Button):
            btn.remove()
        panel.update("\n".join(lines))

        if available:
            btn = Button("Set Active", id="set-active-btn", name=backend, variant="success")
            panel.mount(btn)
