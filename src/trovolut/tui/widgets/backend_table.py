"""Reusable DataTable showing all backend statuses."""

from textual.widgets import DataTable


class BackendTable(DataTable):
    """DataTable pre-populated with trovolut backend availability."""

    def on_mount(self) -> None:
        from trovolut.backends import list_all

        self.add_columns("Backend", "Status", "Platform")
        for name, info in list_all().items():
            status = "[green]available[/]" if info["available"] else "[red]unavailable[/]"
            self.add_row(name, status, info["platform"], key=name)
