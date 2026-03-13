"""Composite widget for adding and configuring constraints."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.message import Message
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import Button, Checkbox, DataTable, Input, Label, Select

from trovolut.tui.scaffold import BACKEND_CONSTRAINTS, CONSTRAINT_PARAMS


class ConstraintEditor(Widget):
    """Interactive editor for building a list of constraint configurations.

    Attributes
    ----------
    backend : str
        Currently selected backend (determines which constraint types are
        available in the dropdown).
    """

    DEFAULT_CSS = """
    ConstraintEditor {
        layout: vertical;
        height: auto;
    }
    """

    backend: reactive[str] = reactive("cpu")

    class Changed(Message):
        """Posted when the constraint list changes."""

    def __init__(self, backend: str = "cpu", **kwargs) -> None:
        super().__init__(**kwargs)
        self.backend = backend
        self._constraints: list[dict] = []

    def compose(self) -> ComposeResult:
        with Vertical(id="constraint-editor"):
            yield Label("Constraint type")
            yield Select(
                self._type_options(),
                id="constraint-type-select",
                prompt="Choose constraint...",
            )
            yield Vertical(id="constraint-params")
            yield Button("Add constraint", id="constraint-add-btn", variant="primary")
            yield DataTable(id="constraint-table")

    def on_mount(self) -> None:
        table = self.query_one("#constraint-table", DataTable)
        table.add_columns("Type", "Parameters", "")

    # ── reactive watchers ─────────────────────────────────

    def watch_backend(self, new_backend: str) -> None:
        try:
            sel = self.query_one("#constraint-type-select", Select)
        except Exception:
            return
        sel.set_options(self._type_options())
        # Clear params
        container = self.query_one("#constraint-params", Vertical)
        container.remove_children()

    # ── events ────────────────────────────────────────────

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id != "constraint-type-select":
            return
        self._render_params(str(event.value) if event.value is not Select.BLANK else None)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "constraint-add-btn":
            self._add_current()
        elif str(event.button.id).startswith("remove-constraint-"):
            idx = int(str(event.button.id).split("-")[-1])
            self._remove(idx)

    # ── public API ────────────────────────────────────────

    @property
    def constraints_config(self) -> list[dict]:
        """Return the current list of constraint dicts."""
        return list(self._constraints)

    # ── internals ─────────────────────────────────────────

    def _type_options(self) -> list[tuple[str, str]]:
        available = BACKEND_CONSTRAINTS.get(self.backend, [])
        return [(name, name) for name in available]

    def _render_params(self, constraint_type: str | None) -> None:
        container = self.query_one("#constraint-params", Vertical)
        container.remove_children()
        if constraint_type is None or constraint_type not in CONSTRAINT_PARAMS:
            return
        for p in CONSTRAINT_PARAMS[constraint_type]:
            if p["type"] == "bool":
                container.mount(Checkbox(p["name"], value=p["default"], id=f"cparam-{p['name']}"))
            else:
                container.mount(Label(f"{p['name']} ({p['type']}, default: {p['default']})"))
                container.mount(
                    Input(
                        value=str(p["default"]),
                        id=f"cparam-{p['name']}",
                        placeholder=str(p["default"]),
                    )
                )

    def _add_current(self) -> None:
        sel = self.query_one("#constraint-type-select", Select)
        if sel.value is Select.BLANK:
            return
        constraint_type = str(sel.value)
        params = self._collect_params(constraint_type)
        entry = {"type": constraint_type, "params": params}
        self._constraints.append(entry)
        self._refresh_table()
        self.post_message(self.Changed())

    def _collect_params(self, constraint_type: str) -> dict:
        params: dict = {}
        for p in CONSTRAINT_PARAMS.get(constraint_type, []):
            wid = f"cparam-{p['name']}"
            try:
                if p["type"] == "bool":
                    params[p["name"]] = self.query_one(f"#{wid}", Checkbox).value
                else:
                    raw = self.query_one(f"#{wid}", Input).value
                    if p["type"] == "float":
                        params[p["name"]] = float(raw)
                    elif p["type"] == "int":
                        params[p["name"]] = int(raw)
                    else:
                        params[p["name"]] = raw
            except Exception:
                params[p["name"]] = p["default"]
        return params

    def _remove(self, idx: int) -> None:
        if 0 <= idx < len(self._constraints):
            self._constraints.pop(idx)
            self._refresh_table()
            self.post_message(self.Changed())

    def _refresh_table(self) -> None:
        table = self.query_one("#constraint-table", DataTable)
        table.clear()
        for i, c in enumerate(self._constraints):
            param_str = ", ".join(f"{k}={v!r}" for k, v in c["params"].items())
            table.add_row(
                c["type"],
                param_str,
                f"[red]remove[/]",
                key=f"constraint-{i}",
            )
        # Mount remove buttons below table (using row key click instead)
        # The DataTable doesn't support inline buttons, so we use a simple
        # approach: a remove-all button when there are constraints.
