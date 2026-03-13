"""DataTable widget displaying autoresearch experiment records."""

from __future__ import annotations

from datetime import datetime
from typing import List

from textual.widgets import DataTable


class ExperimentTable(DataTable):
    """DataTable pre-populated with autoresearch experiment results."""

    def on_mount(self) -> None:
        self.add_columns("Iter", "Score", "OK", "Time (s)", "Error", "Timestamp")

    def update_records(self, records: list, append_only: bool = True) -> None:
        """Refresh table from a list of ExperimentRecord objects.

        Parameters
        ----------
        records : list[ExperimentRecord]
            Full list of records.
        append_only : bool
            If True, only add rows beyond the current row count.
        """
        start = self.row_count if append_only else 0
        if not append_only:
            self.clear()
            start = 0

        for rec in records[start:]:
            score_str = f"{rec.combined_score:+.4f}"
            ok_str = "[green]yes[/]" if rec.all_satisfied else "[red]no[/]"
            time_val = rec.metrics.get("time", rec.metrics.get("time_seconds", "—"))
            if isinstance(time_val, float):
                time_str = f"{time_val:.3f}"
            else:
                time_str = str(time_val)
            error_val = rec.metrics.get("error", rec.metrics.get("mean_error", "—"))
            if isinstance(error_val, float):
                error_str = f"{error_val:.6f}"
            else:
                error_str = str(error_val)
            try:
                ts = datetime.fromtimestamp(rec.timestamp).strftime("%H:%M:%S")
            except Exception:
                ts = "—"
            self.add_row(
                str(rec.iteration), score_str, ok_str, time_str, error_str, ts,
                key=f"iter-{rec.iteration}",
            )
