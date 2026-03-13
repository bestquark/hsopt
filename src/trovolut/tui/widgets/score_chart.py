"""Simple sparkline widget showing best-score progression."""

from __future__ import annotations

from typing import List

from textual.widgets import Static

# Unicode block elements for sparkline (8 levels)
_BLOCKS = " \u2581\u2582\u2583\u2584\u2585\u2586\u2587\u2588"


class ScoreChart(Static):
    """Renders a sparkline of best-score-so-far over iterations."""

    def __init__(self, **kwargs) -> None:
        super().__init__("", **kwargs)
        self._scores: List[float] = []

    def update_scores(self, scores: List[float]) -> None:
        """Update with the full list of combined_scores."""
        self._scores = list(scores)
        self._render_chart()

    def _render_chart(self) -> None:
        if not self._scores:
            self.update("No data yet")
            return

        # Compute running best (cumulative max)
        running_best: List[float] = []
        best_so_far = float("-inf")
        for s in self._scores:
            best_so_far = max(best_so_far, s)
            running_best.append(best_so_far)

        # Normalize to [0, 8] for block characters
        lo = min(running_best)
        hi = max(running_best)
        span = hi - lo if hi != lo else 1.0

        # If chart would be too wide, sample evenly
        max_width = 60
        if len(running_best) > max_width:
            step = len(running_best) / max_width
            sampled = [running_best[int(i * step)] for i in range(max_width)]
        else:
            sampled = running_best

        bars = ""
        for v in sampled:
            level = int((v - lo) / span * 7.999)
            level = max(0, min(8, level))
            bars += _BLOCKS[level]

        latest = self._scores[-1]
        best = max(self._scores)
        n = len(self._scores)

        text = (
            f"[bold]Best score over time[/bold]\n"
            f"[green]{bars}[/green]\n"
            f"Best: [bold]{best:+.4f}[/]  |  "
            f"Latest: {latest:+.4f}  |  "
            f"{n} iterations"
        )
        self.update(text)
