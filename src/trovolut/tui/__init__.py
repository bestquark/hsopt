"""Trovolut TUI — interactive terminal interface for common tasks."""

from trovolut.tui.app import TrovolutApp


def main():
    """Launch the Trovolut TUI."""
    app = TrovolutApp()
    app.run()


__all__ = ["TrovolutApp", "main"]
