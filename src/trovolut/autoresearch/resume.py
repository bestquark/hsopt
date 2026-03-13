"""Auto-resume wrapper for long-running autoresearch optimization on Slurm clusters.

Ported from tebd/core/evolve_resume.py. Integrates with NVIDIA cluster's
AutoResume (via submit_job) so that optimization runs can span multiple
time-limited jobs transparently.

Usage:
    from trovolut.autoresearch.resume import run_evolution_with_resume
    result = run_evolution_with_resume(
        initial_program=initial_program,
        evaluator=evaluator,
        config=config,
        output_dir=output_dir,
    )
"""

from __future__ import annotations

import asyncio
import glob
import os
import signal
import sys
from pathlib import Path
from typing import Optional, Union

# ---------------------------------------------------------------------------
# AutoResume integration (graceful no-op when not on the cluster)
# ---------------------------------------------------------------------------

_auto_resume_available = False
_termination_requested = False

try:
    sys.path.append(os.environ.get("SUBMIT_SCRIPTS", "."))
    from userlib.auto_resume import AutoResume  # type: ignore[import-untyped]
    _auto_resume_available = True
except ImportError:
    pass


def _init_auto_resume() -> None:
    """Initialize AutoResume timers if available."""
    global _auto_resume_available
    if _auto_resume_available:
        try:
            AutoResume.init()
            print("[auto-resume] Initialized AutoResume timers")
        except Exception as e:
            print(f"[auto-resume] AutoResume.init() failed: {e}")
            _auto_resume_available = False


def _get_resume_details() -> Optional[dict]:
    """Get resume details from a previous job, if any."""
    if not _auto_resume_available:
        return None
    try:
        details = AutoResume.get_resume_details()
        if details:
            print(f"[auto-resume] Resuming from previous job: {details}")
        return details
    except Exception:
        return None


def _request_resume(user_dict: Optional[dict] = None) -> None:
    """Request a follow-up job via AutoResume."""
    if not _auto_resume_available:
        return
    try:
        AutoResume.request_resume(user_dict=user_dict or {})
        print("[auto-resume] Requested follow-up job")
    except Exception as e:
        print(f"[auto-resume] request_resume failed: {e}")


def _termination_requested_check() -> bool:
    """Check whether AutoResume or signal handler requested termination."""
    global _termination_requested
    if _termination_requested:
        return True
    if _auto_resume_available:
        try:
            if AutoResume.termination_requested():
                _termination_requested = True
                return True
        except Exception:
            pass
    return False


def _install_signal_handler() -> None:
    """Install SIGUSR1 handler for --pre_timeout_signal graceful shutdown."""
    def _handler(signum, frame):
        global _termination_requested
        _termination_requested = True
        print(f"\n[auto-resume] Received signal {signum} -- will save and exit")

    if hasattr(signal, "SIGUSR1"):
        signal.signal(signal.SIGUSR1, _handler)
        print("[auto-resume] Installed SIGUSR1 handler for pre-timeout signal")


# ---------------------------------------------------------------------------
# Checkpoint discovery
# ---------------------------------------------------------------------------

def find_latest_checkpoint(output_dir: str) -> Optional[str]:
    """Find the latest OpenEvolve checkpoint in output_dir/checkpoints/.

    OpenEvolve saves checkpoints as: checkpoints/checkpoint_N/
    Returns the path to the latest checkpoint directory, or None.
    """
    ckpt_pattern = os.path.join(output_dir, "checkpoints", "checkpoint_*")
    ckpt_dirs = sorted(glob.glob(ckpt_pattern))
    if not ckpt_dirs:
        return None

    def _iter_num(p):
        try:
            return int(os.path.basename(p).split("_")[-1])
        except (ValueError, IndexError):
            return -1
    ckpt_dirs.sort(key=_iter_num)
    latest = ckpt_dirs[-1]
    print(f"[auto-resume] Found checkpoint: {latest}")
    return latest


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_evolution_with_resume(
    initial_program: Union[str, Path],
    evaluator: Union[str, Path],
    config,  # openevolve.config.Config
    output_dir: str,
    max_iterations: Optional[int] = None,
):
    """Run OpenEvolve with auto-resume support.

    On first invocation, starts a fresh evolution. On subsequent invocations
    (after Slurm requeue), detects the latest checkpoint and resumes.

    Args:
        initial_program: Path to the initial policy .py file.
        evaluator: Path to the evaluator .py file.
        config: OpenEvolve Config object.
        output_dir: Directory for OpenEvolve output and checkpoints.
        max_iterations: Override config.max_iterations if set.

    Returns:
        The best Program object from OpenEvolve, or None.
    """
    _init_auto_resume()
    _install_signal_handler()
    _get_resume_details()

    os.makedirs(output_dir, exist_ok=True)

    checkpoint_path = find_latest_checkpoint(output_dir)

    if checkpoint_path:
        print(f"[auto-resume] Resuming evolution from {checkpoint_path}")
    else:
        print("[auto-resume] Starting fresh evolution")

    from openevolve.main import OpenEvolve as OpenEvolveController

    controller = OpenEvolveController(
        initial_program_path=str(initial_program),
        evaluation_file=str(evaluator),
        config=config,
        output_dir=output_dir,
    )

    iterations = max_iterations or getattr(config, "max_iterations", None)

    best_program = asyncio.run(
        _run_with_termination_check(controller, iterations, checkpoint_path)
    )

    return best_program


async def _run_with_termination_check(
    controller,
    iterations: Optional[int],
    checkpoint_path: Optional[str],
):
    """Run the controller and handle termination signals."""
    best_program = None
    evolution_complete = False

    try:
        best_program = await controller.run(
            iterations=iterations,
            checkpoint_path=checkpoint_path,
        )
        evolution_complete = True
        print("[auto-resume] Evolution completed all iterations")
    except (KeyboardInterrupt, SystemExit):
        print("[auto-resume] Interrupted -- saving state")
    except Exception as e:
        if _termination_requested:
            print(f"[auto-resume] Termination during evolution: {e}")
        else:
            raise

    if not evolution_complete and _termination_requested:
        _request_resume()
        print("[auto-resume] Exiting cleanly for Slurm requeue (exit 0)")
    elif not evolution_complete:
        _request_resume()

    return best_program


def save_best_policy(best_program, output_path: str) -> None:
    """Extract and save the best policy code from an OpenEvolve result."""
    if best_program is None:
        print("[auto-resume] No best program to save")
        return
    code = getattr(best_program, "code", None)
    if code:
        with open(output_path, "w") as f:
            f.write(code)
        print(f"[auto-resume] Best policy saved to {output_path}")


__all__ = [
    "run_evolution_with_resume",
    "find_latest_checkpoint",
    "save_best_policy",
]
