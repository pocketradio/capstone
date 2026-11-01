"""Resolve notebook artifact directories across local and Kaggle sessions."""

from __future__ import annotations

from pathlib import Path
import os


def resolve_artifact_dir(required_relative: str | Path | None = None) -> Path:
    """Find a writable local artifact root or a mounted Kaggle artifact input."""
    candidates: list[Path] = []
    configured = os.environ.get("IEEE_CIS_ARTIFACT_DIR")
    if configured:
        candidates.append(Path(configured))
    candidates.extend([Path("artifacts"), Path("/kaggle/working/artifacts")])
    kaggle_inputs = Path("/kaggle/input")
    if kaggle_inputs.exists():
        for child in kaggle_inputs.iterdir():
            candidates.extend([child, child / "artifacts"])
    for base in candidates:
        if base.exists() and (required_relative is None or (base / required_relative).exists()):
            return base
    # Kaggle notebook outputs can be nested below /kaggle/input/notebooks/
    # or /kaggle/input/competitions/, so inspect exact relative-path suffixes.
    if required_relative is not None and kaggle_inputs.exists():
        relative_parts = Path(required_relative).parts
        for match in kaggle_inputs.rglob(relative_parts[-1]):
            if tuple(match.parts[-len(relative_parts):]) == relative_parts:
                return match.parents[len(relative_parts) - 1]
    if required_relative is not None:
        raise FileNotFoundError(
            f"Could not find an artifact root containing {required_relative}. "
            "Attach the previous Kaggle notebook output as an input or set IEEE_CIS_ARTIFACT_DIR."
        )
    # The default output location in Kaggle is writable; local runs use ./artifacts.
    return Path("/kaggle/working/artifacts") if Path("/kaggle/working").exists() else Path("artifacts")

