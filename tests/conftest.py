"""Pytest configuration for ensuring native extensions are available."""

from __future__ import annotations

import os
import subprocess
import sys
from importlib.util import find_spec
from pathlib import Path
from typing import Iterable


def pytest_sessionstart(session):  # type: ignore[override]
    """Ensure the compiled native modules are present before running tests."""

    if os.environ.get("MOTOR_MODEL_SKIP_NATIVE_BUILD"):
        return

    missing = tuple(
        module_name
        for module_name in _NATIVE_MODULES
        if find_spec(module_name) is None
    )

    if not missing:
        return

    project_root = Path(__file__).resolve().parent.parent
    _ensure_native_extensions(project_root)

    still_missing = tuple(
        module_name
        for module_name in missing
        if find_spec(module_name) is None
    )
    if still_missing:
        missing_list = ", ".join(still_missing)
        raise RuntimeError(
            "Unable to locate the compiled native modules after building:"
            f" {missing_list}"
        )


_NATIVE_MODULES: Iterable[str] = (
    "motor_model._native.test_module",
    "motor_model._native.continuous_mpc",
    "motor_model._native.brushed_motor",
)


def _ensure_native_extensions(project_root: Path) -> None:
    install_cmd = [sys.executable, "-m", "pip", "install", "-e", str(project_root)]
    subprocess.run(install_cmd, cwd=project_root, check=True)

