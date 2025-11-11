"""Builds and executes the C++-only gtest suite for the MPC controller."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

def test_cpp_continuous_mpc_gtest(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[1]
    source_dir = project_root / "cpp_tests"
    build_dir = tmp_path / "build"
    build_dir.mkdir()

    env = os.environ.copy()

    cmake_args = [
        "cmake",
        "-S",
        str(source_dir),
        "-B",
        str(build_dir),
    ]
    subprocess.run(cmake_args, check=True, cwd=project_root, env=env)

    build_args = ["cmake", "--build", str(build_dir)]
    subprocess.run(build_args, check=True, cwd=project_root, env=env)

    ctest_args = [
        "ctest",
        "--test-dir",
        str(build_dir),
        "--output-on-failure",
    ]
    subprocess.run(ctest_args, check=True, cwd=project_root, env=env)
