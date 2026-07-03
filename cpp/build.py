"""Compile the C++ distance extension (``vecsearch_native``) in place.

Usage
-----
    python cpp/build.py

This invokes the platform C++ compiler with the pybind11 and Python include
paths and drops the resulting shared object next to the source so that
``import vecsearch_native`` works from the project root. We deliberately avoid a
full setuptools/CMake build to keep the stretch goal one command and zero
config — the extension is a single translation unit.
"""

from __future__ import annotations

import os
import subprocess
import sys
import sysconfig
from pathlib import Path

import pybind11


def main() -> int:
    here = Path(__file__).resolve().parent
    src = here / "distance.cpp"
    ext_suffix = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
    out = here.parent / f"vecsearch_native{ext_suffix}"

    py_include = sysconfig.get_path("include")
    pybind_include = pybind11.get_include()

    cxx = os.environ.get("CXX", "c++")
    cmd = [
        cxx,
        "-O3", "-Wall", "-shared", "-std=c++14", "-fPIC",
        f"-I{py_include}",
        f"-I{pybind_include}",
        str(src),
        "-o", str(out),
    ]
    # macOS needs these flags for a Python extension module.
    if sys.platform == "darwin":
        cmd += ["-undefined", "dynamic_lookup"]

    print("[build]", " ".join(cmd))
    result = subprocess.run(cmd)
    if result.returncode == 0:
        print(f"[build] OK -> {out.name}")
    else:
        print("[build] FAILED", file=sys.stderr)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
