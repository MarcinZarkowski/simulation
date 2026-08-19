"""Build the obt_engine extension in place."""
import os
import sys

import pybind11
from setuptools import Extension, setup

here = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(here)

if len(sys.argv) == 1:
    sys.argv.extend(["build_ext", f"--build-lib={repo_root}"])

setup(
    name="obt_engine",
    # The repo root holds several packages; this builds only the extension.
    packages=[],
    py_modules=[],
    ext_modules=[
        Extension(
            "obt_engine",
            [os.path.join(here, "bindings.cpp")],
            include_dirs=[pybind11.get_include(), here],
            language="c++",
            extra_compile_args=["-O3", "-std=c++17", "-Wall"],
        )
    ],
)
