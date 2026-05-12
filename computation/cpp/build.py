import os
import sys
import pybind11
from setuptools import setup, Extension

if len(sys.argv) == 1:
    sys.argv.extend(["build_ext", "--inplace"])

current_dir = os.path.dirname(os.path.abspath(__file__))
computation_dir = os.path.dirname(current_dir)
root_dir = os.path.dirname(computation_dir)

# Define the C++ etension
ext_modules = [
    Extension(
        "backtest_engine",
        [os.path.join(current_dir, "backtest.cpp")],
        include_dirs=[pybind11.get_include(), root_dir],
        language="c++",
        extra_compile_args=["-O3", "-std=c++17"],
    ),
]

setup(
    name="backtest_engine",
    ext_modules=ext_modules,
    packages=[],
)
