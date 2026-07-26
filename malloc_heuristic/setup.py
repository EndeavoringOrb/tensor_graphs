import pybind11
from setuptools import setup, Extension

ext_modules = [
    Extension(
        "malloc_rl",
        ["malloc_env.cpp"],
        include_dirs=[pybind11.get_include()],
        language="c++",
        extra_compile_args=["-O3", "-std=c++17"],
    ),
]

setup(
    name="malloc_rl",
    ext_modules=ext_modules,
)
