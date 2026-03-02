from setuptools import setup, Extension
from pybind11.setup_helpers import Pybind11Extension, build_ext

ext_modules = [
    Pybind11Extension(
        "tg_cpp",
        ["tensor_graphs_cpp/bindings.cpp"],
        include_dirs=["tensor_graphs_cpp", "tensor_graphs_cpp/core"],
        cxx_std=17,
    ),
]

setup(
    name="tensor_graphs",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)