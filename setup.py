import os
import sys
import setuptools
import pathlib

from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext
from setuptools.config import read_configuration

import importlib

if importlib.util.find_spec("pybind11") is None:
    from setuptools.command.build_ext import build_ext
    from setuptools import Extension
else:
    from pybind11.setup_helpers import (
        ParallelCompile,
        naive_recompile,
        Pybind11Extension,
        build_ext,
    )

    Extension = Pybind11Extension
    ParallelCompile("NPY_NUM_BUILD_JOBS", needs_recompile=naive_recompile).install()


class get_pybind_include(object):
    """Helper class to determine the pybind11 include path

    The purpose of this class is to postpone importing pybind11
    until it is actually installed, so that the ``get_include()``
    method can be invoked."""

    def __init__(self, user=False):
        self.user = user

    def __str__(self):
        import pybind11

        return pybind11.get_include(self.user)


def find_compiled_basegraph(build_path):
    if not os.path.isdir(build_path) or os.listdir(build_path) == []:
        raise RuntimeError("Submodule BaseGraph was not compiled.")
    lib_path = None
    for extension in [".a"]:
        if os.path.isfile(os.path.join(build_path, "libBaseGraph" + extension)):
            lib_path = os.path.join(build_path, "libBaseGraph" + extension)
            break

    if lib_path is None:
        raise RuntimeError(
            f'Could not find libBaseGraph in "{build_path}".'
            + "Verify that the library is compiled."
        )
    return lib_path


def find_compiled_SamplableSet(build_path):
    if not os.path.isdir(build_path) or os.listdir(build_path) == []:
        raise RuntimeError("Submodule SamplableSet was not compiled.")
    lib_path = os.path.join(build_path, "libsamplableset.a")
    return lib_path


def find_files_recursively(path, ext=[]):
    if isinstance(ext, str):
        ext = [ext]
    elif not isinstance(ext, list):
        raise TypeError(
            f"type `{type(ext)}` for extension is incorrect,"
            + "expect `str` or `list`."
        )
    file_list = []

    for e in ext:
        for root, subdirs, files in os.walk(path):
            for f in files:
                if f.split(".")[-1] == e:
                    file_list.append(os.path.join(root, f))
    return file_list


ext_modules = [
    Extension(
        "_graphinf",
        include_dirs=[
            get_pybind_include(),
            get_pybind_include(user=True),
            "_graphinf/include",
            "_graphinf/base_graph/include",
            "_graphinf/SamplableSet/src",
        ],
        sources=[
            "_graphinf/src/rng.cpp",
            "_graphinf/src/exceptions.cpp",
            "_graphinf/src/generators.cpp",
            *find_files_recursively("_graphinf/src/utility", "cpp"),
            *find_files_recursively("_graphinf/src/random_graph", "cpp"),
            *find_files_recursively("_graphinf/src/data", "cpp"),
            *find_files_recursively("_graphinf/src/mcmc", "cpp"),
            "_graphinf/pybind_wrapper/pybind_main.cpp",
        ],
        language="c++",
        extra_objects=[
            find_compiled_basegraph("_graphinf/base_graph/build"),
            find_compiled_SamplableSet("_graphinf/SamplableSet/src/build"),
        ],
    ),
]


# As of Python 3.6, C Compiler has a `has_flag` method.
# cf http://bugs.python.org/issue26689
def has_flag(compiler, flagname):
    """Return a boolean indicating whether a flag name is supported on
    the specified compiler.
    """
    import tempfile

    with tempfile.NamedTemporaryFile("w", suffix=".cpp") as f:
        f.write("int main (int argc, char **argv) { return 0; }")
        try:
            compiler.compile([f.name], extra_postargs=[flagname])
        except setuptools.distutils.errors.CompileError:
            return False
    return True


def cpp_flag(compiler):
    """Return the -std=c++[11/14/17] compiler flag.

    The newer version is prefered over c++11 (when it is available).
    """
    flags = ["-std=c++17", "-std=c++11"]

    for flag in flags:
        if has_flag(compiler, flag):
            return flag

    raise RuntimeError("Unsupported compiler -- at least C++11 support " "is needed!")


class BuildExt(build_ext):
    """A custom build extension for adding compiler-specific options."""

    c_opts = {
        "msvc": ["/EHsc"],
        "unix": [],
    }
    l_opts = {
        "msvc": [],
        "unix": [],
    }

    if sys.platform == "darwin":
        darwin_opts = ["-stdlib=libc++", "-mmacosx-version-min=10.7"]
        c_opts["unix"] += darwin_opts
        l_opts["unix"] += darwin_opts

    def build_extensions(self):
        ct = self.compiler.compiler_type
        opts = self.c_opts.get(ct, [])
        link_opts = self.l_opts.get(ct, [])
        if ct == "unix":
            opts.append('-DVERSION_INFO="%s"' % self.distribution.get_version())
            opts.append(cpp_flag(self.compiler))
            if has_flag(self.compiler, "-fvisibility=hidden"):
                opts.append("-fvisibility=hidden")
        elif ct == "msvc":
            opts.append('/DVERSION_INFO=\\"%s\\"' % self.distribution.get_version())
        for ext in self.extensions:
            ext.extra_compile_args = opts
            ext.extra_link_args = link_opts
        build_ext.build_extensions(self)


description = "Package for the analysis of stochastic processes on random graphs."


def compile_from_cmake(path_to_cmake):
    current_dir = os.getcwd()
    p = pathlib.Path(path_to_cmake)
    if not os.path.isdir(p / "build"):
        os.mkdir(p / "build")
    os.chdir(p / "build")
    os.system("cmake ..; make clean; make -j4")
    os.chdir(current_dir)


def install_from_setup(path_to_setup):
    current_dir = os.getcwd()
    p = pathlib.Path(path_to_setup)
    os.chdir(p)
    os.system("pip install .")
    os.chdir(current_dir)


try:
    find_compiled_SamplableSet("_graphinf/SamplableSet/src/build")
except RuntimeError:
    print("Compiling SamplableSet")
    compile_from_cmake("_graphinf/SamplableSet/src")
if importlib.util.find_spec("SamplableSet") is None:
    print("Installing SamplableSet")
    install_from_setup("_graphinf/SamplableSet")

try:
    find_compiled_basegraph("_graphinf/base_graph/build")
except RuntimeError:
    print("Compiling BaseGraph")
    compile_from_cmake("_graphinf/base_graph")
if importlib.util.find_spec("basegraph") is None:
    print("Installing BaseGraph")
    install_from_setup("_graphinf/base_graph")

setup(
    name="graphinf",
    version=0.1,
    author="Charles Murphy",
    author_email="charles.murphy.1@ulaval.ca",
    url="https://github.com/charlesmurphy1/graphinf",
    license="MIT",
    description=description,
    packages=find_packages(),
    install_requires=[
        "pybind11>=2.3",
        "numpy>=1.20.3",
        "scipy>=1.7.1",
        "psutil>=5.8.0",
    ],
    python_requires=">=3.6",
    zip_safe=False,
    ext_modules=ext_modules,
    include_package_data=True,
    cmdclass={"build_ext": BuildExt},
)
