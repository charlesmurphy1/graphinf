# fast-midynet
Framework for graph inference.

![Test](https://github.com/charlesmurphy1/graphinf/actions/workflows/cpp-build-and-tests.yml/badge.svg)
![Test](https://github.com/charlesmurphy1/graphinf/actions/workflows/python-build.yml/badge.svg)

## Installation

### Requirements

- pybind11
- scikit-build
- numpy
- scipy
- networkx
- pandas
- pytest

### Installation with pip

You can use pip to install the python module:
```bash
pip install .
```

Note that the `basegraph` must be installed. You can either download it and install it yourself, or you can load the submodules and install it directly from the repository:

```bash
git submodule --init
pip install ext/base_graph
```

### Build the C++ library

TO build the C++ library, we use cmake. First, we must set up the CMake environment. From the root directory (i.e., where the root file `CMakeFile.txt` is located), run the following command:

```bash
cmake -S . -B build 
```

This commands build the necessary scripts for building the library. To set up the environment for building the C++ unit tests as well, set the `BUILD_TEST` argument to `ON`:

```bash
cmake -S . -B build -DBUILD_TESTS=ON -Wno-dev
```

Finally, build the C++ library by running the following command:

```bash
cmake --build build -j4
```


