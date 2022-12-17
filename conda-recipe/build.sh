#!/usr/bin/env
cd /Users/charlesmurphy/Documents/ulaval/doctorat/projects/codes/graphinf/graphinf_cpp/SamplableSet/src/
mkdir -p build
cd build
cmake ..
make
cd ../../../..
cd /Users/charlesmurphy/Documents/ulaval/doctorat/projects/codes/graphinf/graphinf_cpp/base_graph/
mkdir -p build
cd build
cmake ..
make
cd ../../..
$PYTHON setup.py install