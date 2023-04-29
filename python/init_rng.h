#ifndef GRAPH_INF_PYWRAPPER_INIT_RNG_H
#define GRAPH_INF_PYWRAPPER_INIT_RNG_H

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "GraphInf/rng.h"

namespace py = pybind11;
using namespace GraphInf;
void initRNG(py::module &m)
{
    m.def("seed", &seed, py::arg("n"));
    m.def("seedWithTime", &seedWithTime);
}

#endif
