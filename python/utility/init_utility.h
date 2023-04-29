#ifndef GRAPH_INF_PYWRAPPER_INIT_UTILITY_H
#define GRAPH_INF_PYWRAPPER_INIT_UTILITY_H

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "GraphInf/utility/maps.hpp"
#include "init_maps.h"
#include "init_functions.h"
#include "init_integerpartition.h"
// #include "init_distance.h"

namespace py = pybind11;
using namespace GraphInf;

void initUtility(py::module &m)
{
    initMaps(m);
    initFunctions(m);
    initIntegerPartition(m);
    // initDistances(m);
}

#endif
