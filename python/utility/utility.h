#ifndef GRAPH_INF_PYWRAPPER_INIT_UTILITY_H
#define GRAPH_INF_PYWRAPPER_INIT_UTILITY_H

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "GraphInf/utility/maps.hpp"
#include "GraphInf/mcmc.h"
#include "maps.h"
#include "functions.h"
#include "integerpartition.h"
#include "mcmc.h"

namespace py = pybind11;
using namespace GraphInf;

void initUtility(py::module &m)
{

    initMaps(m);
    initFunctions(m);
    initIntegerPartition(m);
    initMoveTypes(m);
    initStepResult(m);
    initMCMCSummary(m);
}

#endif
