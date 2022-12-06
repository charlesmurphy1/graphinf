#ifndef GRAPH_INF_PYWRAPPER_INIT_ACTIONS_H
#define GRAPH_INF_PYWRAPPER_INIT_ACTIONS_H

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "BaseGraph/types.h"
#include "GraphInf/mcmc/callbacks/callback.hpp"
#include "GraphInf/mcmc/callbacks/action.h"
#include "GraphInf/mcmc/mcmc.h"
// #include "GraphInf/utility/distance.h"

namespace py = pybind11;
namespace GraphInf{

void initActions(py::module&m){
    py::class_<CheckConsistencyOnSweep, CallBack<MCMC>>(m, "CheckConsistencyOnSweep")
        .def(py::init<>())
        ;

    py::class_<CheckSafetyOnSweep, CallBack<MCMC>>(m, "CheckSafetyOnSweep")
        .def(py::init<>())
        ;
}

}
#endif
