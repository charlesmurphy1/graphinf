#ifndef GRAPH_INF_PYWRAPPER_INIT_UTILITY_H
#define GRAPH_INF_PYWRAPPER_INIT_UTILITY_H

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "GraphInf/utility/maps.hpp"
#include "GraphInf/utility/mcmc.h"
#include "maps.h"
#include "functions.h"
#include "integerpartition.h"

namespace py = pybind11;
using namespace GraphInf;

void initUtility(py::module &m)
{
    py::class_<MCMCSummary>(m, "MCMCSummary")
        .def(py::init<std::string, double, bool>(), py::arg("move"), py::arg("accept_prob"), py::arg("is_accepted"))
        .def("__repr__", [](const MCMCSummary &self)
             { return "MCMCSummary(move=" + self.move + ", accept_prob=" + std::to_string(self.acceptProb) + ", accepted=" + std::to_string((int)self.isAccepted) + ")"; })
        .def_readonly("move", &MCMCSummary::move)
        .def_readonly("accept_prob", &MCMCSummary::acceptProb)
        .def_readonly("is_accepted", &MCMCSummary::isAccepted);

    initMaps(m);
    initFunctions(m);
    initIntegerPartition(m);
}

#endif
