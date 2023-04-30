#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "GraphInf/rv.hpp"
#include "GraphInf/generators.h"
#include "GraphInf/python/rv.hpp"

#include "utility/utility.h"
#include "exceptions.h"
#include "generator.h"
#include "rng.h"
#include "graph/graph.h"
#include "data.h"

namespace py = pybind11;
PYBIND11_MODULE(_graphinf, m)
{
    m.import("basegraph");

    py::module utility = m.def_submodule("utility");
    initUtility(utility);
    initGenerators(utility);
    initRNG(utility);
    initExceptions(utility);

    py::class_<NestedRandomVariable, PyNestedRandomVariable<>>(m, "NestedRandomVariable")
        .def(py::init<>())
        .def("is_root", [&](const NestedRandomVariable &self)
             { return self.isRoot(); })
        .def("is_processed", [&](const NestedRandomVariable &self)
             { return self.isProcessed(); })
        .def("check_consistency", &NestedRandomVariable::checkConsistency)
        .def("check_safety", &NestedRandomVariable::checkSafety)
        .def("is_safe", &NestedRandomVariable::isSafe);

    py::module graph = m.def_submodule("graph");
    initRandomGraph(graph);

    py::module data = m.def_submodule("data");
    initDataModels(data);
}
