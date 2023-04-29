#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "GraphInf/rv.hpp"
#include "GraphInf/generators.h"
#include "GraphInf/python/rv.hpp"

#include "utility/init_utility.h"
#include "init_exceptions.h"
#include "init_generator.h"
#include "init_rng.h"
// #include "random_graph/init.h"
// #include "data/init.h"
// #include "mcmc/init.h"

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

    // py::module random_graph = m.def_submodule("random_graph");
    // initRandomGraph(random_graph);

    // py::module data = m.def_submodule("data");
    // initDataModels(data);

    // py::module mcmc = m.def_submodule("mcmc");
    // initMCMC(mcmc);
}
