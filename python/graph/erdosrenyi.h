#ifndef GRAPH_INF_PYWRAPPER_INIT_ERDOSRENYI_H
#define GRAPH_INF_PYWRAPPER_INIT_ERDOSRENYI_H

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "GraphInf/graph/erdosrenyi.h"

namespace py = pybind11;
namespace GraphInf
{

    void initErdosRenyi(py::module &m)
    {
        py::class_<ErdosRenyiModelBase, RandomGraph>(m, "ErdosRenyiModelBase");
        py::class_<ErdosRenyiModel, ErdosRenyiModelBase>(m, "ErdosRenyiModel")
            .def(
                py::init<size_t, double, bool, bool, bool>(),
                py::arg("size"),
                py::arg("edge_count"),
                py::arg("canonical") = false,
                py::arg("with_self_loops") = true,
                py::arg("with_parallel_edges") = true);
    }

}

#endif
