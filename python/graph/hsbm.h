#ifndef GRAPH_INF_PYWRAPPER_INIT_HSBM_H
#define GRAPH_INF_PYWRAPPER_INIT_HSBM_H

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "GraphInf/graph/hsbm.h"

namespace py = pybind11;
namespace GraphInf
{

    void initNestedStochasticBlockModel(py::module &m)
    {
        py::class_<NestedStochasticBlockModelBase, NestedBlockLabeledRandomGraph>(m, "NestedStochasticBlockModelBase")
            // .def(
            //     py::init<size_t, EdgeCountPrior&, bool, bool, bool>(),
            //     py::arg("size"),
            //     py::arg("edge_count_prior"),
            //     py::arg("stub_labeled")=true,
            //     py::arg("with_self_loops")=true,
            //     py::arg("with_parallel_edges")=true
            // )
            .def("nested_label_graph_prior", &NestedStochasticBlockModelBase::getNestedLabelGraphPrior, py::return_value_policy::reference_internal);

        py::class_<NestedStochasticBlockModelFamily, NestedStochasticBlockModelBase>(m, "NestedStochasticBlockModelFamily")
            .def(
                py::init<size_t, double, bool, bool, bool, bool, std::string, std::string, double, double, double>(),
                py::arg("size"),
                py::arg("edge_count"),
                py::arg("canonical") = false,
                py::arg("stub_labeled") = true,
                py::arg("with_self_loops") = true,
                py::arg("with_parallel_edges") = true,
                py::arg("edge_proposer_type") = "uniform",
                py::arg("block_proposer_type") = "mixed",
                py::arg("sample_label_count_prob") = 0.1,
                py::arg("label_creation_prob") = 0.5,
                py::arg("shift") = 1);
    }

}

#endif
