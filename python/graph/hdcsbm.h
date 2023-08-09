#ifndef GRAPH_INF_PYWRAPPER_INIT_HDCSBM_H
#define GRAPH_INF_PYWRAPPER_INIT_HDCSBM_H

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "GraphInf/graph/hdcsbm.h"

namespace py = pybind11;
namespace GraphInf
{

    void initNestedDegreeCorrectedStochasticBlockModel(py::module &m)
    {
        py::class_<NestedDegreeCorrectedStochasticBlockModelBase, NestedBlockLabeledRandomGraph>(m, "NestedDegreeCorrectedStochasticBlockModelBase")
            // .def(
            //     py::init<size_t, EdgeCountPrior&, VertexLabeledDegreePrior&>(),
            //     py::arg("size"),
            //     py::arg("edge_count_prior"),
            //     py::arg("degree_prior")
            // )
            .def("degree_prior", &NestedDegreeCorrectedStochasticBlockModelBase::getDegreePrior, py::return_value_policy::reference_internal)
            .def("set_degree_prior", &NestedDegreeCorrectedStochasticBlockModelBase::setDegreePrior, py::arg("prior"));

        py::class_<NestedDegreeCorrectedStochasticBlockModelFamily, NestedDegreeCorrectedStochasticBlockModelBase>(m, "NestedDegreeCorrectedStochasticBlockModelFamily")
            .def(
                py::init<size_t, double, bool, bool, std::string, std::string, double, double, double>(),
                py::arg("size"),
                py::arg("edge_count"),
                py::arg("degree_hyperprior") = true,
                py::arg("canonical") = false,
                py::arg("edge_proposer_type") = "degree",
                py::arg("block_proposer_type") = "mixed",
                py::arg("sample_label_count_prob") = 0.1,
                py::arg("label_creation_prob") = 0.5,
                py::arg("shift") = 1);
    }

}

#endif
