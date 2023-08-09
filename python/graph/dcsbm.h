#ifndef GRAPH_INF_PYWRAPPER_INIT_DCSBM_H
#define GRAPH_INF_PYWRAPPER_INIT_DCSBM_H

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "GraphInf/graph/dcsbm.h"

namespace py = pybind11;
namespace GraphInf
{

    void initDegreeCorrectedStochasticBlockModel(py::module &m)
    {
        py::class_<DegreeCorrectedStochasticBlockModelBase, BlockLabeledRandomGraph>(m, "DegreeCorrectedStochasticBlockModelBase")
            .def("degree_prior", &DegreeCorrectedStochasticBlockModelBase::getDegreePrior, py::return_value_policy::reference_internal)
            .def("set_degree_prior", &DegreeCorrectedStochasticBlockModelBase::setDegreePrior, py::arg("prior"))
            .def("degree", &DegreeCorrectedStochasticBlockModelBase::getDegree, py::arg("vertex"))
            .def("degrees", &DegreeCorrectedStochasticBlockModelBase::getDegrees, py::return_value_policy::reference_internal)
            .def("degrees_copy", &DegreeCorrectedStochasticBlockModelBase::getDegrees, py::return_value_policy::copy);

        py::class_<DegreeCorrectedStochasticBlockModelFamily, DegreeCorrectedStochasticBlockModelBase>(m, "DegreeCorrectedStochasticBlockModelFamily")
            .def(
                py::init<size_t, double, size_t, bool, bool, bool, bool, std::string, std::string, double, double, double>(),
                py::arg("size"),
                py::arg("edge_count"),
                py::arg("block_count") = 0,
                py::arg("block_hyperprior") = true,
                py::arg("degree_hyperprior") = true,
                py::arg("planted") = false,
                py::arg("canonical") = false,
                py::arg("edge_proposer_type") = "degree",
                py::arg("block_proposer_type") = "mixed",
                py::arg("sample_label_count_prob") = 0.1,
                py::arg("label_creation_prob") = 0.5,
                py::arg("shift") = 1);
    }

}

#endif
