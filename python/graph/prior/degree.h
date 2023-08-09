#ifndef GRAPH_INF_PYWRAPPER_PRIOR_INIT_DEGREE_H
#define GRAPH_INF_PYWRAPPER_PRIOR_INIT_DEGREE_H

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>

#include "GraphInf/graph/prior/python/prior.hpp"
#include "GraphInf/graph/prior/python/degree.hpp"
#include "GraphInf/graph/prior/edge_count.h"
#include "GraphInf/graph/prior/degree.h"

namespace py = pybind11;
namespace GraphInf
{

    void initDegreePrior(py::module &m)
    {
        py::class_<DegreePrior, Prior<std::vector<size_t>>, PyDegreePrior<>>(m, "DegreePrior")
            .def(py::init<size_t>(), py::arg("size"))
            .def(py::init<size_t, EdgeCountPrior &>(), py::arg("size"), py::arg("edge_count_prior"))
            .def("size", &DegreePrior::getSize)
            .def("edge_count", &DegreePrior::getEdgeCount)
            .def("degree_of_idx", &DegreePrior::getDegree)
            .def("degree_counts", &DegreePrior::getDegreeCounts, py::return_value_policy::reference_internal)
            .def("edge_count_prior", &DegreePrior::getEdgeCountPrior, py::return_value_policy::reference_internal)
            .def("set_edge_count_prior", &DegreePrior::setEdgeCountPrior, py::arg("edge_count_prior"));

        py::class_<DegreeDeltaPrior, DegreePrior>(m, "DegreeDeltaPrior")
            .def(py::init<const DegreeSequence &>(), py::arg("degrees"));

        py::class_<DegreeUniformPrior, DegreePrior>(m, "DegreeUniformPrior")
            .def(py::init<size_t>(), py::arg("size"))
            .def(py::init<size_t, EdgeCountPrior &>(), py::arg("size"), py::arg("edge_count_prior"));

        py::class_<DegreeUniformHyperPrior, DegreePrior>(m, "DegreeUniformHyperPrior")
            .def(py::init<size_t, bool>(), py::arg("size"), py::arg("exact") = false)
            .def(py::init<size_t, EdgeCountPrior &, bool>(), py::arg("size"), py::arg("edge_count_prior"), py::arg("exact") = false);
    }

}

#endif
