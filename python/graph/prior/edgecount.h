#ifndef GRAPH_INF_PYWRAPPER_PRIOR_INIT_EDGECOUNT_H
#define GRAPH_INF_PYWRAPPER_PRIOR_INIT_EDGECOUNT_H

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "GraphInf/graph/prior/prior.hpp"
#include "GraphInf/graph/prior/edge_count.h"
#include "GraphInf/graph/prior/python/edgecount.hpp"

namespace py = pybind11;
namespace GraphInf
{

    void initEdgeCountPrior(py::module &m)
    {
        py::class_<EdgeCountPrior, Prior<size_t>, PyEdgeCountPrior<>>(m, "EdgeCountPrior");

        py::class_<EdgeCountDeltaPrior, EdgeCountPrior>(m, "EdgeCountDeltaPrior")
            .def(py::init<>())
            .def(py::init<size_t>(), py::arg("edge_count"));

        py::class_<EdgeCountPoissonPrior, EdgeCountPrior>(m, "EdgeCountPoissonPrior")
            .def(py::init<>())
            .def(py::init<double>(), py::arg("mean"));

        py::class_<EdgeCountExponentialPrior, EdgeCountPrior>(m, "EdgeCountExponentialPrior")
            .def(py::init<>())
            .def(py::init<double>(), py::arg("mean"));
    }

}

#endif
