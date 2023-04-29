#ifndef GRAPH_INF_PYWRAPPER_INIT_INTEGERPARTITION_H
#define GRAPH_INF_PYWRAPPER_INIT_INTEGERPARTITION_H

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "GraphInf/utility/integer_partition.h"

namespace py = pybind11;
using namespace GraphInf;

void initIntegerPartition(py::module &m)
{
    m.def("q_rec", &q_rec, py::arg("n"), py::arg("k"));
    m.def("log_q_approx", &log_q_approx, py::arg("n"), py::arg("k"));
    m.def("log_q_approx_big", &log_q_approx_big, py::arg("n"), py::arg("k"));
    m.def("log_q_approx_small", &log_q_approx_small, py::arg("n"), py::arg("k"));
}

#endif
