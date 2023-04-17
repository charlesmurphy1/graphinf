#ifndef GRAPH_INF_PYWRAPPER_INIT_DISTANCE_H
#define GRAPH_INF_PYWRAPPER_INIT_DISTANCE_H

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "GraphInf/utility/distance.h"
#include "GraphInf/utility/python/distance.hpp"

namespace py = pybind11;
namespace GraphInf{

void initDistances(py::module& m){
    py::class_< GraphDistance, PyGraphDistance<> >(m, "GraphDistance")
        .def(py::init<>())
        .def("compute", &GraphDistance::compute, py::arg("g1"), py::arg("g2"))
        ;
    py::class_< HammingDistance, GraphDistance >(m, "HammingDistance")
        .def(py::init<>())
        ;
}

}

#endif
