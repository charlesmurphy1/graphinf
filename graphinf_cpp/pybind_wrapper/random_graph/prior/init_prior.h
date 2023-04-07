#ifndef GRAPH_INF_PYWRAPPER_INIT_PRIOR_H
#define GRAPH_INF_PYWRAPPER_INIT_PRIOR_H

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>

#include "GraphInf/rv.hpp"
#include "GraphInf/random_graph/prior/prior.hpp"
#include "GraphInf/random_graph/prior/python/prior.hpp"


namespace py = pybind11;
namespace GraphInf{


template <typename StateType>
py::class_<Prior<StateType>, NestedRandomVariable, PyPrior<StateType>> declarePriorBaseClass(py::module& m, std::string pyName){
    return py::class_<Prior<StateType>, NestedRandomVariable, PyPrior<StateType>>(m, pyName.c_str())
        .def(py::init<>())
        .def("get_state", &Prior<StateType>::getState)
        .def("set_state", &Prior<StateType>::setState, py::arg("state"))
        .def("sample_state", &Prior<StateType>::sampleState)
        .def("sample_priors", &Prior<StateType>::samplePriors)
        .def("sample", &Prior<StateType>::sample)
        .def("get_log_likelihood", &Prior<StateType>::getLogLikelihood)
        .def("get_log_prior", &Prior<StateType>::getLogPrior)
        .def("get_log_joint", &Prior<StateType>::getLogJoint)
        .def("get_log_joint_ratio_from_graph_move", &Prior<StateType>::getLogJointRatioFromGraphMove, py::arg("move"))
        .def("apply_graph_move", &Prior<StateType>::applyGraphMove, py::arg("move"))
        ;
}

template <typename StateType, typename Label>
py::class_<VertexLabeledPrior<StateType, Label>, Prior<StateType>, PyVertexLabeledPrior<StateType, Label>>
declareVertexLabeledPriorBaseClass(py::module& m, std::string pyName){
    return py::class_<VertexLabeledPrior<StateType, Label>, Prior<StateType>, PyVertexLabeledPrior<StateType, Label>>(m, pyName.c_str())
        .def(py::init<>())
        .def("get_log_joint_ratio_from_label_move", &VertexLabeledPrior<StateType, Label>::getLogJointRatioFromLabelMove, py::arg("move"))
        .def("apply_label_move", &VertexLabeledPrior<StateType, Label>::applyGraphMove, py::arg("move"))
        ;
}

void initPriorBaseClass(pybind11::module& m){
    declarePriorBaseClass<size_t>(m, "UnIntPrior");
    declarePriorBaseClass<int>(m, "IntPrior");
    declareVertexLabeledPriorBaseClass<size_t, BlockIndex>(m, "UnIntVertexLabeledPrior");
    declareVertexLabeledPriorBaseClass<int, BlockIndex>(m, "IntVertexLabeledPrior");

    declarePriorBaseClass<std::vector<size_t>>(m, "UnIntVectorPrior");
    declarePriorBaseClass<std::vector<int>>(m, "IntVectorPrior");
    declareVertexLabeledPriorBaseClass<std::vector<size_t>, BlockIndex>(m, "UnIntVectorVertexLabeledPrior");
    declareVertexLabeledPriorBaseClass<std::vector<int>, BlockIndex>(m, "IntVectorVertexLabeledPrior");

    declarePriorBaseClass<MultiGraph>(m, "MultigraphPrior");
    declareVertexLabeledPriorBaseClass<MultiGraph, BlockIndex>(m, "MultigraphVertexLabeledPrior");
}

}

#endif