#ifndef GRAPH_INF_PYWRAPPER_DATA_INIT_H
#define GRAPH_INF_PYWRAPPER_DATA_INIT_H

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "GraphInf/data/python/data_model.h"
#include "GraphInf/data/data_model.h"
#include "GraphInf/data/dynamics/dynamics.h"
#include "init_dynamics.h"

namespace py = pybind11;
namespace GraphInf
{

    template <typename GraphPriorType>
    py::class_<DataModel<GraphPriorType>, NestedRandomVariable, PyDataModel<GraphPriorType>> declareDataModel(py::module &m, std::string pyName)
    {
        return py::class_<DataModel<GraphPriorType>, NestedRandomVariable, PyDataModel<GraphPriorType>>(m, pyName.c_str())
            .def(py::init<>())
            .def(py::init<GraphPriorType &>(), py::arg("graph_prior"))
            .def("get_size", &DataModel<GraphPriorType>::getSize)
            .def("get_graph", &DataModel<GraphPriorType>::getGraph)
            .def("set_graph", &DataModel<GraphPriorType>::setGraph, py::arg("graph"))
            .def("get_graph_prior", &DataModel<GraphPriorType>::getGraphPrior, py::return_value_policy::reference_internal)
            .def("set_graph_prior", &DataModel<GraphPriorType>::setGraphPrior, py::arg("prior"))
            .def("sample_prior", &DataModel<GraphPriorType>::samplePrior)
            .def("get_log_likelihood", &DataModel<GraphPriorType>::getLogLikelihood)
            .def("get_log_prior", &DataModel<GraphPriorType>::getLogPrior)
            .def("get_log_joint", &DataModel<GraphPriorType>::getLogJoint)
            .def("get_log_likelihood_ratio_from_graph_move", &DataModel<GraphPriorType>::getLogLikelihoodRatioFromGraphMove,
                 py::arg("move"))
            .def("get_log_prior_ratio_from_graph_move", &DataModel<GraphPriorType>::getLogPriorRatioFromGraphMove,
                 py::arg("move"))
            .def("get_log_joint_ratio_from_graph_move", &DataModel<GraphPriorType>::getLogJointRatioFromGraphMove,
                 py::arg("move"))
            .def("apply_graph_move", &DataModel<GraphPriorType>::applyGraphMove,
                 py::arg("move"));
    }

    void initDataModels(py::module &m)
    {
        declareDataModel<RandomGraph>(m, "DataModel");
        declareDataModel<BlockLabeledRandomGraph>(m, "BlockLabeledDataModel");
        declareDataModel<NestedBlockLabeledRandomGraph>(m, "NestedBlockLabeledDataModel");

        py::module dynamics = m.def_submodule("dynamics");
        initDynamics(dynamics);
    }

}

#endif