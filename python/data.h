#ifndef GRAPH_INF_PYWRAPPER_DATA_INIT_H
#define GRAPH_INF_PYWRAPPER_DATA_INIT_H

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "GraphInf/data/python/data_model.hpp"
#include "GraphInf/data/data_model.h"
#include "GraphInf/data/dynamics/dynamics.h"
#include "GraphInf/data/dynamics/binary_dynamics.h"
#include "GraphInf/data/dynamics/degree.h"
#include "GraphInf/data/dynamics/glauber.h"
#include "GraphInf/data/dynamics/cowan.h"
#include "GraphInf/data/dynamics/sis.h"

namespace py = pybind11;
namespace GraphInf
{

    void initDataModels(py::module &m)
    {
        py::class_<DataModel, NestedRandomVariable, PyDataModel<>>(m, "DataModel")
            .def(py::init<RandomGraph &>(), py::arg("graph_prior"))
            .def("get_size", &DataModel::getSize)
            .def("get_graph", &DataModel::getGraph)
            .def("set_graph", &DataModel::setGraph, py::arg("graph"))
            .def("get_graph_prior", &DataModel::getGraphPrior, py::return_value_policy::reference_internal)
            .def("set_graph_prior", &DataModel::setGraphPrior, py::arg("prior"))
            .def("sample_prior", &DataModel::samplePrior)
            .def("get_log_likelihood", &DataModel::getLogLikelihood)
            .def("get_log_prior", &DataModel::getLogPrior)
            .def("get_log_joint", &DataModel::getLogJoint)
            .def("get_log_likelihood_ratio_from_graph_move", &DataModel::getLogLikelihoodRatioFromGraphMove,
                 py::arg("move"))
            .def("get_log_prior_ratio_from_graph_move", &DataModel::getLogPriorRatioFromGraphMove,
                 py::arg("move"))
            .def("get_log_joint_ratio_from_graph_move", &DataModel::getLogJointRatioFromGraphMove,
                 py::arg("move"))
            .def("apply_graph_move", &DataModel::applyGraphMove,
                 py::arg("move"));

        py::module dynamics = m.def_submodule("dynamics");
        py::class_<Dynamics, DataModel, PyDynamics<>>(dynamics, "Dynamics")
            .def(py::init<RandomGraph &, size_t, size_t>(),
                 py::arg("graph_prior"),
                 py::arg("num_states"),
                 py::arg("length"))
            .def(
                "sample_state", [](Dynamics &self, const State &initial, bool asyncMode = false, size_t initialBurn = 0)
                { self.sampleState(initial, asyncMode, initialBurn); },
                py::arg("initial"), py::arg("async_mode") = false, py::arg("initial_burn") = 0)
            .def(
                "sample_state", [](Dynamics &self, bool asyncMode = false, size_t initialBurn = 0)
                { self.sampleState({}, asyncMode, initialBurn); },
                py::arg("async_mode") = false, py::arg("initial_burn") = 0)
            .def(
                "sample", [](Dynamics &self, const State &initial, bool asyncMode = false, size_t initialBurn = 0)
                { self.sample(initial, asyncMode, initialBurn); },
                py::arg("initial"), py::arg("async_mode") = false, py::arg("initial_burn") = 0)
            .def(
                "sample", [](Dynamics &self, bool asyncMode = false, size_t initialBurn = 0)
                { self.sample({}, asyncMode, initialBurn); },
                py::arg("async_mode") = false, py::arg("initial_burn") = 0)
            .def("get_state", &Dynamics::getState)
            .def("set_state", &Dynamics::setState, py::arg("state"))
            .def("get_neighbors_state", &Dynamics::getNeighborsState)
            .def("get_past_states", &Dynamics::getPastStates)
            .def("get_past_neighbors_states", &Dynamics::getNeighborsPastStates)
            .def("get_future_states", &Dynamics::getFutureStates)
            .def("get_num_states", &Dynamics::getNumStates)
            .def("get_length", &Dynamics::getLength)
            .def("set_length", &Dynamics::setLength)
            .def("get_past_length", &Dynamics::getPastLength)
            .def("set_past_length", &Dynamics::setPastLength)
            .def("get_random_state", &Dynamics::getRandomState)
            .def("accept_selfloops", [](Dynamics &self)
                 { return self.acceptSelfLoops(); })
            .def(
                "accept_selfloops", [](Dynamics &self, bool condition)
                { self.acceptSelfLoops(condition); },
                py::arg("condition"))
            .def("sync_update_state", &Dynamics::syncUpdateState)
            .def("async_update_state", &Dynamics::asyncUpdateState,
                 py::arg("num_updates") = 1)
            .def("get_transition_prob", &Dynamics::getTransitionProb,
                 py::arg("prev_vertex_state"), py::arg("next_vertex_state"), py::arg("neighbor_state"))
            .def(
                "get_transition_probs",
                [](const Dynamics &self, BaseGraph::VertexIndex vertex)
                {
                    return self.getTransitionProbs(vertex);
                },
                py::arg("vertex"));

        py::class_<BinaryDynamics, Dynamics, PyBinaryDynamics<>>(dynamics, "BinaryDynamics")
            .def(py::init<RandomGraph &, size_t, double, double>(),
                 py::arg("graph_prior"), py::arg("length"),
                 py::arg("auto_activation_prob") = 0., py::arg("auto_deactivation_prob") = 0.)
            .def("get_activation_prob", &BinaryDynamics::getActivationProb, py::arg("neighbor_state"))
            .def("get_deactivation_prob", &BinaryDynamics::getDeactivationProb, py::arg("neighbor_state"))
            .def("set_auto_activation_prob", &BinaryDynamics::setAutoActivationProb, py::arg("auto_activation_prob"))
            .def("set_auto_deactivation_prob", &BinaryDynamics::setAutoDeactivationProb, py::arg("auto_deactivation_prob"))
            .def("get_auto_activation_prob", &BinaryDynamics::getAutoActivationProb)
            .def("get_auto_deactivation_prob", &BinaryDynamics::getAutoDeactivationProb)
            .def("get_random_state", [](const BinaryDynamics &self)
                 { return self.getRandomState(); })
            .def(
                "get_random_state", [](const BinaryDynamics &self, int initial)
                { return self.getRandomState(initial); },
                py::arg("initial_active"))
            .def(
                "sample_state", [](BinaryDynamics &self, bool asyncMode = false, size_t initialBurn = 0)
                { self.sampleState({}, asyncMode, initialBurn); },
                py::arg("async_mode") = false, py::arg("initial_burn") = 0)
            .def(
                "sample_state", [](BinaryDynamics &self, int initialActives, bool asyncMode = false, size_t initialBurn = 0)
                { self.sampleState(self.getRandomState(initialActives), asyncMode, initialBurn); },
                py::arg("initial_actives"), py::arg("async_mode") = false, py::arg("initial_burn") = 0)
            .def(
                "sample_state", [](BinaryDynamics &self, const State &initial, bool asyncMode = false, size_t initialBurn = 0)
                { self.sampleState(initial, asyncMode, initialBurn); },
                py::arg("initial"), py::arg("async_mode") = false, py::arg("initial_burn") = 0)
            .def(
                "sample", [](BinaryDynamics &self, bool asyncMode = false, size_t initialBurn = 0)
                { self.sample({}, asyncMode, initialBurn); },
                py::arg("async_mode") = false, py::arg("initial_burn") = 0)
            .def(
                "sample", [](BinaryDynamics &self, const State &initial, bool asyncMode = false, size_t initialBurn = 0)
                { self.sample(initial, asyncMode, initialBurn); },
                py::arg("initial"), py::arg("async_mode") = false, py::arg("initial_burn") = 0)
            .def(
                "sample", [](BinaryDynamics &self, int initialActives, bool asyncMode = false, size_t initialBurn = 0)
                { self.sample(self.getRandomState(initialActives), asyncMode, initialBurn); },
                py::arg("initial_actives"), py::arg("async_mode") = false, py::arg("initial_burn") = 0);

        py::class_<CowanDynamics, BinaryDynamics>(dynamics, "CowanDynamics")
            .def(py::init<RandomGraph &, size_t, double, double, double, double, double, double>(),
                 py::arg("graph_prior"), py::arg("length"), py::arg("nu") = 1,
                 py::arg("a") = 1, py::arg("mu") = 1, py::arg("eta") = 0.5,
                 py::arg("auto_activation_prob") = 1e-6, py::arg("auto_deactivation_prob") = 0.)
            .def("get_a", &CowanDynamics::getA)
            .def("set_a", &CowanDynamics::setA, py::arg("a"))
            .def("get_nu", &CowanDynamics::getNu)
            .def("set_nu", &CowanDynamics::setNu, py::arg("nu"))
            .def("get_mu", &CowanDynamics::getMu)
            .def("set_mu", &CowanDynamics::setMu, py::arg("mu"))
            .def("get_eta", &CowanDynamics::getEta)
            .def("set_eta", &CowanDynamics::setEta, py::arg("eta"));

        py::class_<DegreeDynamics, BinaryDynamics>(dynamics, "DegreeDynamics")
            .def(py::init<RandomGraph &, size_t, double>(),
                 py::arg("graph_prior"), py::arg("length"), py::arg("C"))
            .def("get_c", &DegreeDynamics::getC)
            .def("set_c", &DegreeDynamics::setC, py::arg("c"));

        py::class_<GlauberDynamics, BinaryDynamics>(dynamics, "GlauberDynamics")
            .def(py::init<RandomGraph &, int, float, float, float>(),
                 py::arg("random_graph"), py::arg("length"), py::arg("coupling") = 1,
                 py::arg("auto_activation_prob") = 0., py::arg("auto_deactivation_prob") = 0.)
            .def("get_coupling", &GlauberDynamics::getCoupling)
            .def("set_coupling", &GlauberDynamics::setCoupling, py::arg("coupling"));

        py::class_<SISDynamics, BinaryDynamics>(dynamics, "SISDynamics")
            .def(py::init<RandomGraph &, size_t, double, double, double, double>(),
                 py::arg("random_graph"), py::arg("length"), py::arg("infection_prob") = 0.5, py::arg("recovery_prob") = 0.5,
                 py::arg("auto_activation_prob") = 1e-6, py::arg("auto_deactivation_prob") = 0.)
            .def("get_infection_prob", &SISDynamics::getInfectionProb)
            .def("set_infection_prob", &SISDynamics::setInfectionProb, py::arg("infection_prob"))
            .def("get_recovery_prob", &SISDynamics::getRecoveryProb)
            .def("set_recovery_prob", &SISDynamics::setRecoveryProb, py::arg("recovery_prob"));
    }

}

#endif
