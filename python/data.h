#ifndef GRAPH_INF_PYWRAPPER_DATA_INIT_H
#define GRAPH_INF_PYWRAPPER_DATA_INIT_H

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "GraphInf/types.h"

#include "GraphInf/data/python/proposer.hpp"
#include "GraphInf/data/python/data_model.hpp"
#include "GraphInf/data/data_model.h"

#include "GraphInf/data/dynamics/dynamics.h"
#include "GraphInf/data/dynamics/binary_dynamics.h"
#include "GraphInf/data/dynamics/degree.h"
#include "GraphInf/data/dynamics/glauber.h"
#include "GraphInf/data/dynamics/cowan.h"
#include "GraphInf/data/dynamics/sis.h"

#include "GraphInf/data/uncertain/uncertain.h"
#include "GraphInf/data/uncertain/poisson.h"

namespace py = pybind11;
namespace GraphInf
{

    void initDataModels(py::module &m)
    {
        py::class_<ParamProposer, PyParamProposer<>>(m, "ParamProposer")
            .def(py::init<>())
            .def("proposer_move", &ParamProposer::proposeMove)
            .def("log_proposal", &ParamProposer::logProposal, py::arg("move"))
            .def("log_proposal_ratio", &ParamProposer::logProposalRatio, py::arg("move"));

        py::class_<StepParamProposer, ParamProposer>(m, "StepParamProposer")
            .def(py::init<double, double>(), py::arg("step_size") = 0.01, py::arg("p") = 0.5);
        py::class_<GaussianParamProposer, ParamProposer>(m, "GaussianParamProposer")
            .def(py::init<double, double>(), py::arg("mean") = 0.0, py::arg("stddev") = 0.1);

        py::class_<ParamMove>(m, "ParamMove")
            .def(py::init<std::string, double>(), py::arg("key"), py::arg("value"))
            .def_readwrite("key", &ParamMove::key)
            .def_readwrite("value", &ParamMove::value)
            .def("__repr__", [](ParamMove &self)
                 { return self.display(); });

        py::class_<MultiParamProposer>(m, "MultiParamProposer")
            .def(py::init<double, double>(), py::arg("min_weight") = 1, py::arg("max_weight") = 10)
            .def("insert_step_proposer", &MultiParamProposer::insertStepProposer, py::arg("key"), py::arg("rate") = 1, py::arg("step_size") = 0.01, py::arg("p") = 0.5)
            .def("insert_gaussian_proposer", &MultiParamProposer::insertGaussianProposer, py::arg("key"), py::arg("rate") = 1, py::arg("mean") = 0.0, py::arg("stddev") = 0.1)
            .def("erase", &MultiParamProposer::erase, py::arg("key"))
            .def("size", &MultiParamProposer::size)
            .def("propose_move", [](MultiParamProposer &self)
                 { return self.proposeMove(); })
            .def("propose_move", [](MultiParamProposer &self, std::string key)
                 { return self.proposeMove(key); })
            .def("log_proposal_ratio", &MultiParamProposer::logProposalRatio, py::arg("move"));

        py::class_<DataModel, NestedRandomVariable, PyDataModel<>>(m, "DataModel")
            .def(py::init<RandomGraph &>(), py::arg("graph_prior"))
            .def("size", &DataModel::getSize)
            .def("graph", &DataModel::getGraph, py::return_value_policy::reference_internal)
            .def("graph_copy", &DataModel::getGraph, py::return_value_policy::copy)
            .def("set_graph", &DataModel::setGraph, py::arg("graph"))
            .def("graph_prior", &DataModel::getGraphPrior, py::return_value_policy::reference_internal)
            .def("set_graph_prior", &DataModel::setGraphPrior, py::arg("prior"))
            .def("sample_prior", &DataModel::samplePrior)
            .def("log_likelihood", &DataModel::getLogLikelihood)
            .def("log_prior", &DataModel::getLogPrior)
            .def("log_joint", &DataModel::getLogJoint)
            .def("log_likelihood_ratio_from_graph_move", &DataModel::getLogLikelihoodRatioFromGraphMove, py::arg("move"))
            .def("log_prior_ratio_from_graph_move", &DataModel::getLogPriorRatioFromGraphMove, py::arg("move"))
            .def("log_joint_ratio_from_graph_move", &DataModel::getLogJointRatioFromGraphMove, py::arg("move"))
            .def("apply_graph_move", &DataModel::applyGraphMove, py::arg("move"))
            .def("apply_param_move", &DataModel::applyParamMove, py::arg("move"))
            .def("is_valid_param_move", &DataModel::isValidParamMove, py::arg("move"))
            .def("log_acceptance_prob_from_graph_move", &DataModel::getLogAcceptanceProbFromGraphMove, py::arg("move"), py::arg("beta_prior") = 1, py::arg("beta_likelihood") = 1)
            .def("metropolis_graph_sweep", &DataModel::metropolisGraphSweep, py::arg("num_steps"), py::arg("beta_prior") = 1, py::arg("beta_likelihood") = 1)
            .def("metropolis_param_sweep", &DataModel::metropolisParamSweep, py::arg("num_steps"), py::arg("beta_prior") = 1, py::arg("beta_likelihood") = 1);

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
            .def("state", &Dynamics::getState, py::return_value_policy::reference_internal)
            .def("state_copy", &Dynamics::getState, py::return_value_policy::copy)
            .def("set_current_state", &Dynamics::setCurrentState, py::arg("state"))
            .def("set_state_from", [](Dynamics &self, const Dynamics &other)
                 { 
                    self.setGraph(other.getGraph());
                    self.setState(other.getPastStates(), other.getFutureStates()); })
            .def("set_state", py::overload_cast<Matrix<VertexState>>(&Dynamics::setState), py::arg("state"))
            .def("set_state", py::overload_cast<Matrix<VertexState>, Matrix<VertexState>>(&Dynamics::setState), py::arg("past"), py::arg("future"))
            .def("neighbors_state", &Dynamics::getNeighborsState, py::return_value_policy::reference_internal)
            .def("past_states", &Dynamics::getPastStates, py::return_value_policy::reference_internal)
            .def("past_neighbors_states", &Dynamics::getNeighborsPastStates, py::return_value_policy::reference_internal)
            .def("future_states", &Dynamics::getFutureStates, py::return_value_policy::reference_internal)
            .def("neighbors_state_copy", &Dynamics::getNeighborsState, py::return_value_policy::copy)
            .def("past_states_copy", &Dynamics::getPastStates, py::return_value_policy::copy)
            .def("past_neighbors_states_copy", &Dynamics::getNeighborsPastStates, py::return_value_policy::copy)
            .def("future_states_copy", &Dynamics::getFutureStates, py::return_value_policy::copy)
            .def("num_states", &Dynamics::getNumStates)
            .def("length", &Dynamics::getLength)
            .def("set_length", &Dynamics::setLength)
            .def("random_state", &Dynamics::getRandomState)
            .def("transition_matrix", &Dynamics::getTransitionMatrix, py::arg("out_state") = -1)
            .def("accept_selfloops", [](Dynamics &self)
                 { return self.acceptSelfLoops(); })
            .def(
                "accept_selfloops", [](Dynamics &self, bool condition)
                { self.acceptSelfLoops(condition); },
                py::arg("condition"))
            .def("sync_update_state", &Dynamics::syncUpdateState)
            .def("async_update_state", &Dynamics::asyncUpdateState,
                 py::arg("num_updates") = 1)
            .def("transition_prob", &Dynamics::getTransitionProb,
                 py::arg("prev_vertex_state"), py::arg("next_vertex_state"), py::arg("neighbor_state"))
            .def(
                "transition_probs",
                [](const Dynamics &self, BaseGraph::VertexIndex vertex)
                {
                    return self.getTransitionProbs(vertex);
                },
                py::arg("vertex"));

        py::class_<BinaryDynamics, Dynamics, PyBinaryDynamics<>>(dynamics, "BinaryDynamics")
            .def(py::init<RandomGraph &, size_t, double, double>(),
                 py::arg("graph_prior"), py::arg("length"),
                 py::arg("auto_activation_prob") = 0., py::arg("auto_deactivation_prob") = 0.)
            .def("activation_prob", &BinaryDynamics::getActivationProb, py::arg("neighbor_state"))
            .def("deactivation_prob", &BinaryDynamics::getDeactivationProb, py::arg("neighbor_state"))
            .def("set_auto_activation_prob", &BinaryDynamics::setAutoActivationProb, py::arg("auto_activation_prob"))
            .def("set_auto_deactivation_prob", &BinaryDynamics::setAutoDeactivationProb, py::arg("auto_deactivation_prob"))
            .def("auto_activation_prob", &BinaryDynamics::getAutoActivationProb)
            .def("auto_deactivation_prob", &BinaryDynamics::getAutoDeactivationProb)
            .def("random_state", [](const BinaryDynamics &self)
                 { return self.getRandomState(); })
            .def(
                "random_state", [](const BinaryDynamics &self, int initial)
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
            .def("a", &CowanDynamics::getA)
            .def("set_a", &CowanDynamics::setA, py::arg("a"))
            .def("nu", &CowanDynamics::getNu)
            .def("set_nu", &CowanDynamics::setNu, py::arg("nu"))
            .def("mu", &CowanDynamics::getMu)
            .def("set_mu", &CowanDynamics::setMu, py::arg("mu"))
            .def("eta", &CowanDynamics::getEta)
            .def("set_eta", &CowanDynamics::setEta, py::arg("eta"));

        py::class_<DegreeDynamics, BinaryDynamics>(dynamics, "DegreeDynamics")
            .def(py::init<RandomGraph &, size_t, double>(),
                 py::arg("graph_prior"), py::arg("length"), py::arg("C"))
            .def("c", &DegreeDynamics::getC)
            .def("set_c", &DegreeDynamics::setC, py::arg("c"));

        py::class_<GlauberDynamics, BinaryDynamics>(dynamics, "GlauberDynamics")
            .def(py::init<RandomGraph &, int, float, float, float>(),
                 py::arg("random_graph"), py::arg("length"), py::arg("coupling") = 1,
                 py::arg("auto_activation_prob") = 0., py::arg("auto_deactivation_prob") = 0.)
            .def("coupling", &GlauberDynamics::getCoupling)
            .def("set_coupling", &GlauberDynamics::setCoupling, py::arg("coupling"));

        py::class_<SISDynamics, BinaryDynamics>(dynamics, "SISDynamics")
            .def(py::init<RandomGraph &, size_t, double, double, double, double>(),
                 py::arg("random_graph"), py::arg("length"), py::arg("infection_prob") = 0.5, py::arg("recovery_prob") = 0.5,
                 py::arg("auto_activation_prob") = 1e-6, py::arg("auto_deactivation_prob") = 0.)
            .def("infection_prob", &SISDynamics::getInfectionProb)
            .def("set_infection_prob", &SISDynamics::setInfectionProb, py::arg("infection_prob"))
            .def("recovery_prob", &SISDynamics::getRecoveryProb)
            .def("set_recovery_prob", &SISDynamics::setRecoveryProb, py::arg("recovery_prob"));

        auto uncertain = m.def_submodule("uncertain");
        py::class_<UncertainGraph, DataModel, PyUncertainGraph<>>(uncertain, "UncertainGraph")
            .def(py::init<RandomGraph &>(), py::arg("prior"))
            .def("sample", &UncertainGraph::sample)
            .def("sample_state", &UncertainGraph::sampleState)
            .def("set_state", &UncertainGraph::setState, py::arg("state"))
            .def("set_state_from", [](UncertainGraph &self, const UncertainGraph &other)
                 {
                self.setGraph(other.getGraph());
                self.setState(other.getState()); })
            .def("state", &UncertainGraph::getState, py::return_value_policy::reference_internal)
            .def("state_copy", &UncertainGraph::getState, py::return_value_policy::copy);

        py::class_<PoissonUncertainGraph, UncertainGraph>(uncertain, "PoissonUncertainGraph")
            .def(py::init<RandomGraph &, double, double>(), py::arg("prior"), py::arg("mu"), py::arg("mu_no_edge") = 0);
    }

}

#endif
