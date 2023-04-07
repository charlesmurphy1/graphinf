#ifndef GRAPH_INF_PYWRAPPER_INIT_DYNAMICS_H
#define GRAPH_INF_PYWRAPPER_INIT_DYNAMICS_H

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "GraphInf/data/python/data_model.hpp"
#include "GraphInf/data/dynamics/dynamics.hpp"
#include "GraphInf/data/dynamics/binary_dynamics.hpp"
#include "GraphInf/data/dynamics/cowan.hpp"
#include "GraphInf/data/dynamics/degree.hpp"
#include "GraphInf/data/dynamics/glauber.hpp"
#include "GraphInf/data/dynamics/sis.hpp"

namespace py = pybind11;
namespace GraphInf{

template <typename GraphPriorType>
py::class_<Dynamics<GraphPriorType>, DataModel<GraphPriorType>, PyDynamics<GraphPriorType>> declareDynamicsBaseClass(py::module& m, std::string pyName){
    return py::class_<Dynamics<GraphPriorType>, DataModel<GraphPriorType>, PyDynamics<GraphPriorType>>(m, pyName.c_str())
        .def(py::init<size_t, size_t>(),
            py::arg("num_states"),
            py::arg("length"))
        .def(py::init<GraphPriorType&, size_t, size_t>(),
            py::arg("graph_prior"),
            py::arg("num_states"),
            py::arg("length"))
        .def("sample_state", [](Dynamics<GraphPriorType>& self, const State& initial, bool asyncMode=false, size_t initialBurn=0){
            self.sampleState(initial, asyncMode, initialBurn);
        }, py::arg("initial"), py::arg("async_mode")=false, py::arg("initial_burn")=0)
        .def("sample_state", [](Dynamics<GraphPriorType>& self, bool asyncMode=false, size_t initialBurn=0){
            self.sampleState({}, asyncMode, initialBurn);
        }, py::arg("async_mode")=false, py::arg("initial_burn")=0)
        .def("sample", [](Dynamics<GraphPriorType>& self, const State& initial, bool asyncMode=false, size_t initialBurn=0){
            self.sample(initial, asyncMode, initialBurn);
        }, py::arg("initial"), py::arg("async_mode")=false, py::arg("initial_burn")=0)
        .def("sample", [](Dynamics<GraphPriorType>& self, bool asyncMode=false, size_t initialBurn=0){
            self.sample({}, asyncMode, initialBurn);
        }, py::arg("async_mode")=false, py::arg("initial_burn")=0)
        .def("get_state", &Dynamics<GraphPriorType>::getState)
        .def("set_state", &Dynamics<GraphPriorType>::setState, py::arg("state"))
        .def("get_neighbors_state", &Dynamics<GraphPriorType>::getNeighborsState)
        .def("get_past_states", &Dynamics<GraphPriorType>::getPastStates)
        .def("get_past_neighbors_states", &Dynamics<GraphPriorType>::getNeighborsPastStates)
        .def("get_future_states", &Dynamics<GraphPriorType>::getFutureStates)
        .def("get_num_states", &Dynamics<GraphPriorType>::getNumStates)
        .def("get_length", &Dynamics<GraphPriorType>::getLength)
        .def("set_length", &Dynamics<GraphPriorType>::setLength)
        .def("get_past_length", &Dynamics<GraphPriorType>::getPastLength)
        .def("set_past_length", &Dynamics<GraphPriorType>::setPastLength)
        .def("get_random_state", &Dynamics<GraphPriorType>::getRandomState)
        .def("accept_selfloops", [](Dynamics<GraphPriorType>& self){ return self.acceptSelfLoops(); })
        .def("accept_selfloops", [](Dynamics<GraphPriorType>& self, bool condition){ self.acceptSelfLoops(condition); }, py::arg("condition"))
        .def("sync_update_state", &Dynamics<GraphPriorType>::syncUpdateState)
        .def("async_update_state", &Dynamics<GraphPriorType>::asyncUpdateState,
            py::arg("num_updates")=1)
        .def("get_transition_prob", &Dynamics<GraphPriorType>::getTransitionProb,
            py::arg("prev_vertex_state"), py::arg("next_vertex_state"), py::arg("neighbor_state"))
        .def("get_transition_probs",
            [](const Dynamics<GraphPriorType>& self, BaseGraph::VertexIndex vertex) {
                return self.getTransitionProbs(vertex);
            }, py::arg("vertex"))
        ;
}

template <typename GraphPriorType>
py::class_<BinaryDynamics<GraphPriorType>, Dynamics<GraphPriorType>, PyBinaryDynamics<GraphPriorType>> declareBinaryDynamicsBaseClass(py::module& m, std::string pyName){
    return py::class_<BinaryDynamics<GraphPriorType>, Dynamics<GraphPriorType>, PyBinaryDynamics<GraphPriorType>>(m, pyName.c_str())
        .def(py::init<GraphPriorType&, size_t, double, double>(),
             py::arg("random_graph"), py::arg("length"),
             py::arg("auto_activation_prob")=0., py::arg("auto_deactivation_prob")=0.)
        .def(py::init<size_t, double, double>(), py::arg("length"),
             py::arg("auto_activation_prob")=0., py::arg("auto_deactivation_prob")=0.)
        .def("get_activation_prob", &BinaryDynamics<GraphPriorType>::getActivationProb, py::arg("neighbor_state"))
        .def("get_deactivation_prob", &BinaryDynamics<GraphPriorType>::getDeactivationProb, py::arg("neighbor_state"))
        .def("set_auto_activation_prob", &BinaryDynamics<GraphPriorType>::setAutoActivationProb, py::arg("auto_activation_prob"))
        .def("set_auto_deactivation_prob", &BinaryDynamics<GraphPriorType>::setAutoDeactivationProb, py::arg("auto_deactivation_prob"))
        .def("get_auto_activation_prob", &BinaryDynamics<GraphPriorType>::getAutoActivationProb)
        .def("get_auto_deactivation_prob", &BinaryDynamics<GraphPriorType>::getAutoDeactivationProb)
        .def("get_random_state", [](const BinaryDynamics<GraphPriorType>& self){ return self.getRandomState(); })
        .def("get_random_state", [](const BinaryDynamics<GraphPriorType>& self, int initial){ return self.getRandomState(initial); },
            py::arg("initial_active"))
        .def("sample_state", [](BinaryDynamics<GraphPriorType>& self, bool asyncMode=false, size_t initialBurn=0){
            self.sampleState({}, asyncMode, initialBurn);
        }, py::arg("async_mode")=false, py::arg("initial_burn")=0)
        .def("sample_state", [](BinaryDynamics<GraphPriorType>& self, int initialActives, bool asyncMode=false, size_t initialBurn=0){
            self.sampleState(self.getRandomState(initialActives), asyncMode, initialBurn);
        }, py::arg("initial_actives"), py::arg("async_mode")=false, py::arg("initial_burn")=0)
        .def("sample_state", [](BinaryDynamics<GraphPriorType>& self, const State& initial, bool asyncMode=false, size_t initialBurn=0){
            self.sampleState(initial, asyncMode, initialBurn);
        }, py::arg("initial"), py::arg("async_mode")=false, py::arg("initial_burn")=0)
        .def("sample", [](BinaryDynamics<GraphPriorType>& self, bool asyncMode=false, size_t initialBurn=0){
            self.sample({}, asyncMode, initialBurn);
        }, py::arg("async_mode")=false, py::arg("initial_burn")=0)
        .def("sample", [](BinaryDynamics<GraphPriorType>& self, const State& initial, bool asyncMode=false, size_t initialBurn=0){
            self.sample(initial, asyncMode, initialBurn);
        }, py::arg("initial"), py::arg("async_mode")=false, py::arg("initial_burn")=0)
        .def("sample", [](BinaryDynamics<GraphPriorType>& self, int initialActives, bool asyncMode=false, size_t initialBurn=0){
            self.sample(self.getRandomState(initialActives), asyncMode, initialBurn);
        }, py::arg("initial_actives"), py::arg("async_mode")=false, py::arg("initial_burn")=0)
        ;
}

template<typename GraphPriorType>
py::class_<CowanDynamics<GraphPriorType>, BinaryDynamics<GraphPriorType>> declareCowanDynamicsBaseClass(py::module& m, std::string pyName){
    return py::class_<CowanDynamics<GraphPriorType>, BinaryDynamics<GraphPriorType>>(m, pyName.c_str())
        .def(py::init<size_t, double, double, double, double, double, double>(),
            py::arg("length"), py::arg("nu")=1,
            py::arg("a")=1, py::arg("mu")=1, py::arg("eta")=0.5,
            py::arg("auto_activation_prob")=1e-6, py::arg("auto_deactivation_prob")=0.)
        .def(py::init<GraphPriorType&, size_t, double, double, double, double, double, double>(),
            py::arg("random_graph"), py::arg("length"), py::arg("nu")=1,
            py::arg("a")=1, py::arg("mu")=1, py::arg("eta")=0.5,
            py::arg("auto_activation_prob")=1e-6, py::arg("auto_deactivation_prob")=0.)
        .def("get_a", &CowanDynamics<GraphPriorType>::getA)
        .def("set_a", &CowanDynamics<GraphPriorType>::setA, py::arg("a"))
        .def("get_nu", &CowanDynamics<GraphPriorType>::getNu)
        .def("set_nu", &CowanDynamics<GraphPriorType>::setNu, py::arg("nu"))
        .def("get_mu", &CowanDynamics<GraphPriorType>::getMu)
        .def("set_mu", &CowanDynamics<GraphPriorType>::setMu, py::arg("mu"))
        .def("get_eta", &CowanDynamics<GraphPriorType>::getEta)
        .def("set_eta", &CowanDynamics<GraphPriorType>::setEta, py::arg("eta"));
}

template<typename GraphPriorType>
py::class_<DegreeDynamics<GraphPriorType>, BinaryDynamics<GraphPriorType>> declareDegreeDynamicsBaseClass(py::module& m, std::string pyName){
    return py::class_<DegreeDynamics<GraphPriorType>, BinaryDynamics<GraphPriorType>>(m, pyName.c_str())
        .def(py::init<size_t, double>(),
            py::arg("length"), py::arg("C"))
        .def(py::init<GraphPriorType&, size_t, double>(),
            py::arg("random_graph"), py::arg("length"), py::arg("C"))
        .def("get_c", &DegreeDynamics<GraphPriorType>::getC)
        .def("set_c", &DegreeDynamics<GraphPriorType>::setC, py::arg("c"));
}

template<typename GraphPriorType>
py::class_<GlauberDynamics<GraphPriorType>, BinaryDynamics<GraphPriorType>> declareGlauberDynamicsBaseClass(py::module& m, std::string pyName){
    return py::class_<GlauberDynamics<GraphPriorType>, BinaryDynamics<GraphPriorType>>(m, pyName.c_str())
        .def(py::init<size_t, double, double, double>(),
            py::arg("length"), py::arg("coupling")=1,
            py::arg("auto_activation_prob")=0., py::arg("auto_deactivation_prob")=0.)
        .def(py::init<GraphPriorType&, size_t, double, double, double>(),
            py::arg("random_graph"), py::arg("length"), py::arg("coupling")=1,
            py::arg("auto_activation_prob")=0., py::arg("auto_deactivation_prob")=0.)
        .def("get_coupling", &GlauberDynamics<GraphPriorType>::getCoupling)
        .def("set_coupling", &GlauberDynamics<GraphPriorType>::setCoupling, py::arg("coupling"));
}

template<typename GraphPriorType>
py::class_<SISDynamics<GraphPriorType>, BinaryDynamics<GraphPriorType>> declareSISDynamicsBaseClass(py::module& m, std::string pyName){
    return py::class_<SISDynamics<GraphPriorType>, BinaryDynamics<GraphPriorType>>(m, pyName.c_str())
        .def(py::init<size_t, double, double, double, double>(),
            py::arg("length"), py::arg("infection_prob")=0.5, py::arg("recovery_prob")=0.5,
            py::arg("auto_activation_prob")=1e-6, py::arg("auto_deactivation_prob")=0.)
        .def(py::init<GraphPriorType&, size_t, double, double, double, double>(),
            py::arg("random_graph"), py::arg("length"), py::arg("infection_prob")=0.5, py::arg("recovery_prob")=0.5,
            py::arg("auto_activation_prob")=1e-6, py::arg("auto_deactivation_prob")=0.)
        .def("get_infection_prob", &SISDynamics<GraphPriorType>::getInfectionProb)
        .def("set_infection_prob", &SISDynamics<GraphPriorType>::setInfectionProb, py::arg("infection_prob"))
        .def("get_recovery_prob", &SISDynamics<GraphPriorType>::getRecoveryProb)
        .def("set_recovery_prob", &SISDynamics<GraphPriorType>::setRecoveryProb, py::arg("recovery_prob"));
}



void initDynamics(py::module& m){

    declareDynamicsBaseClass<RandomGraph>(m, "Dynamics");
    declareDynamicsBaseClass<BlockLabeledRandomGraph>(m, "BlockLabeledDynamics");
    declareDynamicsBaseClass<NestedBlockLabeledRandomGraph>(m, "NestedBlockLabeledDynamics");

    declareBinaryDynamicsBaseClass<RandomGraph>(m, "BinaryDynamics");
    declareBinaryDynamicsBaseClass<BlockLabeledRandomGraph>(m, "BlockLabeledBinaryDynamics");
    declareBinaryDynamicsBaseClass<NestedBlockLabeledRandomGraph>(m, "NestedBlockLabeledBinaryDynamics");

    declareCowanDynamicsBaseClass<RandomGraph>(m, "CowanDynamics");
    declareCowanDynamicsBaseClass<BlockLabeledRandomGraph>(m, "BlockLabeledCowanDynamics");
    declareCowanDynamicsBaseClass<NestedBlockLabeledRandomGraph>(m, "NestedBlockLabeledCowanDynamics");

    declareDegreeDynamicsBaseClass<RandomGraph>(m, "DegreeDynamics");
    declareDegreeDynamicsBaseClass<BlockLabeledRandomGraph>(m, "BlockLabeledDegreeDynamics");
    declareDegreeDynamicsBaseClass<NestedBlockLabeledRandomGraph>(m, "NestedBlockLabeledDegreeDynamics");

    declareGlauberDynamicsBaseClass<RandomGraph>(m, "GlauberDynamics");
    declareGlauberDynamicsBaseClass<BlockLabeledRandomGraph>(m, "BlockLabeledGlauberDynamics");
    declareGlauberDynamicsBaseClass<NestedBlockLabeledRandomGraph>(m, "NestedBlockLabeledGlauberDynamics");

    declareSISDynamicsBaseClass<RandomGraph>(m, "SISDynamics");
    declareSISDynamicsBaseClass<BlockLabeledRandomGraph>(m, "BlockLabeledSISDynamics");
    declareSISDynamicsBaseClass<NestedBlockLabeledRandomGraph>(m, "NestedBlockLabeledSISDynamics");
}

}
#endif