#ifndef GRAPH_INF_PYWRAPPER_INIT_EDGEPROPOSER_H
#define GRAPH_INF_PYWRAPPER_INIT_EDGEPROPOSER_H

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "GraphInf/graph/proposer/python/edge_proposer.hpp"

#include "GraphInf/mcmc.h"
#include "GraphInf/graph/proposer/proposer.hpp"
#include "GraphInf/graph/proposer/edge/edge_proposer.h"
#include "GraphInf/graph/proposer/edge/double_edge_swap.h"
#include "GraphInf/graph/proposer/edge/hinge_flip.h"
#include "GraphInf/graph/proposer/edge/single_edge.h"

namespace py = pybind11;
namespace GraphInf
{

    void initEdgeProposer(py::module &m)
    {
        py::class_<EdgeProposer, Proposer<GraphMove>, PyEdgeProposer<>>(m, "EdgeProposer")
            .def(py::init<bool, bool>(), py::arg("allow_self_loops") = true, py::arg("allow_multiedges") = true)
            .def("set_up_with_graph", &EdgeProposer::setUpWithGraph, py::arg("graph"))
            .def("allow_self_loops", &EdgeProposer::allowSelfLoops)
            .def("allow_multiedges", &EdgeProposer::allowMultiEdges)
            .def("get_log_proposal_ratio", &EdgeProposer::getLogProposalProbRatio, py::arg("move"))
            .def("apply_graph_move", &EdgeProposer::applyGraphMove, py::arg("move"));

        /* Double edge swap proposers */
        py::class_<DoubleEdgeSwapProposer, EdgeProposer>(m, "DoubleEdgeSwapProposer")
            .def(py::init<bool, bool>(), py::arg("allow_self_loops") = true, py::arg("allow_multiedges") = true);

        /* Hinge flip proposers */
        py::class_<HingeFlipProposer, EdgeProposer, PyHingeFlipProposer<>>(m, "HingeFlipProposer")
            .def(py::init<bool, bool>(), py::arg("allow_self_loops") = true, py::arg("allow_multiedges") = true)
            .def("set_vertex_sampler", &HingeFlipProposer::setVertexSampler, py::arg("vertex_sampler"))
            .def("get_edge_proposal_counts", &HingeFlipProposer::getEdgeProposalCounts)
            .def("get_vertex_proposal_counts", &HingeFlipProposer::getVertexProposalCounts)
            .def("edge_sampler", &HingeFlipProposer::getEdgeSampler);

        py::class_<HingeFlipUniformProposer, HingeFlipProposer>(m, "HingeFlipUniformProposer")
            .def(py::init<bool, bool>(), py::arg("allow_self_loops") = true, py::arg("allow_multiedges") = true)
            .def("vertex_sampler", &HingeFlipUniformProposer::getVertexSampler);

        /* Single edge proposers */
        py::class_<SingleEdgeProposer, EdgeProposer>(m, "SingleEdgeProposer")
            .def(py::init<std::map<BaseGraph::Edge, double>, double, bool, bool>(), py::arg("weights"), py::arg("sample_new_edge_prob") = 0.5, py::arg("allow_self_loops") = true, py::arg("allow_multiedges") = true)
            .def(py::init<std::vector<std::vector<double>>, double, bool, bool>(), py::arg("weights"), py::arg("sample_new_edge_prob") = 0.5, py::arg("allow_self_loops") = true, py::arg("allow_multiedges") = true)
            .def(py::init<size_t, double, bool, bool>(), py::arg("size"), py::arg("sample_new_edge_prob") = 0.5, py::arg("allow_self_loops") = true, py::arg("allow_multiedges") = true)
            .def("get_edge_sampler", &SingleEdgeProposer::getEdgeSampler)
            .def("set_default_weights", &SingleEdgeProposer::setDefaultWeights, py::arg("size"))
            .def("set_weights", py::overload_cast<std::map<BaseGraph::Edge, double>>(&SingleEdgeProposer::setWeights), py::arg("weights"))
            .def("set_weights", py::overload_cast<std::vector<std::vector<double>>>(&SingleEdgeProposer::setWeights), py::arg("weights"))
            .def("update_weight", &SingleEdgeProposer::updateWeight, py::arg("edge"), py::arg("weight"));
    }

}

#endif
