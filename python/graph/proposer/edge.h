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
            .def("get_vertex_proposal_counts", &HingeFlipProposer::getVertexProposalCounts);

        py::class_<HingeFlipUniformProposer, HingeFlipProposer>(m, "HingeFlipUniformProposer")
            .def(py::init<bool, bool>(), py::arg("allow_self_loops") = true, py::arg("allow_multiedges") = true);

        // py::class_<HingeFlipDegreeProposer, HingeFlipProposer>(m, "HingeFlipDegreeProposer")
        //     .def(py::init<bool, bool, double>(), py::arg("allow_self_loops") = true, py::arg("allow_multiedges") = true, py::arg("shift") = 1);

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
        // py::class_<SingleEdgeProposer, EdgeProposer, PySingleEdgeProposer<>>(m, "SingleEdgeProposer")
        //     .def(py::init<bool, bool, double, double, double>(), py::arg("allow_self_loops") = true, py::arg("allow_multiedges") = true, py::arg("bias") = 1.0, py::arg("min_add_edge_prob") = 0.1, py::arg("add_edge_prob") = -1)
        //     .def("set_vertex_sampler", &SingleEdgeProposer::setVertexSampler, py::arg("vertex_sampler"))
        //     .def("bias", &SingleEdgeProposer::getBias)
        //     .def("set_bias", &SingleEdgeProposer::setBias, py::arg("bias"))
        //     .def("add_edge_prob", &SingleEdgeProposer::getAddEdgeProb)
        //     .def("set_add_edge_prob", &SingleEdgeProposer::setAddEdgeProb, py::arg("add_edge_prob"));

        // py::class_<SingleEdgeUniformProposer, SingleEdgeProposer>(m, "SingleEdgeUniformProposer")
        //     .def(py::init<bool, bool, double>(), py::arg("allow_self_loops") = true, py::arg("allow_multiedges") = true, py::arg("bias") = 1.0);

        // py::class_<SingleEdgeDegreeProposer, SingleEdgeProposer>(m, "SingleEdgeDegreeProposer")
        //     .def(py::init<bool, bool, double, double>(), py::arg("allow_self_loops") = true, py::arg("allow_multiedges") = true, py::arg("bias") = 1.0, py::arg("shift") = 1);

        // py::class_<EdgeCountPreservingProposer, EdgeProposer>(m, "EdgeCountPreservingProposer")
        //     .def(py::init<bool, bool>(), py::arg("allow_self_loops") = true, py::arg("allow_multiedges") = true);

        // py::class_<NonPreservingProposer, EdgeProposer>(m, "NonPreservingProposer")
        //     .def(py::init<bool, bool>(), py::arg("allow_self_loops") = true, py::arg("allow_multiedges") = true)
        //     .def("set_single_edge_bias", &NonPreservingProposer::setSingleEdgeBias, py::arg("bias"));

        // /* Labeled edge proposers */
        // py::class_<LabeledEdgeProposer, EdgeProposer, PyLabeledEdgeProposer<>>(m, "LabeledEdgeProposer")
        //     .def(py::init<bool, bool, double>(), py::arg("allow_self_loops")=true, py::arg("allow_multiedges")=true,
        //          py::arg("label_pair_shift")=1)
        //     .def("on_label_creation", &LabeledEdgeProposer::onLabelCreation, py::arg("move"))
        //     .def("on_label_deletion", &LabeledEdgeProposer::onLabelDeletion, py::arg("move"));
        //
        // py::class_<LabeledDoubleEdgeSwapProposer, LabeledEdgeProposer>(m, "LabeledDoubleEdgeSwapProposer")
        //     .def(py::init<bool, bool, double>(), py::arg("allow_self_loops")=true, py::arg("allow_multiedges")=true,
        //          py::arg("label_pair_shift")=1);
        //
        // py::class_<LabeledHingeFlipProposer, LabeledEdgeProposer, PyLabeledHingeFlipProposer<>>(m, "LabeledHingeFlipProposer")
        //     .def(py::init<bool, bool, double>(), py::arg("allow_self_loops")=true, py::arg("allow_multiedges")=true,
        //          py::arg("label_pair_shift")=1);
        //
        // py::class_<LabeledHingeFlipUniformProposer, LabeledHingeFlipProposer>(m, "LabeledHingeFlipUniformProposer")
        //     .def(py::init<bool, bool, double>(), py::arg("allow_self_loops")=true, py::arg("allow_multiedges")=true,
        //          py::arg("label_pair_shift")=1);
        //
        // py::class_<LabeledHingeFlipDegreeProposer, LabeledHingeFlipProposer>(m, "LabeledHingeFlipDegreeProposer")
        //     .def(py::init<bool, bool, double, double>(), py::arg("allow_self_loops")=true, py::arg("allow_multiedges")=true,
        //          py::arg("label_pair_shift")=1, py::arg("vertex_shift")=1);
    }

}

#endif
