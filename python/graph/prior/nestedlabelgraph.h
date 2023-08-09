#ifndef GRAPH_INF_PYWRAPPER_INIT_NESTEDLABELGRAPH_H
#define GRAPH_INF_PYWRAPPER_INIT_NESTEDLABELGRAPH_H

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>

#include "GraphInf/graph/prior/python/nested_label_graph.hpp"
#include "GraphInf/graph/prior/nested_label_graph.h"

namespace py = pybind11;
namespace GraphInf
{

    void initNestedLabelGraphPrior(py::module &m)
    {
        py::class_<NestedLabelGraphPrior, LabelGraphPrior, PyNestedLabelGraphPrior<>>(m, "NestedLabelGraphPrior")
            .def(py::init<>())
            .def(py::init<EdgeCountPrior &, NestedBlockPrior &>(), py::arg("edge_count_prior"), py::arg("nested_block_prior"))
            .def(
                "sample_state", [](const NestedLabelGraphPrior &self, Level level)
                { return self.sampleState(level); },
                py::arg("level"))
            .def(
                "log_likelihood_at_level", [](const NestedLabelGraphPrior &self, Level level)
                { return self.getLogLikelihoodAtLevel(level); },
                py::arg("level"))
            .def(
                "nested_state", [](const NestedLabelGraphPrior &self)
                { return self.getNestedState(); },
                py::return_value_policy::reference_internal)
            .def(
                "nested_state", [](const NestedLabelGraphPrior &self, Level level)
                { return self.getNestedState(level); },
                py::arg("level"), py::return_value_policy::reference_internal)
            .def("set_nested_state", &NestedLabelGraphPrior::setNestedState, py::arg("state"))
            .def("nested_block_prior", &NestedLabelGraphPrior::getNestedBlockPrior, py::return_value_policy::reference_internal)
            .def("set_nested_block_prior", &NestedLabelGraphPrior::setNestedBlockPrior, py::arg("prior"))
            .def("nested_block_count", [](const NestedLabelGraphPrior &self)
                 { return self.getNestedBlockCount(); })
            .def(
                "nested_block_count", [](const NestedLabelGraphPrior &self, Level level)
                { return self.getNestedBlockCount(level); },
                py::arg("level"))
            .def(
                "nested_blocks", [](const NestedLabelGraphPrior &self)
                { return self.getNestedBlocks(); },
                py::return_value_policy::reference_internal)
            .def(
                "nested_blocks", [](const NestedLabelGraphPrior &self, Level level)
                { return self.getNestedBlocks(level); },
                py::arg("level"), py::return_value_policy::reference_internal)
            .def(
                "nested_vertex_counts", [](const NestedLabelGraphPrior &self)
                { return self.getNestedVertexCounts(); },
                py::return_value_policy::reference_internal)
            .def(
                "nested_vertex_counts", [](const NestedLabelGraphPrior &self, Level level)
                { return self.getNestedVertexCounts(level); },
                py::arg("level"), py::return_value_policy::reference_internal)
            .def(
                "nested_edge_counts", [](const NestedLabelGraphPrior &self)
                { return self.getNestedEdgeCounts(); },
                py::return_value_policy::reference_internal)
            .def(
                "nested_edge_counts", [](const NestedLabelGraphPrior &self, Level level)
                { return self.getNestedEdgeCounts(level); },
                py::arg("level"), py::return_value_policy::reference_internal)
            .def("block", [](const NestedLabelGraphPrior &self, BaseGraph::VertexIndex vertex, Level level)
                 { return self.getBlock(vertex, level); })
            .def("nested_block", [](const NestedLabelGraphPrior &self, BlockIndex vertex, Level level)
                 { return self.getNestedBlock(vertex, level); })
            .def("depth", &NestedLabelGraphPrior::getDepth)
            .def("set_nested_partition", &NestedLabelGraphPrior::setNestedPartition, py::arg("nested_partition"))
            .def(
                "reduce_partition", [](NestedLabelGraphPrior &self, Level minLevel)
                { return self.reducePartition(minLevel); },
                py::arg("min_level"));

        py::class_<NestedStochasticBlockLabelGraphPrior, NestedLabelGraphPrior>(m, "NestedStochasticBlockLabelGraphPrior")
            .def(py::init<size_t>(), py::arg("graph_size"))
            .def(py::init<size_t, EdgeCountPrior &>(), py::arg("graph_size"), py::arg("edge_count_prior"));
    }

}

#endif
