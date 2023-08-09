#ifndef GRAPH_INF_PYWRAPPER_INIT_LABELGRAPH_H
#define GRAPH_INF_PYWRAPPER_INIT_LABELGRAPH_H

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>

#include "GraphInf/graph/prior/python/prior.hpp"
#include "GraphInf/graph/prior/python/label_graph.hpp"
#include "GraphInf/graph/prior/edge_count.h"
#include "GraphInf/graph/prior/block.h"
#include "GraphInf/graph/prior/label_graph.h"

namespace py = pybind11;
namespace GraphInf
{

    void initLabelGraphPrior(py::module &m)
    {
        py::class_<LabelGraphPrior, BlockLabeledPrior<LabelGraph>, PyLabelGraphPrior<>>(m, "LabelGraphPrior")
            .def(py::init<>())
            .def(py::init<EdgeCountPrior &, BlockPrior &>(), py::arg("edge_count_prior"), py::arg("block_prior"))
            .def("edge_count", &LabelGraphPrior::getEdgeCount)
            .def("edge_counts", &LabelGraphPrior::getEdgeCounts, py::return_value_policy::reference_internal)
            .def("block_count", &LabelGraphPrior::getBlockCount)
            .def("blocks", &LabelGraphPrior::getBlocks, py::return_value_policy::reference_internal)
            .def("block", &LabelGraphPrior::getBlock, py::arg("vertex"))
            .def("graph", &LabelGraphPrior::getGraph, py::return_value_policy::reference_internal)
            .def("set_graph", &LabelGraphPrior::setGraph, py::arg("graph"))
            .def("set_partition", &LabelGraphPrior::setPartition)
            .def("edge_count_prior", &LabelGraphPrior::getEdgeCountPrior, py::return_value_policy::reference_internal)
            .def("set_edge_count_prior", &LabelGraphPrior::setEdgeCountPrior, py::arg("prior"))
            .def("block_prior", &LabelGraphPrior::getBlockPrior, py::return_value_policy::reference_internal)
            .def("set_block_prior", &LabelGraphPrior::setBlockPrior, py::arg("prior"))
            .def("reduce_partition", &LabelGraphPrior::reducePartition);

        py::class_<LabelGraphDeltaPrior, LabelGraphPrior>(m, "LabelGraphDeltaPrior")
            .def(py::init<>())
            .def(py::init<const std::vector<BlockIndex> &, const LabelGraph &>(), py::arg("blocks"), py::arg("label_graph"));

        py::class_<LabelGraphErdosRenyiPrior, LabelGraphPrior>(m, "LabelGraphErdosRenyiPrior")
            .def(py::init<>())
            .def(py::init<EdgeCountPrior &, BlockPrior &>(), py::arg("edge_count_prior"), py::arg("block_prior"));

        py::class_<LabelGraphPlantedPartitionPrior, LabelGraphPrior>(m, "LabelGraphPlantedPartitionPrior")
            .def(py::init<>())
            .def(py::init<EdgeCountPrior &, BlockPrior &>(), py::arg("edge_count_prior"), py::arg("block_prior"))
            .def("edge_count_in", &LabelGraphPlantedPartitionPrior::getEdgeCountIn)
            .def("edge_count_out", &LabelGraphPlantedPartitionPrior::getEdgeCountOut);
    }

}

#endif
