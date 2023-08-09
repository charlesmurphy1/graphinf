#ifndef GRAPH_INF_PYWRAPPER_PRIOR_INIT_BLOCK_H
#define GRAPH_INF_PYWRAPPER_PRIOR_INIT_BLOCK_H

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>

#include "GraphInf/graph/prior/python/prior.hpp"
#include "GraphInf/graph/prior/block_count.h"
#include "GraphInf/graph/prior/block.h"
#include "GraphInf/graph/prior/python/block.hpp"

namespace py = pybind11;
namespace GraphInf
{

    void initBlockPrior(py::module &m)
    {
        py::class_<BlockPrior, BlockLabeledPrior<std::vector<BlockIndex>>, PyBlockPrior<>>(m, "BlockPrior")
            .def(py::init<>())
            .def(py::init<size_t, BlockCountPrior &>(), py::arg("size"), py::arg("block_count_prior"))
            .def("size", &BlockPrior::getSize)
            .def("set_size", &BlockPrior::setSize)
            .def("block_count", &BlockPrior::getBlockCount)
            .def("effective_block_count", &BlockPrior::getEffectiveBlockCount)
            .def("block_count_prior", &BlockPrior::getBlockCountPrior, py::return_value_policy::reference_internal)
            .def("block_count_prior_ref", &BlockPrior::getBlockCountPriorRef, py::return_value_policy::reference_internal)
            .def("set_block_count_prior", &BlockPrior::setBlockCountPrior)
            .def("vertex_counts", &BlockPrior::getVertexCounts, py::return_value_policy::reference_internal)
            .def("block", &BlockPrior::getBlock)
            .def("reduce_state", &BlockPrior::reduceState);

        py::class_<BlockDeltaPrior, BlockPrior>(m, "BlockDeltaPrior")
            .def(py::init<>())
            .def(py::init<const std::vector<BlockIndex> &>(), py::arg("blocks"));

        py::class_<BlockUniformPrior, BlockPrior>(m, "BlockUniformPrior")
            .def(py::init<>())
            .def(py::init<size_t, BlockCountPrior &>(), py::arg("size"), py::arg("block_count_prior"));

        py::class_<BlockUniformHyperPrior, BlockPrior>(m, "BlockUniformHyperPrior")
            .def(py::init<>())
            .def(py::init<size_t, BlockCountPrior &>(), py::arg("size"), py::arg("block_count_prior"));
    }

}

#endif
