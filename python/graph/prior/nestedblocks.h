#ifndef GRAPH_INF_PYWRAPPER_PRIOR_INIT_NESTEDBLOCK_H
#define GRAPH_INF_PYWRAPPER_PRIOR_INIT_NESTEDBLOCK_H

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>

#include "BaseGraph/types.h"

#include "GraphInf/graph/prior/python/nested_block.hpp"
#include "GraphInf/graph/prior/nested_block.h"

namespace py = pybind11;
namespace GraphInf
{

     void initNestedBlockPrior(py::module &m)
     {
          py::class_<NestedBlockPrior, BlockPrior, PyNestedBlockPrior<>>(m, "NestedBlockPrior")
              .def(py::init<>())
              .def(py::init<size_t, NestedBlockCountPrior &>(), py::arg("size"), py::arg("nested_block_count_prior"))
              .def(
                  "nested_state", [](const NestedBlockPrior &self)
                  { return self.getNestedState(); },
                  py::return_value_policy::reference_internal)
              .def(
                  "nested_state", [](const NestedBlockPrior &self, Level level)
                  { return self.getNestedState(level); },
                  py::arg("level"), py::return_value_policy::reference_internal)
              .def("set_nested_state", [](NestedBlockPrior &self, std::vector<std::vector<BlockIndex>> state)
                   { self.setNestedState(state); })
              .def("depth", &NestedBlockPrior::getDepth)
              .def("nested_block_count_prior", &NestedBlockPrior::getNestedBlockCountPrior, py::return_value_policy::reference_internal)
              .def("set_nested_block_count_prior", &NestedBlockPrior::setNestedBlockCountPrior, py::arg("prior"))
              .def("nested_block_count", [](const NestedBlockPrior &self)
                   { return self.getNestedBlockCount(); })
              .def("nested_block_count", [](const NestedBlockPrior &self, Level level)
                   { return self.getNestedBlockCount(level); })
              .def("nested_max_block_count", [](const NestedBlockPrior &self)
                   { return self.getNestedMaxBlockCount(); })
              .def("nested_max_block_count", [](const NestedBlockPrior &self, Level level)
                   { return self.getNestedMaxBlockCount(level); })
              .def("nested_effective_block_count", [](const NestedBlockPrior &self)
                   { return self.getNestedEffectiveBlockCount(); })
              .def("nested_effective_block_count", [](const NestedBlockPrior &self, Level level)
                   { return self.getNestedEffectiveBlockCount(level); })
              .def(
                  "nested_vertex_counts", [](const NestedBlockPrior &self)
                  { return self.getNestedVertexCounts(); },
                  py::return_value_policy::reference_internal)
              .def(
                  "nested_vertex_counts", [](const NestedBlockPrior &self, Level level)
                  { return self.getNestedVertexCounts(level); },
                  py::return_value_policy::reference_internal)
              .def(
                  "nested_abs_vertex_counts", [](const NestedBlockPrior &self)
                  { return self.getNestedAbsVertexCounts(); },
                  py::return_value_policy::reference_internal)
              .def(
                  "nested_abs_vertex_counts", [](const NestedBlockPrior &self, Level level)
                  { return self.getNestedAbsVertexCounts(level); },
                  py::return_value_policy::reference_internal)
              .def("block_of_id", [](const NestedBlockPrior &self, BaseGraph::VertexIndex vertex, Level level)
                   { return self.getBlock(vertex, level); })
              .def("nested_block_of_id", [](const NestedBlockPrior &self, BlockIndex vertex, Level level)
                   { return self.getNestedBlock(vertex, level); })
              .def(
                  "reduce_state", [](NestedBlockPrior &self, Level minLevel)
                  { self.reduceState(minLevel); },
                  py::arg("min_level") = 0)
              .def(
                  "sample_state", [](const NestedBlockPrior &self, Level level)
                  { return self.sampleState(level); },
                  py::arg("level"))
              .def("sample_state", [](NestedBlockPrior &self)
                   { self.sampleState(); })
              .def("is_valid_block_move", &NestedBlockPrior::isValidBlockMove, py::arg("move"))
              .def("creating_new_block", &NestedBlockPrior::creatingNewBlock, py::arg("move"))
              .def("destroying_block", &NestedBlockPrior::destroyingBlock, py::arg("move"))
              .def("creating_new_Level", &NestedBlockPrior::creatingNewLevel, py::arg("move"));

          py::class_<NestedBlockUniformHyperPrior, NestedBlockPrior>(m, "NestedBlockUniformHyperPrior")
              .def(py::init<size_t>(), py::arg("graph_size"));
     }

}

#endif
