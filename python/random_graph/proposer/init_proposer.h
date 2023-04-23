#ifndef GRAPH_INF_PYWRAPPER_INIT_PROPOSER_BASECLASS_H
#define GRAPH_INF_PYWRAPPER_INIT_PROPOSER_BASECLASS_H

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "GraphInf/random_graph/proposer/python/proposer.hpp"

#include "GraphInf/random_graph/proposer/movetypes.h"
#include "GraphInf/random_graph/proposer/proposer.hpp"

namespace py = pybind11;
namespace GraphInf{


template<typename MoveType>
py::class_<Proposer<MoveType>, NestedRandomVariable, PyProposer<MoveType>> declareProposerBaseClass(py::module& m, std::string pyName){
    return py::class_<Proposer<MoveType>, NestedRandomVariable, PyProposer<MoveType>>(m, pyName.c_str())
        .def(py::init<>())
        .def("propose_move", &Proposer<MoveType>::proposeMove)
        .def("clear", &Proposer<MoveType>::clear);
}

void initProposerBaseClass(py::module& m){
    declareProposerBaseClass<GraphMove>(m, "EdgeProposerBase");
    declareProposerBaseClass<BlockMove>(m, "BlockProposerBase");
}

}

#endif
