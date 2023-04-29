#ifndef GRAPH_INF_PYWRAPPER_INIT_PROPOSER_BASECLASS_H
#define GRAPH_INF_PYWRAPPER_INIT_PROPOSER_BASECLASS_H

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "GraphInf/graph/proposer/python/proposer.hpp"

#include "GraphInf/graph/proposer/movetypes.h"
#include "GraphInf/graph/proposer/proposer.hpp"

#include "movetypes.h"
#include "proposer.h"
#include "sampler.h"
#include "edge.h"
#include "label.h"

namespace py = pybind11;
namespace GraphInf
{

    template <typename MoveType>
    py::class_<Proposer<MoveType>, NestedRandomVariable, PyProposer<MoveType>> declareProposerBaseClass(py::module &m, std::string pyName)
    {
        return py::class_<Proposer<MoveType>, NestedRandomVariable, PyProposer<MoveType>>(m, pyName.c_str())
            .def(py::init<>())
            .def("propose_move", &Proposer<MoveType>::proposeMove)
            .def("clear", &Proposer<MoveType>::clear);
    }

    void initProposerBaseClass(py::module &m)
    {
        declareProposerBaseClass<GraphMove>(m, "EdgeProposerBase");
        declareProposerBaseClass<BlockMove>(m, "BlockProposerBase");
    }

    void initProposers(py::module &m)
    {
        initMoveTypes(m);
        initProposerBaseClass(m);

        auto sampler = m.def_submodule("sampler");
        initSampler(sampler);

        auto edge_proposer = m.def_submodule("edge");
        initEdgeProposer(edge_proposer);

        auto label_proposer = m.def_submodule("label");
        initLabelProposer(label_proposer);
    }

}

#endif
