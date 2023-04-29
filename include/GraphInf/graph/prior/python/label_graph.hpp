#ifndef GRAPH_INF_PYTHON_LABELGRAPH_H
#define GRAPH_INF_PYTHON_LABELGRAPH_H

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>

#include "GraphInf/types.h"
#include "GraphInf/graph/prior/python/prior.hpp"
#include "GraphInf/graph/prior/prior.hpp"
#include "GraphInf/graph/prior/edge_count.h"
#include "GraphInf/graph/prior/block.h"
#include "GraphInf/graph/prior/label_graph.h"

namespace GraphInf
{

    template <typename BaseClass = LabelGraphPrior>
    class PyLabelGraphPrior : public PyVertexLabeledPrior<LabelGraph, BlockIndex, BaseClass>
    {
    protected:
        void applyGraphMoveToState(const GraphMove &move) override { PYBIND11_OVERRIDE(void, BaseClass, applyGraphMoveToState, move); }
        void applyLabelMoveToState(const BlockMove &move) override { PYBIND11_OVERRIDE(void, BaseClass, applyLabelMoveToState, move); }
        void recomputeStateFromGraph() override { PYBIND11_OVERRIDE(void, BaseClass, recomputeStateFromGraph, ); }
        void recomputeConsistentState() override { PYBIND11_OVERRIDE(void, BaseClass, recomputeConsistentState, ); }

    public:
        using PyVertexLabeledPrior<LabelGraph, BlockIndex, BaseClass>::PyVertexLabeledPrior;
        /* Pure abstract methods */
        const double getLogLikelihoodRatioFromGraphMove(const GraphMove &move) const override { PYBIND11_OVERRIDE_PURE(const double, BaseClass, getLogLikelihoodRatioFromGraphMove, move); }
        const double getLogLikelihoodRatioFromLabelMove(const BlockMove &move) const override { PYBIND11_OVERRIDE_PURE(const double, BaseClass, getLogLikelihoodRatioFromLabelMove, move); }

        /* Overloaded abstract methods */
        void setPartition(const std::vector<BlockIndex> &partition) override { PYBIND11_OVERRIDE(void, BaseClass, setPartition, partition); }
        void checkSelfConsistency() const override { PYBIND11_OVERRIDE(void, BaseClass, checkSelfConsistency, ); }
    };

}

#endif
