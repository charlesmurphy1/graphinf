#ifndef GRAPH_INF_LIKELIHOOD_DCSBM_H
#define GRAPH_INF_LIKELIHOOD_DCSBM_H

#include "BaseGraph/types.h"
#include "GraphInf/graph/prior/block.h"
#include "GraphInf/graph/prior/label_graph.h"
#include "GraphInf/graph/prior/labeled_degree.h"
#include "GraphInf/utility/maps.hpp"
#include "GraphInf/graph/likelihood/likelihood.hpp"
#include "GraphInf/types.h"

namespace GraphInf
{

    class DegreeCorrectedStochasticBlockModelLikelihood : public VertexLabeledGraphLikelihoodModel<BlockIndex>
    {
    protected:
        void getDiffEdgeMatMapFromEdgeMove(const BaseGraph::Edge &, int, IntMap<std::pair<BlockIndex, BlockIndex>> &) const;
        void getDiffAdjMatMapFromEdgeMove(const BaseGraph::Edge &, int, IntMap<std::pair<BaseGraph::VertexIndex, BaseGraph::VertexIndex>> &) const;
        void getDiffEdgeMatMapFromBlockMove(const BlockMove &, IntMap<std::pair<BlockIndex, BlockIndex>> &) const;
        const double getLogLikelihoodRatioEdgeTerm(const GraphMove &) const;
        const double getLogLikelihoodRatioAdjTerm(const GraphMove &) const;

    public:
        const MultiGraph sample() const override
        {
            const auto &blocks = (*m_degreePriorPtrPtr)->getBlockPrior().getState();
            const auto &labelGraph = (*m_degreePriorPtrPtr)->getLabelGraphPrior().getState();
            const auto &degrees = (*m_degreePriorPtrPtr)->getState();
            return generateDCSBM(blocks, labelGraph, degrees);
        }
        const double getLogLikelihood() const override;
        const double getLogLikelihoodRatioFromGraphMove(const GraphMove &) const override;
        const double getLogLikelihoodRatioFromLabelMove(const BlockMove &) const override;
        VertexLabeledDegreePrior **m_degreePriorPtrPtr = nullptr;
    };

}

#endif
