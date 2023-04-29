#ifndef GRAPH_INF_LIKELIHOOD_SBM_H
#define GRAPH_INF_LIKELIHOOD_SBM_H

#include "BaseGraph/types.h"
#include "GraphInf/graph/likelihood/likelihood.hpp"
#include "GraphInf/graph/prior/label_graph.h"
#include "GraphInf/graph/prior/block.h"
#include "GraphInf/utility/maps.hpp"
#include "GraphInf/types.h"

namespace GraphInf
{

    class StochasticBlockModelLikelihood : public VertexLabeledGraphLikelihoodModel<BlockIndex>
    {
    protected:
        void getDiffEdgeMatMapFromEdgeMove(const BaseGraph::Edge &, int, IntMap<std::pair<BlockIndex, BlockIndex>> &) const;
        void getDiffAdjMatMapFromEdgeMove(const BaseGraph::Edge &, int, IntMap<std::pair<BaseGraph::VertexIndex, BaseGraph::VertexIndex>> &) const;
        void getDiffEdgeMatMapFromBlockMove(const BlockMove &, IntMap<std::pair<BlockIndex, BlockIndex>> &) const;

    public:
        LabelGraphPrior **m_labelGraphPriorPtrPtr = nullptr;
        bool *m_withSelfLoopsPtr = nullptr;
        bool *m_withParallelEdgesPtr = nullptr;
    };

    class StubLabeledStochasticBlockModelLikelihood : public StochasticBlockModelLikelihood
    {
    protected:
        const double getLogLikelihoodRatioEdgeTerm(const GraphMove &) const;
        const double getLogLikelihoodRatioAdjTerm(const GraphMove &) const;

    public:
        const MultiGraph sample() const override
        {
            const auto &blocks = (*m_labelGraphPriorPtrPtr)->getBlocks();
            const auto &labelGraph = (*m_labelGraphPriorPtrPtr)->getState();
            return generateStubLabeledSBM(blocks, labelGraph, true);
        }
        const double getLogLikelihood() const override;
        const double getLogLikelihoodRatioFromGraphMove(const GraphMove &) const override;
        const double getLogLikelihoodRatioFromLabelMove(const BlockMove &) const override;
    };

    class UniformStochasticBlockModelLikelihood : public StochasticBlockModelLikelihood
    {
        const int vFunc(int nr) const
        {
            if (*m_withSelfLoopsPtr)
                return nr * (nr + 1) / 2;
            else
                return nr * (nr - 1) / 2;
        }
        const double logLikelihoodFunc(int nr, int ns, int ers, bool isSelfLoop) const
        {
            double vTerm;
            if (isSelfLoop)
            {
                if (*m_withSelfLoopsPtr)
                    vTerm = nr * (nr + 1) / 2;
                else
                    vTerm = nr * (nr - 1) / 2;
            }
            else
                vTerm = nr * ns;
            if (*m_withParallelEdgesPtr)
                return logMultisetCoefficient(vTerm, ers);
            return logBinomialCoefficient(vTerm, ers);
        }

    public:
        const MultiGraph sample() const override
        {
            const auto &blocks = (*m_labelGraphPriorPtrPtr)->getBlocks();
            const auto &labelGraph = (*m_labelGraphPriorPtrPtr)->getState();
            const auto &generate = (*m_withParallelEdgesPtr) ? generateMultiGraphSBM : generateSBM;
            return generate(blocks, labelGraph, *m_withSelfLoopsPtr);
        }
        const double getLogLikelihood() const override;
        const double getLogLikelihoodRatioFromGraphMove(const GraphMove &) const override;
        const double getLogLikelihoodRatioFromLabelMove(const BlockMove &) const override;
    };

}

#endif
