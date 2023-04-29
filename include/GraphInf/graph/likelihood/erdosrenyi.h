#ifndef GRAPH_INF_LIKELIHOOD_ERDOSRENYI_H
#define GRAPH_INF_LIKELIHOOD_ERDOSRENYI_H

#include "BaseGraph/types.h"
#include "GraphInf/graph/likelihood/likelihood.hpp"
#include "GraphInf/graph/prior/edge_count.h"
#include "GraphInf/utility/maps.hpp"
#include "GraphInf/generators.h"
#include "GraphInf/types.h"

namespace GraphInf
{

    class ErdosRenyiLikelihood : public GraphLikelihoodModel
    {
    protected:
        const double getLogLikelihoodFromEdgeCount(size_t edgeCount) const
        {
            size_t N = *m_graphSizePtr;
            size_t A = 0;
            if (*m_withSelfLoopsPtr)
                A = N * (N + 1) / 2;
            else
                A = N * (N - 1) / 2;
            if (A < edgeCount and not *m_withParallelEdgesPtr)
                return -INFINITY;
            if (*m_withParallelEdgesPtr)
                return -logMultisetCoefficient(A, edgeCount);
            else
                return -logBinomialCoefficient(A, edgeCount);
        };

    public:
        const MultiGraph sample() const
        {
            const auto &generate = (*m_withParallelEdgesPtr) ? generateMultiGraphErdosRenyi : generateErdosRenyi;
            return generate(*m_graphSizePtr, (*m_edgeCountPriorPtrPtr)->getState(), *m_withSelfLoopsPtr);
        }
        const double getLogLikelihood() const
        {
            return getLogLikelihoodFromEdgeCount((*m_edgeCountPriorPtrPtr)->getState());
        }
        const double getLogLikelihoodRatioFromGraphMove(const GraphMove &move) const
        {
            int dE = move.addedEdges.size() - move.removedEdges.size();
            size_t E = (*m_edgeCountPriorPtrPtr)->getState();
            return getLogLikelihoodFromEdgeCount(E + dE) - getLogLikelihoodFromEdgeCount(E);
        }
        EdgeCountPrior **m_edgeCountPriorPtrPtr = nullptr;
        size_t *m_graphSizePtr = nullptr;
        bool *m_withSelfLoopsPtr = nullptr;
        bool *m_withParallelEdgesPtr = nullptr;
    };

}

#endif
