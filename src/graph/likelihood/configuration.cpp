#include "GraphInf/graph/likelihood/configuration.h"

namespace GraphInf
{

    const double ConfigurationModelLikelihood::getLogLikelihood() const
    {
        const size_t &E = (*m_degreePriorPtrPtr)->getEdgeCount();
        double logLikelihood = logDoubleFactorial(2 * E) - logFactorial(2 * E);

        for (const auto &vertex : *m_statePtr)
            logLikelihood += logFactorial(m_statePtr->getDegree(vertex));
        for (const auto &edge : m_statePtr->edges())
        {
            auto mult = m_statePtr->getEdgeMultiplicity(edge.first, edge.second);
            if (edge.first == edge.second)
                logLikelihood -= logDoubleFactorial(2 * mult);
            else
                logLikelihood -= logFactorial(mult);
        }

        return logLikelihood;
    }

    const double ConfigurationModelLikelihood::getLogLikelihoodRatioFromGraphMove(const GraphMove &move) const
    {
        IntMap<size_t> degreeDiffMap;
        IntMap<BaseGraph::Edge> edgeMultDiffMap;
        const auto &degrees = (*m_degreePriorPtrPtr)->getState();
        int dE = move.addedEdges.size() - move.removedEdges.size();
        const size_t &E = (*m_degreePriorPtrPtr)->getEdgeCount();

        for (auto edge : move.addedEdges)
        {
            edgeMultDiffMap.increment(getOrderedEdge(edge));
            degreeDiffMap.increment(edge.first);
            degreeDiffMap.increment(edge.second);
        }

        for (auto edge : move.removedEdges)
        {
            edgeMultDiffMap.decrement(getOrderedEdge(edge));
            degreeDiffMap.decrement(edge.first);
            degreeDiffMap.decrement(edge.second);
        }

        double logLikelihoodRatio = logDoubleFactorial(2 * (E + dE)) - logFactorial(2 * (E + dE));
        logLikelihoodRatio -= logDoubleFactorial(2 * E) - logFactorial(2 * E);
        for (auto diff : degreeDiffMap)
        {
            logLikelihoodRatio += logFactorial(degrees[diff.first] + diff.second) - logFactorial(degrees[diff.first]);
        }

        for (auto diff : edgeMultDiffMap)
        {
            const auto &edge = diff.first;
            size_t edgeMult = m_statePtr->getEdgeMultiplicity(edge.first, edge.second);
            if (edge.first == edge.second)
                logLikelihoodRatio -= logDoubleFactorial(2 * (edgeMult + diff.second)) - logDoubleFactorial(2 * edgeMult);
            else
                logLikelihoodRatio -= logFactorial(edgeMult + diff.second) - logFactorial(edgeMult);
        }

        return logLikelihoodRatio;
    }

}
