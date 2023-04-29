#include "GraphInf/graph/likelihood/dcsbm.h"
#include "BaseGraph/types.h"

using namespace BaseGraph;
namespace GraphInf
{
    void DegreeCorrectedStochasticBlockModelLikelihood::getDiffEdgeMatMapFromEdgeMove(const BaseGraph::Edge &edge, int counter, IntMap<std::pair<BlockIndex, BlockIndex>> &diffEdgeMatMap) const
    {
        const BlockSequence &blockSeq = (*m_degreePriorPtrPtr)->getBlockPrior().getState();
        diffEdgeMatMap.increment(
            getOrderedPair<BlockIndex>({blockSeq[edge.first], blockSeq[edge.second]}),
            counter);
    }

    void DegreeCorrectedStochasticBlockModelLikelihood::getDiffAdjMatMapFromEdgeMove(const Edge &edge, int counter, IntMap<std::pair<VertexIndex, VertexIndex>> &diffAdjMatMap) const
    {
        Edge orderedEdge = getOrderedEdge(edge);
        diffAdjMatMap.increment({orderedEdge.first, orderedEdge.second}, counter);
    }

    void DegreeCorrectedStochasticBlockModelLikelihood::getDiffEdgeMatMapFromBlockMove(
        const BlockMove &move, IntMap<std::pair<BlockIndex, BlockIndex>> &diffEdgeMatMap) const
    {
        const BlockSequence &blockSeq = (*m_degreePriorPtrPtr)->getBlockPrior().getState();
        for (auto neighbor : m_statePtr->getOutNeighbours(move.vertexIndex))
        {
            BlockIndex blockIdx = blockSeq[neighbor];
            size_t mult = m_statePtr->getEdgeMultiplicity(move.vertexIndex, neighbor);
            std::pair<BlockIndex, BlockIndex> orderedBlockPair = getOrderedPair<BlockIndex>({move.prevLabel, blockIdx});
            diffEdgeMatMap.decrement(orderedBlockPair, mult);

            if (neighbor == move.vertexIndex) // handling self-loops
                blockIdx = move.nextLabel;

            orderedBlockPair = getOrderedPair<BlockIndex>({move.nextLabel, blockIdx});
            diffEdgeMatMap.increment(orderedBlockPair, mult);
        }
    }

    const double DegreeCorrectedStochasticBlockModelLikelihood::getLogLikelihood() const
    {
        double logLikelihood = 0;
        const MultiGraph &labelGraph = (*m_degreePriorPtrPtr)->getLabelGraphPrior().getState();
        for (auto r : labelGraph)
        {
            logLikelihood -= logFactorial(labelGraph.getDegree(r));
            for (auto s : labelGraph.getOutNeighbours(r))
            {
                const auto ers = labelGraph.getEdgeMultiplicity(r, s);
                if (r == s)
                    logLikelihood += logDoubleFactorial(2 * ers);
                else if (r < s)
                    logLikelihood += logFactorial(ers);
            }
        }

        for (auto i : *m_statePtr)
        {
            logLikelihood += logFactorial(m_statePtr->getDegree(i));
            for (auto j : m_statePtr->getOutNeighbours(i))
            {
                const auto mult = m_statePtr->getEdgeMultiplicity(i, j);
                if (i == j)
                    logLikelihood -= logDoubleFactorial(2 * mult);
                else if (i < j)
                    logLikelihood -= logFactorial(mult);
            }
        }

        return logLikelihood;
    }

    const double DegreeCorrectedStochasticBlockModelLikelihood::getLogLikelihoodRatioEdgeTerm(const GraphMove &move) const
    {
        const BlockSequence &blockSeq = (*m_degreePriorPtrPtr)->getBlockPrior().getState();
        const LabelGraph &labelGraph = (*m_degreePriorPtrPtr)->getLabelGraphPrior().getState();
        double logLikelihoodRatioTerm = 0;

        IntMap<std::pair<BlockIndex, BlockIndex>> diffEdgeMatMap;
        IntMap<BlockIndex> diffEdgeCountsInBlocksMap;

        for (auto edge : move.addedEdges)
        {
            const auto rs = OrderedPair<BlockIndex>({blockSeq[edge.first], blockSeq[edge.second]});
            diffEdgeMatMap.increment(rs);
            diffEdgeCountsInBlocksMap.increment(rs.first);
            diffEdgeCountsInBlocksMap.increment(rs.second);
        }
        for (auto edge : move.removedEdges)
        {
            const auto rs = OrderedPair<BlockIndex>({blockSeq[edge.first], blockSeq[edge.second]});
            diffEdgeMatMap.decrement(rs);
            diffEdgeCountsInBlocksMap.decrement(rs.first);
            diffEdgeCountsInBlocksMap.decrement(rs.second);
        }

        for (auto diff : diffEdgeMatMap)
        {
            auto r = diff.first.first, s = diff.first.second;
            const auto ers = labelGraph.getEdgeMultiplicity(r, s);
            if (r == s)
                logLikelihoodRatioTerm += logDoubleFactorial(2 * (ers + diff.second)) - logDoubleFactorial(2 * ers);
            else
                logLikelihoodRatioTerm += logFactorial(ers + diff.second) - logFactorial(ers);
        }

        for (auto diff : diffEdgeCountsInBlocksMap)
        {
            const auto er = labelGraph.getDegree(diff.first);
            logLikelihoodRatioTerm -= logFactorial(er + diff.second) - logFactorial(er);
        }
        return logLikelihoodRatioTerm;
    }

    const double DegreeCorrectedStochasticBlockModelLikelihood::getLogLikelihoodRatioAdjTerm(const GraphMove &move) const
    {
        IntMap<std::pair<VertexIndex, VertexIndex>> diffAdjMatMap;
        IntMap<VertexIndex> diffDegreeMap;
        double logLikelihoodRatioTerm = 0;

        for (auto edge : move.addedEdges)
        {
            diffAdjMatMap.increment(getOrderedEdge(edge));
            diffDegreeMap.increment(edge.first);
            diffDegreeMap.increment(edge.second);
        }
        for (auto edge : move.removedEdges)
        {
            diffAdjMatMap.decrement(getOrderedEdge(edge));
            diffDegreeMap.decrement(edge.first);
            diffDegreeMap.decrement(edge.second);
        }

        for (auto diff : diffAdjMatMap)
        {
            auto i = diff.first.first, j = diff.first.second;
            auto mult = m_statePtr->getEdgeMultiplicity(i, j);
            if (i == j)
                logLikelihoodRatioTerm -= logDoubleFactorial(2 * (mult + diff.second)) - logDoubleFactorial(2 * mult);
            else
                logLikelihoodRatioTerm -= logFactorial(mult + diff.second) - logFactorial(mult);
        }

        const DegreeSequence &degreeSeq = (*m_degreePriorPtrPtr)->getState();
        for (auto diff : diffDegreeMap)
        {
            const auto &k = m_statePtr->getDegree(diff.first);
            logLikelihoodRatioTerm += logFactorial(k + diff.second) - logFactorial(k);
        }
        return logLikelihoodRatioTerm;
    }

    const double DegreeCorrectedStochasticBlockModelLikelihood::getLogLikelihoodRatioFromGraphMove(const GraphMove &move) const
    {
        return getLogLikelihoodRatioEdgeTerm(move) + getLogLikelihoodRatioAdjTerm(move);
    }

    const double DegreeCorrectedStochasticBlockModelLikelihood::getLogLikelihoodRatioFromLabelMove(const BlockMove &move) const
    {
        if (move.prevLabel == move.nextLabel or move.level > 0)
            return 0;
        const BlockSequence &blockSeq = (*m_degreePriorPtrPtr)->getBlockPrior().getState();
        const LabelGraph &labelGraph = (*m_degreePriorPtrPtr)->getLabelGraphPrior().getState();
        double logLikelihoodRatio = 0;

        IntMap<std::pair<BlockIndex, BlockIndex>> diffEdgeMatMap;
        IntMap<BlockIndex> diffEdgesInBlockMap;
        getDiffEdgeMatMapFromBlockMove(move, diffEdgeMatMap);
        for (auto diff : diffEdgeMatMap)
        {
            size_t ers;
            auto r = diff.first.first, s = diff.first.second;
            auto dErs = diff.second;
            diffEdgesInBlockMap.increment(r, dErs);
            diffEdgesInBlockMap.increment(s, dErs);

            if (r < labelGraph.getSize() and s < labelGraph.getSize())
                ers = labelGraph.getEdgeMultiplicity(r, s);
            else
                ers = 0;

            if (r == s)
                logLikelihoodRatio += logDoubleFactorial(2 * ers + 2 * dErs) - logDoubleFactorial(2 * ers);
            else
                logLikelihoodRatio += logFactorial(ers + dErs) - logFactorial(ers);
        }

        for (auto diff : diffEdgesInBlockMap)
        {
            auto r = diff.first;
            auto dEr = diff.second;
            size_t er;
            if (r < labelGraph.getSize())
                er = labelGraph.getDegree(r);
            else
                er = 0;
            logLikelihoodRatio -= logFactorial(er + dEr) - logFactorial(er);
        }
        return logLikelihoodRatio;
    }

}
