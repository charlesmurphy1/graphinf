#include "GraphInf/graph/likelihood/sbm.h"
#include "BaseGraph/types.h"

using namespace BaseGraph;
namespace GraphInf
{

    void StochasticBlockModelLikelihood::getDiffEdgeMatMapFromEdgeMove(const BaseGraph::Edge &edge, int counter, IntMap<std::pair<BlockIndex, BlockIndex>> &diffEdgeMatMap) const
    {
        const BlockSequence &blockSeq = (*m_labelGraphPriorPtrPtr)->getBlockPrior().getState();
        diffEdgeMatMap.increment(
            getOrderedPair<BlockIndex>({blockSeq[edge.first], blockSeq[edge.second]}),
            counter);
    }

    void StochasticBlockModelLikelihood::getDiffAdjMatMapFromEdgeMove(const Edge &edge, int counter, IntMap<std::pair<VertexIndex, VertexIndex>> &diffAdjMatMap) const
    {
        Edge orderedEdge = getOrderedEdge(edge);
        diffAdjMatMap.increment({orderedEdge.first, orderedEdge.second}, counter);
    }

    void StochasticBlockModelLikelihood::getDiffEdgeMatMapFromBlockMove(
        const BlockMove &move, IntMap<std::pair<BlockIndex, BlockIndex>> &diffEdgeMatMap) const
    {
        const BlockSequence &blockSeq = (*m_labelGraphPriorPtrPtr)->getBlockPrior().getState();
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

    const double StubLabeledStochasticBlockModelLikelihood::getLogLikelihoodRatioEdgeTerm(const GraphMove &move) const
    {
        const BlockSequence &blockSeq = (*m_labelGraphPriorPtrPtr)->getBlockPrior().getState();
        const MultiGraph &labelGraph = (*m_labelGraphPriorPtrPtr)->getState();
        const CounterMap<BlockIndex> &edgeCounts = (*m_labelGraphPriorPtrPtr)->getEdgeCounts();
        const CounterMap<BlockIndex> &vertexCounts = (*m_labelGraphPriorPtrPtr)->getBlockPrior().getVertexCounts();
        double logLikelihoodRatioTerm = 0;

        IntMap<std::pair<BlockIndex, BlockIndex>> diffEdgeMatMap;
        IntMap<BlockIndex> diffEdgeCountsMap;

        for (auto edge : move.addedEdges)
            getDiffEdgeMatMapFromEdgeMove(edge, 1, diffEdgeMatMap);
        for (auto edge : move.removedEdges)
            getDiffEdgeMatMapFromEdgeMove(edge, -1, diffEdgeMatMap);

        for (auto diff : diffEdgeMatMap)
        {
            auto r = diff.first.first, s = diff.first.second;

            size_t ers;
            if (r < labelGraph.getSize() and s < labelGraph.getSize())
                ers = labelGraph.getEdgeMultiplicity(r, s);
            else
                ers = 0;
            diffEdgeCountsMap.increment(r, diff.second);
            diffEdgeCountsMap.increment(s, diff.second);
            if (r == s)
            {
                logLikelihoodRatioTerm += logDoubleFactorial(2 * (ers + diff.second));
                logLikelihoodRatioTerm -= logDoubleFactorial(2 * ers);
            }
            else
            {
                logLikelihoodRatioTerm += logFactorial(ers + diff.second);
                logLikelihoodRatioTerm -= logFactorial(ers);
            }
        }

        for (auto diff : diffEdgeCountsMap)
        {
            logLikelihoodRatioTerm -= diff.second * log(vertexCounts[diff.first]);
        }
        return logLikelihoodRatioTerm;
    }

    const double StubLabeledStochasticBlockModelLikelihood::getLogLikelihoodRatioAdjTerm(const GraphMove &move) const
    {
        IntMap<std::pair<VertexIndex, VertexIndex>> diffAdjMatMap;
        double logLikelihoodRatioTerm = 0;

        for (auto edge : move.addedEdges)
            getDiffAdjMatMapFromEdgeMove(edge, 1, diffAdjMatMap);
        for (auto edge : move.removedEdges)
            getDiffAdjMatMapFromEdgeMove(edge, -1, diffAdjMatMap);

        for (auto diff : diffAdjMatMap)
        {
            auto i = diff.first.first, j = diff.first.second;
            auto mult = m_statePtr->getEdgeMultiplicity(i, j);
            if (i == j)
            {
                logLikelihoodRatioTerm += -logDoubleFactorial(2 * (mult + diff.second));
                logLikelihoodRatioTerm -= -logDoubleFactorial(2 * mult);
            }
            else
            {
                logLikelihoodRatioTerm += -logFactorial(mult + diff.second);
                logLikelihoodRatioTerm -= -logFactorial(mult);
            }
        }
        return logLikelihoodRatioTerm;
    }

    const double StubLabeledStochasticBlockModelLikelihood::getLogLikelihood() const
    {

        const MultiGraph &labelGraph = (*m_labelGraphPriorPtrPtr)->getState();
        const CounterMap<BlockIndex> &vertexCounts = (*m_labelGraphPriorPtrPtr)->getBlockPrior().getVertexCounts();

        double logLikelihood = 0;

        for (const auto &r : labelGraph)
        {
            auto er = labelGraph.getDegree(r), nr = vertexCounts.get(r);
            if (er == 0 or nr == 0)
                continue;
            logLikelihood -= er * log(nr);
            for (const auto &s : labelGraph.getOutNeighbours(r))
            {
                const auto ers = labelGraph.getEdgeMultiplicity(r, s);
                if (r > s)
                    continue;
                if (r == s)
                    logLikelihood += logDoubleFactorial(2 * ers);
                else
                    logLikelihood += logFactorial(ers);
            }
        }
        for (const auto &edge : m_statePtr->edges())
        {
            const auto mult = m_statePtr->getEdgeMultiplicity(edge.first, edge.second);
            if (edge.first == edge.second)
                logLikelihood -= logDoubleFactorial(2 * mult);
            else
                logLikelihood -= logFactorial(mult);
        }
        return logLikelihood;
    }

    const double StubLabeledStochasticBlockModelLikelihood::getLogLikelihoodRatioFromGraphMove(const GraphMove &move) const
    {
        return getLogLikelihoodRatioEdgeTerm(move) + getLogLikelihoodRatioAdjTerm(move);
    }

    const double StubLabeledStochasticBlockModelLikelihood::getLogLikelihoodRatioFromLabelMove(const BlockMove &move) const
    {
        if (move.prevLabel == move.nextLabel or move.level > 0)
            return 0;
        const BlockSequence &blockSeq = (*m_labelGraphPriorPtrPtr)->getBlockPrior().getState();
        const MultiGraph &labelGraph = (*m_labelGraphPriorPtrPtr)->getState();
        const CounterMap<BlockIndex> &edgeCounts = (*m_labelGraphPriorPtrPtr)->getEdgeCounts();
        const CounterMap<BlockIndex> &vertexCounts = (*m_labelGraphPriorPtrPtr)->getBlockPrior().getVertexCounts();
        const size_t &degree = m_statePtr->getDegree(move.vertexIndex);
        double logLikelihoodRatio = 0;

        IntMap<std::pair<BlockIndex, BlockIndex>> diffEdgeMatMap;

        getDiffEdgeMatMapFromBlockMove(move, diffEdgeMatMap);

        for (auto diff : diffEdgeMatMap)
        {
            auto r = diff.first.first, s = diff.first.second;
            size_t ers;
            if (r < labelGraph.getSize() and s < labelGraph.getSize())
                ers = labelGraph.getEdgeMultiplicity(r, s);
            else
                ers = 0;
            if (r == s)
            {
                logLikelihoodRatio += logDoubleFactorial(2 * (ers + diff.second));
                logLikelihoodRatio -= logDoubleFactorial(2 * ers);
            }
            else
            {
                logLikelihoodRatio += logFactorial(ers + diff.second);
                logLikelihoodRatio -= logFactorial(ers);
            }
        }

        logLikelihoodRatio += edgeCounts[move.prevLabel] * log(vertexCounts[move.prevLabel]);
        if (vertexCounts.get(move.prevLabel) > 1)
            logLikelihoodRatio -= (edgeCounts[move.prevLabel] - degree) * log(vertexCounts[move.prevLabel] - 1);

        if (vertexCounts.get(move.nextLabel) > 0)
            logLikelihoodRatio += edgeCounts[move.nextLabel] * log(vertexCounts[move.nextLabel]);
        logLikelihoodRatio -= (edgeCounts[move.nextLabel] + degree) * log(vertexCounts[move.nextLabel] + 1);
        return logLikelihoodRatio;
    }

    const double UniformStochasticBlockModelLikelihood::getLogLikelihood() const
    {

        const MultiGraph &labelGraph = (*m_labelGraphPriorPtrPtr)->getState();
        const CounterMap<BlockIndex> &vertexCounts = (*m_labelGraphPriorPtrPtr)->getBlockPrior().getVertexCounts();

        double logLikelihood = 0;

        for (const auto &r : labelGraph)
        {
            if (vertexCounts[r] == 0)
                continue;
            if (vFunc(vertexCounts[r]) < labelGraph.getEdgeMultiplicity(r, r) and not *m_withParallelEdgesPtr)
                return -INFINITY;
            logLikelihood -= logLikelihoodFunc(vertexCounts[r], vertexCounts[r], labelGraph.getEdgeMultiplicity(r, r), true);
            for (const auto &s : labelGraph.getOutNeighbours(r))
            {
                size_t ers = labelGraph.getEdgeMultiplicity(r, s);
                if (r < s)
                {
                    logLikelihood -= logLikelihoodFunc(vertexCounts[r], vertexCounts[s], ers, false);
                }
            }
        }
        return logLikelihood;
    }

    const double UniformStochasticBlockModelLikelihood::getLogLikelihoodRatioFromGraphMove(const GraphMove &move) const
    {
        const MultiGraph &labelGraph = (*m_labelGraphPriorPtrPtr)->getState();
        const CounterMap<BlockIndex> &vertexCounts = (*m_labelGraphPriorPtrPtr)->getBlockPrior().getVertexCounts();
        double logLikelihoodRatio = 0;

        IntMap<std::pair<BlockIndex, BlockIndex>> diffEdgeMatMap;

        for (auto edge : move.addedEdges)
            getDiffEdgeMatMapFromEdgeMove(edge, 1, diffEdgeMatMap);
        for (auto edge : move.removedEdges)
            getDiffEdgeMatMapFromEdgeMove(edge, -1, diffEdgeMatMap);

        for (auto diff : diffEdgeMatMap)
        {
            auto r = diff.first.first, s = diff.first.second;
            size_t nr = vertexCounts[r], ns = vertexCounts[s];
            size_t ers;
            if (r < labelGraph.getSize() and s < labelGraph.getSize())
                ers = labelGraph.getEdgeMultiplicity(r, s);
            else
                ers = 0;
            logLikelihoodRatio -= logLikelihoodFunc(nr, ns, ers + diff.second, r == s);
            logLikelihoodRatio += logLikelihoodFunc(nr, ns, ers, r == s);
        }

        return logLikelihoodRatio;
    }

    const double UniformStochasticBlockModelLikelihood::getLogLikelihoodRatioFromLabelMove(const BlockMove &move) const
    {
        if (move.prevLabel == move.nextLabel or move.level > 0)
            return 0;
        const MultiGraph &labelGraph = (*m_labelGraphPriorPtrPtr)->getState();
        const CounterMap<BlockIndex> &vertexCounts = (*m_labelGraphPriorPtrPtr)->getBlockPrior().getVertexCounts();

        double logLikelihoodRatio = 0;

        IntMap<BlockIndex> vDiffMap;
        vDiffMap.decrement(move.prevLabel);
        vDiffMap.increment(move.nextLabel);

        IntMap<std::pair<BlockIndex, BlockIndex>> eDiffMap;
        getDiffEdgeMatMapFromBlockMove(move, eDiffMap);
        for (auto diff : eDiffMap)
        {
            auto r = diff.first.first, s = diff.first.second;
            size_t nr = vertexCounts[r], ns = vertexCounts[s], dnr = vDiffMap.get(r), dns = vDiffMap.get(s);
            size_t ers;
            if (r < labelGraph.getSize() and s < labelGraph.getSize())
                ers = labelGraph.getEdgeMultiplicity(r, s);
            else
                ers = 0;
            if (r == s)
            {
                logLikelihoodRatio -= logLikelihoodFunc((nr + dnr), (ns + dns), ers + diff.second, true);
                logLikelihoodRatio += logLikelihoodFunc(nr, ns, ers, true);
            }
            else
            {
                logLikelihoodRatio -= logLikelihoodFunc((nr + dnr), (ns + dns), ers + diff.second, false);
                logLikelihoodRatio += logLikelihoodFunc(nr, ns, ers, false);
            }
        }

        // remaining contributions that did not change the edge counts
        std::set<BaseGraph::Edge> visited;
        for (const auto &diff : vDiffMap)
        {
            if (diff.first >= labelGraph.getSize())
                continue;
            for (const auto &s : labelGraph.getOutNeighbours(diff.first))
            {
                BlockIndex r = diff.first;
                // if not empty, the term has been processed in the edgeDiff loop
                auto rs = getOrderedEdge({r, s});
                if (visited.count(rs) > 0 or not eDiffMap.isEmpty(rs))
                    continue;
                visited.insert(rs);
                size_t nr = vertexCounts[r], ns = vertexCounts[s], dnr = vDiffMap.get(r), dns = vDiffMap.get(s);
                int ers = labelGraph.getEdgeMultiplicity(r, s);
                if (r == s)
                {
                    logLikelihoodRatio -= logLikelihoodFunc((nr + dnr), (ns + dns), ers, true);
                    logLikelihoodRatio += logLikelihoodFunc(nr, ns, ers, true);
                }
                else
                {
                    logLikelihoodRatio -= logLikelihoodFunc((nr + dnr), (ns + dns), ers, false);
                    logLikelihoodRatio += logLikelihoodFunc(nr, ns, ers, false);
                }
            }
        }

        return logLikelihoodRatio;
    }

}
