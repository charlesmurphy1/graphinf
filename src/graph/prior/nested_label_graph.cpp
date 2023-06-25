#include "GraphInf/graph/prior/nested_label_graph.h"

namespace GraphInf
{

    void NestedLabelGraphPrior::applyGraphMoveToState(const GraphMove &move)
    {
        LabelGraphPrior::applyGraphMoveToState(move);
        BlockIndex r, s;
        for (auto removedEdge : move.removedEdges)
        {
            r = (BlockIndex)removedEdge.first;
            s = (BlockIndex)removedEdge.second;

            for (Level l = 0; l < getDepth(); ++l)
            {
                r = m_nestedBlockPriorPtr->getNestedState(l)[r];
                s = m_nestedBlockPriorPtr->getNestedState(l)[s];
                m_nestedState[l].removeEdge(r, s);
                m_nestedEdgeCounts[l].decrement(r);
                m_nestedEdgeCounts[l].decrement(s);
            }
        }
        for (auto addedEdge : move.addedEdges)
        {
            r = (BlockIndex)addedEdge.first;
            s = (BlockIndex)addedEdge.second;
            for (Level l = 0; l < getDepth(); ++l)
            {
                r = m_nestedBlockPriorPtr->getNestedState(l)[r];
                s = m_nestedBlockPriorPtr->getNestedState(l)[s];
                m_nestedState[l].addEdge(r, s);
                m_nestedEdgeCounts[l].increment(r);
                m_nestedEdgeCounts[l].increment(s);
            }
        }
    }

    void NestedLabelGraphPrior::applyLabelMoveToState(const BlockMove &move)
    {
        // if (m_nestedBlockPriorPtr->destroyingBlock(move))
        //     return;

        if (move.level == 0)
            LabelGraphPrior::applyLabelMoveToState(move);

        // move creating new label
        if (m_nestedState[move.level].getSize() <= move.nextLabel)
        {
            if (move.level == m_nestedState.size() - 1)
            {
                m_nestedState.push_back(LabelGraph(0));
                m_nestedState[move.level + 1].resize(1);
                m_nestedState[move.level + 1].addMultiedge(0, 0, getEdgeCount());

                m_nestedEdgeCounts.push_back({});
                m_nestedEdgeCounts[move.level + 1].increment(0, 2 * getEdgeCount());
            }
            m_nestedState[move.level].resize(move.nextLabel + 1);
        }

        BlockIndex vertexIndex = getBlock(move.vertexIndex, move.level - 1);
        const LabelGraph &graph = getNestedState(move.level - 1);
        const BlockSequence &blocks = getNestedBlocks(move.level);
        const auto &degree = graph.getDegree(vertexIndex);

        m_nestedEdgeCounts[move.level].decrement(move.prevLabel, degree);
        m_nestedEdgeCounts[move.level].increment(move.nextLabel, degree);

        for (auto neighbor : graph.getOutNeighbours(vertexIndex))
        {
            auto neighborBlock = blocks[neighbor];
            const auto mult = graph.getEdgeMultiplicity(vertexIndex, neighbor);

            if (vertexIndex == neighbor) // for self-loops
                neighborBlock = move.prevLabel;
            m_nestedState[move.level].removeMultiedge(move.prevLabel, neighborBlock, mult);

            if (vertexIndex == neighbor) // for self-loops
                neighborBlock = move.nextLabel;
            m_nestedState[move.level].addMultiedge(move.nextLabel, neighborBlock, mult);
        }
    }

    void NestedLabelGraphPrior::recomputeConsistentState()
    {
        m_nestedEdgeCounts.clear();
        m_nestedEdgeCounts.resize(getDepth(), {});
        for (Level l = 0; l < getDepth(); ++l)
            for (auto r : m_nestedState[l])
                m_nestedEdgeCounts[l].set(r, m_nestedState[l].getDegree(r));
        m_edgeCounts = m_nestedEdgeCounts[0];
        m_edgeCountPriorPtr->setState(m_state.getTotalEdgeNumber());
    }

    void NestedLabelGraphPrior::recomputeStateFromGraph()
    {
        BlockIndex r, s;
        const LabelGraph *graphPtr = nullptr;
        std::vector<LabelGraph> nestedState;
        if (m_graphPtr->getSize() != m_nestedBlockPriorPtr->getSize())
        {
            m_nestedBlockPriorPtr->setSize(m_graphPtr->getSize());
            m_nestedBlockPriorPtr->sample();
        }
        for (Level l = 0; l < getDepth(); ++l)
        {
            nestedState.push_back(MultiGraph(m_nestedBlockPriorPtr->getNestedMaxBlockCount(l)));
            graphPtr = (l == 0) ? m_graphPtr : &nestedState[l - 1];

            for (const auto &edge : graphPtr->edges())
            {
                const auto mult = graphPtr->getEdgeMultiplicity(edge.first, edge.second);
                r = m_nestedBlockPriorPtr->getNestedState(l)[edge.first];
                s = m_nestedBlockPriorPtr->getNestedState(l)[edge.second];
                nestedState[l].addMultiedge(r, s, mult);
            }
        }

        setNestedState(nestedState);
    }

    void NestedLabelGraphPrior::updateNestedEdgeDiffFromEdge(
        const BaseGraph::Edge &edge, std::vector<IntMap<BaseGraph::Edge>> &nestedEdgeDiff, int counter) const
    {
        size_t r = edge.first, s = edge.second;
        for (Level l = 0; l < getDepth(); ++l)
        {
            r = getNestedBlock(r, l);
            s = getNestedBlock(s, l);
            nestedEdgeDiff[l].increment({r, s}, counter);
        }
    }

    void NestedLabelGraphPrior::sampleState()
    {
        m_nestedState = std::vector<LabelGraph>(getDepth());
        for (Level l = getDepth() - 1; l >= 0; --l)
            m_nestedState[l] = sampleState(l);
        m_nestedEdgeCounts = computeNestedEdgeCountsFromNestedState(m_nestedState);
        m_state = m_nestedState[0];
        m_edgeCounts = m_nestedEdgeCounts[0];
    }

    const double NestedLabelGraphPrior::getLogLikelihood() const
    {
        double logLikelihood = 0;
        for (Level l = 0; l < getDepth(); ++l)
            logLikelihood += getLogLikelihoodAtLevel(l);
        return logLikelihood;
    }

    void NestedLabelGraphPrior::checkSelfConsistencyBetweenLevels() const
    {
        BlockIndex r, s;
        std::string prefix;
        if (m_nestedState[0] != m_state)
            ConsistencyError(
                "NestedLabelGraphPrior", "m_nestedState[0]", "m_state");
        for (const auto &er : m_nestedEdgeCounts[0])
        {
            if (er.second != m_edgeCounts[er.first])
                ConsistencyError(
                    "NestedLabelGraphPrior", "m_nestedEdgeCounts[0]", std::to_string(er.second),
                    "m_edgeCounts", std::to_string(m_edgeCounts[er.first]),
                    "r=" + std::to_string(er.second));
        }
        for (Level l = 1; l < getDepth(); ++l)
        {
            const LabelGraph &graph = getNestedState(l - 1);
            LabelGraph actualLabelGraph(getNestedState(l).getSize());
            for (const auto &edge : graph.edges())
            {
                r = m_nestedBlockPriorPtr->getNestedState(l)[edge.first];
                s = m_nestedBlockPriorPtr->getNestedState(l)[edge.second];
                const auto mult = graph.getEdgeMultiplicity(edge.first, edge.second);

                actualLabelGraph.addMultiedge(r, s, mult);
            }
            prefix = "NestedLabelGraphPrior (level=" + std::to_string(l) + ")";
            for (const auto &edge : m_nestedState[l].edges())
            {
                BaseGraph::VertexIndex vertex = edge.first, neighbor = edge.second;
                const auto mult = m_nestedState[l].getEdgeMultiplicity(vertex, neighbor);
                if (actualLabelGraph.getEdgeMultiplicity(vertex, neighbor) != mult)
                    throw ConsistencyError(
                        prefix,
                        "m_nestedState[l-1]", "edgeCount=" + std::to_string(actualLabelGraph.getEdgeMultiplicity(vertex, neighbor)),
                        "m_nestedState[l]", "edgeCount=" + std::to_string(mult),
                        "(r=" + std::to_string(vertex) + ", s=" + std::to_string(neighbor) + ")");

                if (actualLabelGraph.getDegree(vertex) != m_nestedEdgeCounts[l][vertex])
                    throw ConsistencyError(
                        prefix,
                        "m_nestedState[l-1]", "degree=" + std::to_string(actualLabelGraph.getDegree(vertex)),
                        "m_nestedEdgeCounts[l]", "value=" + std::to_string(m_nestedEdgeCounts[l][vertex]),
                        "r=" + std::to_string(vertex));

                if (actualLabelGraph.getDegree(vertex) != m_nestedState[l].getDegree(vertex))
                    throw ConsistencyError(
                        prefix,
                        "m_nestedState[l-1]", "degree=" + std::to_string(actualLabelGraph.getDegree(vertex)),
                        "m_nestedState[l]", "degree=" + std::to_string(m_nestedState[l].getDegree(vertex)),
                        "r=" + std::to_string(vertex));
            }
        }
    }

    const LabelGraph NestedStochasticBlockLabelGraphPrior::sampleState(Level level) const
    {
        BlockSequence blocks;
        if (level == getDepth() - 1)
            blocks.push_back(0);
        else
            blocks = getNestedBlocks(level + 1);

        std::map<std::pair<BlockIndex, BlockIndex>, std::vector<BaseGraph::Edge>> allLabeledEdges;
        for (auto nr : m_nestedBlockPriorPtr->getNestedAbsVertexCounts(level))
        {
            if (nr.second == 0)
                continue;
            for (auto ns : m_nestedBlockPriorPtr->getNestedAbsVertexCounts(level))
            {
                if (ns.second == 0 or nr.first > ns.first)
                    continue;
                auto rs = getOrderedPair<BlockIndex>({blocks[nr.first], blocks[ns.first]});
                allLabeledEdges[rs].push_back({nr.first, ns.first});
            }
        }

        LabelGraph graph(m_nestedBlockPriorPtr->getNestedBlockCount(level));
        for (const auto &labeledEdges : allLabeledEdges)
        {
            BlockIndex r = labeledEdges.first.first, s = labeledEdges.first.second;
            size_t ers;
            if (level == getDepth() - 1)
                ers = getEdgeCount();
            else
                ers = getNestedState(level + 1).getEdgeMultiplicity(r, s);
            auto flatMultiplicity = sampleRandomWeakComposition(ers, labeledEdges.second.size());
            size_t counter = 0;
            for (const auto &m : flatMultiplicity)
            {
                if (m != 0)
                    graph.addMultiedge(labeledEdges.second[counter].first, labeledEdges.second[counter].second, m);
                ++counter;
            }
        }
        return graph;
    }

    const double NestedStochasticBlockLabelGraphPrior::getLogLikelihoodAtLevel(Level level) const
    {
        double logLikelihood = 0;
        size_t nr, ns, label;
        for (const auto &r : getNestedState(level))
        {
            nr = getNestedVertexCounts(level)[r];
            label = getNestedState(level).getEdgeMultiplicity(r, r);
            logLikelihood -= logMultisetCoefficient(nr * (nr + 1) / 2, label);
            for (const auto &s : getNestedState(level))
            {
                if (r >= s)
                    continue;
                ns = getNestedVertexCounts(level)[s];
                label = getNestedState(level).getEdgeMultiplicity(r, s);
                logLikelihood -= logMultisetCoefficient(nr * ns, label);
            }
        }
        return logLikelihood;
    }

    const double NestedStochasticBlockLabelGraphPrior::getLogLikelihoodRatioFromGraphMove(const GraphMove &move) const
    {
        std::vector<IntMap<BaseGraph::Edge>> nestedEdgeDiff(getDepth());
        for (auto edge : move.addedEdges)
            updateNestedEdgeDiffFromEdge(edge, nestedEdgeDiff, 1);
        for (auto edge : move.removedEdges)
            updateNestedEdgeDiffFromEdge(edge, nestedEdgeDiff, -1);

        double logLikelihoodRatio = 0;
        for (Level l = 0; l < getDepth(); ++l)
        {
            size_t vTerm, eTermBefore, eTermAfter, nr, ns;
            for (auto diff : nestedEdgeDiff[l])
            {
                BlockIndex r = diff.first.first, s = diff.first.second;
                nr = getNestedVertexCounts(l)[r], ns = getNestedVertexCounts(l)[s];
                if (r == s)
                    vTerm = nr * (nr + 1) / 2;
                else
                    vTerm = nr * ns;
                eTermBefore = getNestedState(l).getEdgeMultiplicity(r, s);
                eTermAfter = getNestedState(l).getEdgeMultiplicity(r, s) + diff.second;

                logLikelihoodRatio -= logMultisetCoefficient(vTerm, eTermAfter) - logMultisetCoefficient(vTerm, eTermBefore);
            }
        }
        return logLikelihoodRatio;
    }

    const double NestedStochasticBlockLabelGraphPrior::getLogLikelihoodRatioFromLabelMove(const BlockMove &move) const
    {
        BlockIndex nestedIndex = m_nestedBlockPriorPtr->getBlock(move.vertexIndex, move.level - 1);
        IntMap<BaseGraph::Edge> edgeDiff;
        IntMap<BlockIndex> vertexDiff;
        vertexDiff.decrement(move.prevLabel);
        vertexDiff.increment(move.nextLabel);

        for (auto neighbor : getNestedState(move.level - 1).getOutNeighbours(nestedIndex))
        {
            const auto mult = getNestedState(move.level - 1).getEdgeMultiplicity(nestedIndex, neighbor);
            BlockIndex s = m_nestedBlockPriorPtr->getNestedBlock(neighbor, move.level);
            if (neighbor == nestedIndex)
                s = move.prevLabel;
            edgeDiff.decrement(getOrderedEdge({move.prevLabel, s}), mult);
            if (neighbor == nestedIndex)
                s = move.nextLabel;
            edgeDiff.increment(getOrderedEdge({move.nextLabel, s}), mult);
        }
        double logLikelihoodRatio = 0;
        size_t vTermBefore, vTermAfter, eTermBefore, eTermAfter, nr, ns, edgeMult;
        BlockIndex r, s;

        // contributions that changed the edge counts
        for (auto diff : edgeDiff)
        {
            r = diff.first.first, s = diff.first.second;
            nr = getNestedVertexCounts(move.level)[r], ns = getNestedVertexCounts(move.level)[s];
            vTermBefore = (r == s) ? nr * (nr + 1) / 2 : nr * ns;
            vTermAfter = (r == s) ? (nr + vertexDiff.get(r)) * (nr + vertexDiff.get(r) + 1) / 2 : (nr + vertexDiff.get(r)) * (ns + vertexDiff.get(s));
            if (r == s)
            {
                vTermBefore = nr * (nr + 1) / 2;
                vTermAfter = (nr + vertexDiff.get(r)) * (nr + vertexDiff.get(r) + 1) / 2;
            }
            else
            {
                vTermBefore = nr * ns;
                vTermAfter = (nr + vertexDiff.get(r)) * (ns + vertexDiff.get(s));
            }
            size_t ers = 0;
            if (r < getNestedState(move.level).getSize() and s < getNestedState(move.level).getSize())
                ers = getNestedState(move.level).getEdgeMultiplicity(r, s);
            eTermBefore = ers;
            eTermAfter = ers + diff.second;

            logLikelihoodRatio -= logMultisetCoefficient(vTermAfter, eTermAfter) - logMultisetCoefficient(vTermBefore, eTermBefore);
        }

        // remaining contributions that did not change the edge counts
        std::set<BaseGraph::Edge> visited;
        for (const auto &diff : vertexDiff)
        {
            r = diff.first;
            if (move.addedLabels == 1 and r == move.nextLabel)
                continue;
            for (const auto &s : getNestedState(move.level).getOutNeighbours(r))
            {
                const auto ers = getNestedState(move.level).getEdgeMultiplicity(r, s);
                // if not empty, the term has been processed in the edgeDiff loop
                auto rs = getOrderedEdge({r, s});
                if (visited.count(rs) > 0 or not edgeDiff.isEmpty(rs))
                    continue;
                visited.insert(rs);
                nr = getNestedVertexCounts(move.level)[r], ns = getNestedVertexCounts(move.level)[s];
                vTermBefore = (r == s) ? nr * (nr + 1) / 2 : nr * ns;
                vTermAfter = (r == s) ? (nr + vertexDiff.get(r)) * (nr + vertexDiff.get(r) + 1) / 2 : (nr + vertexDiff.get(r)) * (ns + vertexDiff.get(s));
                logLikelihoodRatio -= logMultisetCoefficient(vTermAfter, ers) - logMultisetCoefficient(vTermBefore, ers);
            }
        }

        if (move.addedLabels == 1)
        {
            // if adding new label not in last layer
            if (move.level < getDepth() - 1)
            {
                r = getNestedBlock(move.prevLabel, move.level + 1);
                nr = getNestedVertexCounts(move.level + 1)[r];
                for (const auto &s : getNestedState(move.level + 1).getOutNeighbours(r))
                {
                    ns = getNestedVertexCounts(move.level + 1)[s];
                    const auto ers = getNestedState(move.level + 1).getEdgeMultiplicity(r, s);
                    vTermBefore = (r == s) ? nr * (nr + 1) / 2 : nr * ns;
                    vTermAfter = (r == s) ? (nr + 1) * (nr + 2) / 2 : (nr + 1) * ns;
                    logLikelihoodRatio -= logMultisetCoefficient(vTermAfter, ers) - logMultisetCoefficient(vTermBefore, ers);
                }
                // if adding new label in last layer
            }
            else
            {
                nr = getNestedVertexCounts(move.level).size() + 1;
                logLikelihoodRatio -= logMultisetCoefficient(nr * (nr + 1) / 2, getEdgeCount());
            }
        }

        return logLikelihoodRatio;
    }

}
