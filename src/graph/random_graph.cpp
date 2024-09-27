#include <algorithm>
#include <chrono>
#include <cmath>
#include <map>
#include <random>
#include <stdexcept>
#include <string>

#include "GraphInf/types.h"
#include "GraphInf/graph/proposer/edge/edge_proposer.h"
#include "GraphInf/graph/random_graph.hpp"

namespace GraphInf
{

    void RandomGraph::_applyGraphMove(const GraphMove &move)
    {
        m_edgeCountPriorPtr->applyGraphMove(move);
        _applyGraphMoveToProposers(move);
        for (auto edge : move.addedEdges)
        {
            auto v = edge.first, u = edge.second;
            m_state.addEdge(v, u);
        }
        for (auto edge : move.removedEdges)
        {
            auto v = edge.first, u = edge.second;
            if (m_state.hasEdge(u, v))
                m_state.removeEdge(v, u);
            else
                throw std::runtime_error("Cannot remove non-existing edge (" + std::to_string(u) + ", " + std::to_string(v) + ").");
        }
    }

    void RandomGraph::setUp()
    {
        m_singleEdgeProposer.clear();
        m_hingeFlipProposer.clear();
        m_doubleEdgeSwapProposer.clear();
        m_singleEdgeProposer.setUpWithGraph(m_state);
        m_hingeFlipProposer.setUpWithGraph(m_state);
        m_doubleEdgeSwapProposer.setUpWithGraph(m_state);
    }

    const double RandomGraph::getLogProposalRatioFromGraphMove(const GraphMove &move) const
    {
        if (move.addedEdges.size() == 0 and move.removedEdges.size() == 0)
            return 0;
        if (move.addedEdges.size() == 1 and move.removedEdges.size() == 0)
            return m_singleEdgeProposer.getLogProposalProbRatio(move);
        if (move.addedEdges.size() == 0 and move.removedEdges.size() == 1)
            return m_singleEdgeProposer.getLogProposalProbRatio(move);
        if (move.addedEdges.size() == 1 and move.removedEdges.size() == 1)
            return m_hingeFlipProposer.getLogProposalProbRatio(move);
        if (move.addedEdges.size() == 2 and move.removedEdges.size() == 2)
            return m_doubleEdgeSwapProposer.getLogProposalProbRatio(move);
        throw std::runtime_error("RandomGraph: Cannot compute proposal ratio for move with move " + move.display() + ".");
    }

    void RandomGraph::applyGraphMove(const GraphMove &move)
    {
        processRecursiveFunction([&]()
                                 { _applyGraphMove(move); });
#if DEBUG
        checkConsistency();
#endif
    }

    const GraphMove RandomGraph::proposeGraphMove() const
    {
        if (m_graphMoveType == "single_edge")
            return m_singleEdgeProposer.proposeMove();
        if (m_graphMoveType == "hinge_flip")
            return m_hingeFlipProposer.proposeMove();
        if (m_graphMoveType == "double_edge_swap")
            return m_doubleEdgeSwapProposer.proposeMove();
        if (m_graphMoveType == "microcanonical")
            return proposeMicrocanonicalMove();
        if (m_graphMoveType == "canonical")
            return proposeCanonicalMove();
        throw std::runtime_error("RandomGraph: Cannot propose move for type " + m_graphMoveType + ".");
    }

}
