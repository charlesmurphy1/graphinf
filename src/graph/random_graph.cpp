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
        m_edgeProposerPtr->applyGraphMove(move);
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
        m_edgeProposerPtr->clear();
        m_edgeProposerPtr->setUpWithPrior(*this);
    }

    const double RandomGraph::getLogProposalRatioFromGraphMove(const GraphMove &move) const
    {
        return m_edgeProposerPtr->getLogProposalProbRatio(move);
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
        return m_edgeProposerPtr->proposeMove();
    }

}
