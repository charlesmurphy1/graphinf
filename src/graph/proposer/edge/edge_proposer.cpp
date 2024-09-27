#include "GraphInf/graph/proposer/edge/edge_proposer.h"
#include "GraphInf/graph/random_graph.hpp"
#include "GraphInf/utility/functions.h"

namespace GraphInf
{

    GraphMove EdgeProposer::orderGraphMove(const GraphMove &move) const
    {
        GraphMove orderedMove;
        for (const auto edge : move.addedEdges)
            orderedMove.addedEdges.push_back(getOrderedEdge(edge));
        for (const auto edge : move.removedEdges)
            orderedMove.removedEdges.push_back(getOrderedEdge(edge));
        return orderedMove;
    }

    const GraphMove EdgeProposer::proposeMove() const
    {
        for (size_t i = 0; i < m_maxIteration; i++)
        {
            GraphMove move = proposeRawMove();
            bool isValid = true;
            for (auto e : move.addedEdges)
            {
                if ((isSelfLoop(e) and not m_allowSelfLoops) or (isExistingEdge(e) and not m_allowMultiEdges))
                {
                    isValid = false;
                    break;
                }
            }
            if (not isValid)
                continue;
            return orderGraphMove(move);
        }
        throw std::runtime_error("EdgeProposer: Could not find edge to propose.");
    }

}
