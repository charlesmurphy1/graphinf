#include "GraphInf/utility/functions.h"
#include "GraphInf/rng.h"
#include "GraphInf/graph/proposer/edge/single_edge.h"

namespace GraphInf
{

    const GraphMove SingleEdgeProposer::proposeRawMove() const
    {
        if (m_addOrRemoveDistribution(rng) || m_graphPtr->getTotalEdgeNumber() == 0)
        {
            auto vertex1 = m_vertexSamplerPtr->sample();
            auto vertex2 = m_vertexSamplerPtr->sample();
            BaseGraph::Edge proposedEdge = {vertex1, vertex2};
            return {{}, {proposedEdge}};
        }
        else
        {
            auto edge = m_edgeSampler.sample();
            return {{edge}, {}};
        }
    }

    void SingleEdgeProposer::setUpWithGraph(const MultiGraph &graph)
    {
        m_graphPtr = &graph;
        m_edgeSampler.setUpWithGraph(graph);
        m_vertexSamplerPtr->setUpWithGraph(graph);
    }

    const double SingleEdgeUniformProposer::getLogProposalProbRatio(const GraphMove &move) const
    {
        // Removing an edge
        auto E = m_edgeSampler.getTotalWeight();
        auto N = m_vertexUniformSampler.getTotalWeight();
        if (move.removedEdges.size() == 1)
        {
            auto u = move.removedEdges[0].first, v = move.removedEdges[0].second;
            auto w = m_edgeSampler.getEdgeWeight(move.removedEdges[0]);
            auto r = log(E) - log(w) - 2 * log(N);
            if (u != v)
                r += log(2);
            return r;
        }
        // Adding an edge
        if (move.addedEdges.size() == 1)
        {
            auto u = move.addedEdges[0].first, v = move.addedEdges[0].second;
            auto w = m_edgeSampler.getEdgeWeight(move.addedEdges[0]);
            auto r = 2 * log(N) + log(w + 1) - log(E + 1);
            if (u != v)
                r -= log(2);
            return r;
        }
        return 0;
    }

    const double SingleEdgeDegreeProposer::getLogProposalProbRatio(const GraphMove &move) const
    {
        // Removing an edge
        if (move.removedEdges.size() == 1)
        {
            auto gamma1 = (m_vertexDegreeSampler.getVertexWeight(move.removedEdges[0].first) + 1) / (m_vertexDegreeSampler.getTotalWeight() + 2);
            auto gamma2 = (m_vertexDegreeSampler.getVertexWeight(move.removedEdges[0].second) + 1) / (m_vertexDegreeSampler.getTotalWeight() + 2);
            return log(m_edgeSampler.getTotalWeight()) + log(gamma1) + log(gamma2);
        }
        // Adding an edge
        if (move.addedEdges.size() == 1)
        {
            auto gamma1 = m_vertexDegreeSampler.getVertexWeight(move.addedEdges[0].first) / m_vertexDegreeSampler.getTotalWeight();
            auto gamma2 = m_vertexDegreeSampler.getVertexWeight(move.addedEdges[0].second) / m_vertexDegreeSampler.getTotalWeight();
            return -log(m_edgeSampler.getTotalWeight() + 1) - log(gamma1) - log(gamma2);
        }
        return 0;
    }

} // namespace GraphInf
