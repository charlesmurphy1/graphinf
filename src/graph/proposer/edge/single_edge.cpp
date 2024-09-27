#include "GraphInf/utility/functions.h"
#include "GraphInf/rng.h"
#include "GraphInf/graph/proposer/edge/single_edge.h"

namespace GraphInf
{

    // const GraphMove SingleEdgeProposer::proposeRawMove() const
    // {
    //     if (m_graphPtr->getTotalEdgeNumber() == 0 || m_uniform01(rng) < addEdgeProb(m_graphPtr->getTotalEdgeNumber()))
    //     {
    //         auto vertex1 = m_vertexSamplerPtr->sample();
    //         auto vertex2 = m_vertexSamplerPtr->sample();
    //         BaseGraph::Edge proposedEdge = {vertex1, vertex2};
    //         return {{}, {proposedEdge}};
    //     }
    //     else
    //     {
    //         auto edge = m_edgeSampler.sample();
    //         return {{edge}, {}};
    //     }
    // }

    // void SingleEdgeProposer::setUpWithGraph(const MultiGraph &graph)
    // {
    //     m_graphPtr = &graph;
    //     m_edgeSampler.setUpWithGraph(graph);
    //     m_vertexSamplerPtr->setUpWithGraph(graph);
    // }

    // const double SingleEdgeUniformProposer::getLogProposalProbRatio(const GraphMove &move) const
    // {
    //     // Removing an edge
    //     auto E = (double)m_graphPtr->getTotalEdgeNumber();
    //     auto N = (double)m_graphPtr->getSize();
    //     if (move.removedEdges.size() == 1)
    //     {
    //         double pForward = addEdgeProb(E), pBackward = addEdgeProb(E - 1);

    //         auto edge = move.removedEdges[0];
    //         auto u = edge.first, v = edge.second;
    //         auto w = (double)m_graphPtr->getEdgeMultiplicity(u, v);

    //         auto forwardProb = w / E * (1 - pForward);
    //         auto backwardProb = 1. / N / N * pBackward;
    //         if (E == 1)
    //             backwardProb *= 2;
    //         if (u != v)
    //             backwardProb *= 2;
    //         return log(backwardProb) - log(forwardProb);
    //     }
    //     // Adding an edge
    //     if (move.addedEdges.size() == 1)
    //     {
    //         double pForward = addEdgeProb(E), pBackward = addEdgeProb(E + 1);
    //         auto edge = move.addedEdges[0];
    //         auto u = edge.first, v = edge.second;
    //         auto w = (double)m_graphPtr->getEdgeMultiplicity(u, v);
    //         auto forwardProb = 1. / N / N * pForward;
    //         auto backwardProb = (w + 1) / (E + 1) * (1 - pBackward);
    //         if (E == 0)
    //             forwardProb *= 2;
    //         if (u != v)
    //             forwardProb *= 2;
    //         return log(backwardProb) - log(forwardProb);
    //     }
    //     return 0;
    // }

    // const double SingleEdgeDegreeProposer::getLogProposalProbRatio(const GraphMove &move) const
    // {
    //     // Removing an edge
    //     if (move.removedEdges.size() == 1)
    //     {
    //         auto gamma1 = (m_vertexDegreeSampler.getVertexWeight(move.removedEdges[0].first) + 1) / (m_vertexDegreeSampler.getTotalWeight() + 2);
    //         auto gamma2 = (m_vertexDegreeSampler.getVertexWeight(move.removedEdges[0].second) + 1) / (m_vertexDegreeSampler.getTotalWeight() + 2);
    //         return log(m_edgeSampler.getTotalWeight()) + log(gamma1) + log(gamma2);
    //     }
    //     // Adding an edge
    //     if (move.addedEdges.size() == 1)
    //     {
    //         auto gamma1 = m_vertexDegreeSampler.getVertexWeight(move.addedEdges[0].first) / m_vertexDegreeSampler.getTotalWeight();
    //         auto gamma2 = m_vertexDegreeSampler.getVertexWeight(move.addedEdges[0].second) / m_vertexDegreeSampler.getTotalWeight();
    //         return -log(m_edgeSampler.getTotalWeight() + 1) - log(gamma1) - log(gamma2);
    //     }
    //     return 0;
    // }

} // namespace GraphInf
