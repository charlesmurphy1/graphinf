#ifndef GRAPHINF_EDGECOUNT_PRESEVING_H
#define GRAPHINF_EDGECOUNT_PRESEVING_H

#include "GraphInf/rng.h"
#include "GraphInf/graph/proposer/edge/edge_proposer.h"
#include "GraphInf/graph/proposer/edge/hinge_flip.h"
#include "GraphInf/graph/proposer/edge/double_edge_swap.h"

namespace GraphInf
{
    class EdgeCountPreservingProposer : public EdgeProposer
    {
    private:
    private:
        mutable std::bernoulli_distribution m_sampleHingeFlip = std::bernoulli_distribution(.5);
        HingeFlipDegreeProposer m_hingeFlipProposer;
        DoubleEdgeSwapProposer m_doubleEdgeSwapProposer;

    public:
        using EdgeProposer::EdgeProposer;
        const GraphMove proposeRawMove() const override final
        {
            if (m_graphPtr->getTotalEdgeNumber() == 0)
                return {};
            if (m_sampleHingeFlip(rng) || m_graphPtr->getTotalEdgeNumber() < 2)
                return m_hingeFlipProposer.proposeRawMove();
            return m_doubleEdgeSwapProposer.proposeRawMove();
        }
        void setUpWithGraph(const MultiGraph &graph) override
        {
            EdgeProposer::setUpWithGraph(graph);
            m_hingeFlipProposer.setUpWithGraph(graph);
            m_doubleEdgeSwapProposer.setUpWithGraph(graph);
        }
        void applyGraphMove(const GraphMove &move) override
        {
            m_hingeFlipProposer.applyGraphMove(move);
            m_doubleEdgeSwapProposer.applyGraphMove(move);
        }
        const double getLogProposalProbRatio(const GraphMove &move) const override
        {
            if (move.addedEdges.size() == 1 && move.removedEdges.size() == 1)
                return m_hingeFlipProposer.getLogProposalProbRatio(move);
            if (move.addedEdges.size() == 2 && move.removedEdges.size() == 2)
                return m_doubleEdgeSwapProposer.getLogProposalProbRatio(move);
            return 0;
        }
        void checkSelfSafety() const override
        {
            EdgeProposer::checkSelfSafety();
            m_hingeFlipProposer.checkSelfSafety();
            m_doubleEdgeSwapProposer.checkSelfSafety();
        }

        void checkSelfConsistency() const override
        {
            m_hingeFlipProposer.checkSelfConsistency();
            m_doubleEdgeSwapProposer.checkSelfConsistency();
        }

        void clear() override
        {
            m_hingeFlipProposer.clear();
            m_doubleEdgeSwapProposer.clear();
        }
    };
} // namespace GraphInf

#endif
