#ifndef GRAPHINF_NON_PRESEVING_H
#define GRAPHINF_NON_PRESEVING_H

#include "GraphInf/rng.h"
#include "GraphInf/graph/proposer/edge/edge_proposer.h"
#include "GraphInf/graph/proposer/edge/single_edge.h"
#include "GraphInf/graph/proposer/edge/hinge_flip.h"
#include "GraphInf/graph/proposer/edge/double_edge_swap.h"

namespace GraphInf
{
    class NonPreservingProposer : public EdgeProposer
    {
    private:
    private:
        mutable std::discrete_distribution<int> m_sampleMoveType = std::discrete_distribution<int>({1, 1, 1});
        SingleEdgeUniformProposer m_singleEdgeProposer;
        HingeFlipDegreeProposer m_hingeFlipProposer;
        DoubleEdgeSwapProposer m_doubleEdgeSwapProposer;

    public:
        using EdgeProposer::EdgeProposer;
        const GraphMove proposeRawMove() const override final
        {
            auto choice = m_sampleMoveType(rng);
            if (choice == 0 || m_graphPtr->getTotalEdgeNumber() == 0)
                return m_singleEdgeProposer.proposeRawMove();
            if (choice == 1 || m_graphPtr->getTotalEdgeNumber() == 1)
                return m_hingeFlipProposer.proposeRawMove();
            if (choice == 2)
                return m_doubleEdgeSwapProposer.proposeRawMove();
            return {};
        }
        void setUpWithGraph(const MultiGraph &graph) override
        {
            EdgeProposer::setUpWithGraph(graph);
            m_singleEdgeProposer.setUpWithGraph(graph);
            m_hingeFlipProposer.setUpWithGraph(graph);
            m_doubleEdgeSwapProposer.setUpWithGraph(graph);
        }
        void applyGraphMove(const GraphMove &move) override
        {

            m_singleEdgeProposer.applyGraphMove(move);
            m_hingeFlipProposer.applyGraphMove(move);
            m_doubleEdgeSwapProposer.applyGraphMove(move);
        }
        const double getLogProposalProbRatio(const GraphMove &move) const override
        {
            if ((move.addedEdges.size() == 1 && move.removedEdges.size() == 0) || (move.addedEdges.size() == 0 && move.removedEdges.size() == 1))
                return m_singleEdgeProposer.getLogProposalProbRatio(move);
            if (move.addedEdges.size() == 1 && move.removedEdges.size() == 1)
                return m_hingeFlipProposer.getLogProposalProbRatio(move);
            if (move.addedEdges.size() == 2 && move.removedEdges.size() == 2)
                return m_doubleEdgeSwapProposer.getLogProposalProbRatio(move);
            return 0;
        }
        void checkSelfSafety() const override
        {
            EdgeProposer::checkSelfSafety();
            m_singleEdgeProposer.checkSelfSafety();
            m_hingeFlipProposer.checkSelfSafety();
            m_doubleEdgeSwapProposer.checkSelfSafety();
        }

        void checkSelfConsistency() const override
        {
            m_singleEdgeProposer.checkSelfConsistency();
            m_hingeFlipProposer.checkSelfConsistency();
            m_doubleEdgeSwapProposer.checkSelfConsistency();
        }

        void clear() override
        {
            m_singleEdgeProposer.clear();
            m_hingeFlipProposer.clear();
            m_doubleEdgeSwapProposer.clear();
        }
    };
} // namespace GraphInf

#endif
