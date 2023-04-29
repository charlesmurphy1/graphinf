#ifndef GRAPH_INF_DOUBLE_EDGE_SWAP_H
#define GRAPH_INF_DOUBLE_EDGE_SWAP_H

#include "edge_proposer.h"
#include "GraphInf/graph/proposer/sampler/edge_sampler.h"
#include "SamplableSet.hpp"
#include "hash_specialization.hpp"

namespace GraphInf
{

    class DoubleEdgeSwapProposer : public EdgeProposer
    {
    private:
        mutable std::bernoulli_distribution m_swapOrientationDistribution = std::bernoulli_distribution(.5);
        bool isTrivialMove(const GraphMove &) const;
        bool isHingeMove(const GraphMove &) const;
        const double getLogPropForNormalMove(const GraphMove &move) const;
        const double getLogPropForDoubleLoopyMove(const GraphMove &move) const;
        const double getLogPropForDoubleEdgeMove(const GraphMove &move) const;

    protected:
        mutable EdgeSampler m_edgeSampler;

    public:
        using EdgeProposer::EdgeProposer;
        const GraphMove proposeRawMove() const override;
        void setUpWithGraph(const MultiGraph &) override;
        const double getLogProposalProbRatio(const GraphMove &move) const override;

        void applyGraphMove(const GraphMove &) override;
        void clear() override { m_edgeSampler.clear(); }
    };

} // namespace GraphInf

#endif
