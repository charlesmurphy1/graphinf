#ifndef GRAPH_INF_SINGLE_EDGE_H
#define GRAPH_INF_SINGLE_EDGE_H

#include "GraphInf/exceptions.h"
#include "edge_proposer.h"
#include "GraphInf/graph/proposer/sampler/vertex_sampler.h"
#include "SamplableSet.hpp"
#include "hash_specialization.hpp"
#include "GraphInf/graph/proposer/edge/util.h"

namespace GraphInf
{

    class SingleEdgeProposer : public EdgeProposer
    {
    private:
        mutable std::bernoulli_distribution m_addOrRemoveDistribution = std::bernoulli_distribution(.5);

    protected:
        EdgeSampler m_edgeSampler;
        VertexSampler *m_vertexSamplerPtr = nullptr;

    public:
        SingleEdgeProposer(bool allowSelfLoops = true, bool allowMultiEdges = true) : EdgeProposer(allowSelfLoops, allowMultiEdges), m_edgeSampler(EdgeSampler(1, 100)) {}
        const GraphMove proposeRawMove() const override;
        void setUpWithGraph(const MultiGraph &) override;
        void setEdgeSampler(EdgeSampler &edgeSampler) { m_edgeSampler = edgeSampler; }
        void setVertexSampler(VertexSampler &vertexSampler) { m_vertexSamplerPtr = &vertexSampler; }
        void applyGraphMove(const GraphMove &move) override
        {
            for (auto edge : move.addedEdges)
            {
                m_edgeSampler.onEdgeAddition(edge);
                m_vertexSamplerPtr->onEdgeAddition(edge);
            }
            for (auto edge : move.removedEdges)
            {
                m_edgeSampler.onEdgeRemoval(edge);
                m_vertexSamplerPtr->onEdgeRemoval(edge);
            }
        };
        // void applyBlockMove(const BlockMove& move) override { };

        void checkSelfSafety() const override
        {
            EdgeProposer::checkSelfSafety();
            if (m_graphPtr == nullptr)
                throw SafetyError("SingleEdgeProposer: unsafe proposer since `m_graphPtr` is NULL.");
            if (m_vertexSamplerPtr == nullptr)
                throw SafetyError("SingleEdgeProposer: unsafe proposer since `m_vertexSamplerPtr` is NULL.");
        }
        void clear() override
        {
            m_edgeSampler.clear();
            m_vertexSamplerPtr->clear();
        }
    };

    class SingleEdgeUniformProposer : public SingleEdgeProposer
    {
    private:
        VertexUniformSampler m_vertexUniformSampler;

    public:
        SingleEdgeUniformProposer(bool allowSelfLoops = true, bool allowMultiEdges = true) : SingleEdgeProposer(allowSelfLoops, allowMultiEdges)
        {
            m_vertexSamplerPtr = &m_vertexUniformSampler;
        }
        virtual ~SingleEdgeUniformProposer() {}

        const double getLogProposalProbRatio(const GraphMove &move) const override;
    };

    class SingleEdgeDegreeProposer : public SingleEdgeProposer
    {
    private:
        VertexDegreeSampler m_vertexDegreeSampler;

    public:
        SingleEdgeDegreeProposer(bool allowSelfLoops = true, bool allowMultiEdges = true, double shift = 1) : SingleEdgeProposer(allowSelfLoops, allowMultiEdges),
                                                                                                              m_vertexDegreeSampler(shift) { m_vertexSamplerPtr = &m_vertexDegreeSampler; }

        virtual ~SingleEdgeDegreeProposer() {}

        const double getLogProposalProbRatio(const GraphMove &move) const override;
    };

} // namespace GraphInf

#endif
