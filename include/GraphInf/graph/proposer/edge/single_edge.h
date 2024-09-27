#ifndef GRAPH_INF_SINGLE_EDGE_H
#define GRAPH_INF_SINGLE_EDGE_H

#include "GraphInf/exceptions.h"
#include "edge_proposer.h"
#include "GraphInf/graph/proposer/sampler/vertex_sampler.h"
#include "SamplableSet.hpp"
#include "hash_specialization.hpp"
#include "GraphInf/graph/proposer/edge/util.h"
#include "BaseGraph/types.h"

namespace GraphInf
{
    class SingleEdgeProposer : public EdgeProposer
    {
    private:
        EdgeSampler m_edgeSampler;
        double m_sampleNewEdgeProb;
        mutable std::uniform_real_distribution<double> m_uniform01 = std::uniform_real_distribution<double>(0, 1);

    public:
        SingleEdgeProposer(std::map<BaseGraph::Edge, double> weights, double sampleNewEdgeProb = 0.5, bool withSelfLoops = true, bool withMultiEdges = true) : EdgeProposer(withSelfLoops, withMultiEdges), m_sampleNewEdgeProb(sampleNewEdgeProb)
        {
            setWeights(weights);
        }

        SingleEdgeProposer(std::vector<std::vector<double>> weights, double sampleNewEdgeProb = 0.5, bool withSelfLoops = true, bool withMultiEdges = true) : EdgeProposer(withSelfLoops, withMultiEdges), m_sampleNewEdgeProb(sampleNewEdgeProb)
        {
            setWeights(weights);
        }
        SingleEdgeProposer(size_t size, double sampleNewEdgeProb = 0.5, bool withSelfLoops = true, bool withMultiEdges = true) : EdgeProposer(withSelfLoops, withMultiEdges), m_sampleNewEdgeProb(sampleNewEdgeProb)
        {
            setDefaultWeights(size);
        }

        using EdgeProposer::setUpWithGraph;
        const EdgeSampler &getEdgeSampler() const { return m_edgeSampler; }
        void setDefaultWeights(size_t size)
        {
            m_edgeSampler = EdgeSampler(1, 100);
            for (size_t i = 0; i < size; i++)
            {
                for (size_t j = i + 1; j < size; j++)
                {
                    m_edgeSampler.onEdgeInsertion({i, j}, 1);
                }
            }
        }
        void setWeights(std::map<BaseGraph::Edge, double> weights)
        {
            // weights must be between 1 and 100
            m_edgeSampler = EdgeSampler(1, 100);
            for (auto &edge : weights)
            {
                if (edge.second > 100 || edge.second < 1)
                    throw std::invalid_argument("SingleEdgeProposer: weights (" + std::to_string(edge.first.first) + ", " + std::to_string(edge.first.first) + " must be between 1 and 100.");
                m_edgeSampler.onEdgeInsertion(edge.first, edge.second);
            }
        }
        void setWeights(std::vector<std::vector<double>> weights)
        {
            // weights must be between 1 and 100
            m_edgeSampler = EdgeSampler(1, 100);
            for (size_t i = 0; i < weights.size(); i++)
            {
                for (size_t j = i + 1; j < weights[i].size(); j++)
                {
                    if (weights[i][j] > 100 || weights[i][j] < 1)
                        throw std::invalid_argument("SingleEdgeProposer: weights (" + std::to_string(i) + ", " + std::to_string(j) + " must be between 1 and 100.");
                    m_edgeSampler.onEdgeInsertion({i, j}, weights[i][j]);
                }
            }
        }
        void updateWeight(BaseGraph::Edge edge, double weight)
        {
            m_edgeSampler.onEdgeErasure(edge);
            m_edgeSampler.onEdgeInsertion(edge, weight);
        }
        const GraphMove proposeRawMove() const override
        {
            BaseGraph::Edge potentialEdge = m_edgeSampler.sample();

            if ((m_uniform01(rng) < m_sampleNewEdgeProb) || m_graphPtr->getEdgeMultiplicity(potentialEdge.first, potentialEdge.second) == 0)
            {
                return {{}, {potentialEdge}};
            }
            return {{potentialEdge}, {}};
        }
        const double getForwardProposalProb(const GraphMove &move) const
        {
            if (move.addedEdges.size() == 1)
            {
                return m_sampleNewEdgeProb * m_edgeSampler.getEdgeWeight(move.addedEdges[0]) / m_edgeSampler.getTotalWeight();
            }
            return (1 - m_sampleNewEdgeProb) * m_edgeSampler.getEdgeWeight(move.removedEdges[0]) / m_edgeSampler.getTotalWeight();
        }
        const double getLogProposalProbRatio(const GraphMove &move) const override
        {
            return std::log(getForwardProposalProb(getReverseMove(move))) - std::log(getForwardProposalProb(move));
        }
    };

    // class SingleEdgeProposer : public EdgeProposer
    // {
    // private:
    //     mutable std::uniform_real_distribution<double> m_uniform01 = std::uniform_real_distribution<double>(0, 1);

    // protected:
    //     EdgeSampler m_edgeSampler;
    //     VertexSampler *m_vertexSamplerPtr = nullptr;
    //     double m_bias, m_minAddEdgeProb, m_addEdgeProb;
    //     double addEdgeProb(const int edgeCount) const
    //     {
    //         if (m_addEdgeProb > 0)
    //             return m_addEdgeProb;
    //         double avgDegree = 2.0 * edgeCount / m_graphPtr->getSize();
    //         double p = 1.0 / 2.0 / (m_bias * avgDegree + 1.0);
    //         if (p < m_minAddEdgeProb)
    //             return m_minAddEdgeProb;
    //         return p;
    //     }

    // public:
    //     SingleEdgeProposer(bool allowSelfLoops = true, bool allowMultiEdges = true, double bias = 1., double minAddEdgeProb = 0.1, double addEdgeProb = -1) : EdgeProposer(allowSelfLoops, allowMultiEdges), m_edgeSampler(EdgeSampler(1, 100)), m_bias(bias), m_minAddEdgeProb(minAddEdgeProb), m_addEdgeProb(addEdgeProb) {}
    //     const GraphMove proposeRawMove() const override;
    //     void setUpWithGraph(const MultiGraph &) override;
    //     void setEdgeSampler(EdgeSampler &edgeSampler) { m_edgeSampler = edgeSampler; }
    //     EdgeSampler &getEdgeSampler() { return m_edgeSampler; }
    //     void setVertexSampler(VertexSampler &vertexSampler) { m_vertexSamplerPtr = &vertexSampler; }
    //     VertexSampler &getVertexSampler() { return *m_vertexSamplerPtr; }
    //     void setBias(double bias) { m_bias = bias; }
    //     double getBias() const { return m_bias; }
    //     void setMinAddEdgeProb(double minAddEdgeProb) { m_minAddEdgeProb = minAddEdgeProb; }
    //     double getMinAddEdgeProb() const { return m_minAddEdgeProb; }
    //     void setAddEdgeProb(double addEdgeProb) { m_addEdgeProb = addEdgeProb; }
    //     double getAddEdgeProb() const { return addEdgeProb(m_graphPtr->getTotalEdgeNumber()); }
    //     void applyGraphMove(const GraphMove &move) override
    //     {
    //         for (auto edge : move.addedEdges)
    //         {
    //             m_edgeSampler.onEdgeAddition(edge);
    //             m_vertexSamplerPtr->onEdgeAddition(edge);
    //         }
    //         for (auto edge : move.removedEdges)
    //         {
    //             m_edgeSampler.onEdgeRemoval(edge);
    //             m_vertexSamplerPtr->onEdgeRemoval(edge);
    //         }
    //     };
    //     // void applyBlockMove(const BlockMove& move) override { };

    //     void checkSelfSafety() const override
    //     {
    //         EdgeProposer::checkSelfSafety();
    //         if (m_graphPtr == nullptr)
    //             throw SafetyError("SingleEdgeProposer: unsafe proposer since `m_graphPtr` is NULL.");
    //         if (m_vertexSamplerPtr == nullptr)
    //             throw SafetyError("SingleEdgeProposer: unsafe proposer since `m_vertexSamplerPtr` is NULL.");
    //     }
    //     void clear() override
    //     {
    //         m_edgeSampler.clear();
    //         m_vertexSamplerPtr->clear();
    //     }
    // };

    // class SingleEdgeUniformProposer : public SingleEdgeProposer
    // {
    // private:
    //     VertexUniformSampler m_vertexUniformSampler;

    // public:
    //     SingleEdgeUniformProposer(bool allowSelfLoops = true, bool allowMultiEdges = true, double bias = 1.0) : SingleEdgeProposer(allowSelfLoops, allowMultiEdges, bias)
    //     {
    //         m_vertexSamplerPtr = &m_vertexUniformSampler;
    //     }
    //     virtual ~SingleEdgeUniformProposer() {}

    //     const double getLogProposalProbRatio(const GraphMove &move) const override;
    // };

    // class SingleEdgeDegreeProposer : public SingleEdgeProposer
    // {
    // private:
    //     VertexDegreeSampler m_vertexDegreeSampler;

    // public:
    //     SingleEdgeDegreeProposer(bool allowSelfLoops = true, bool allowMultiEdges = true, double bias = 1.0, double shift = 1) : SingleEdgeProposer(allowSelfLoops, allowMultiEdges, bias),
    //                                                                                                                              m_vertexDegreeSampler(shift) { m_vertexSamplerPtr = &m_vertexDegreeSampler; }

    //     virtual ~SingleEdgeDegreeProposer() {}

    //     const double getLogProposalProbRatio(const GraphMove &move) const override;
    // };

} // namespace GraphInf

#endif
