#ifndef GRAPH_INF_EDGE_PROPOSER_H
#define GRAPH_INF_EDGE_PROPOSER_H

#include <stdexcept>
#include <unordered_set>

#include "GraphInf/graph/proposer/proposer.hpp"
#include "GraphInf/graph/proposer/movetypes.h"
#include "GraphInf/types.h"
#include "GraphInf/exceptions.h"
#include "GraphInf/utility/maps.hpp"

namespace GraphInf
{

    class RandomGraph;

    class EdgeProposer : public Proposer<GraphMove>
    {
    protected:
        const bool m_allowSelfLoops;
        const bool m_allowMultiEdges;
        const size_t m_maxIteration = 100;
        const MultiGraph *m_graphPtr = nullptr;
        mutable std::uniform_real_distribution<double> m_uniform01;
        bool isSelfLoop(BaseGraph::Edge edge) const { return edge.first == edge.second; }
        bool isExistingEdge(BaseGraph::Edge edge) const { return m_graphPtr->getEdgeMultiplicity(edge.first, edge.second) >= 1; }

    public:
        using Proposer<GraphMove>::Proposer;
        EdgeProposer(bool allowSelfLoops = true, bool allowMultiEdges = true) : m_allowSelfLoops(allowSelfLoops), m_allowMultiEdges(allowMultiEdges) {}
        virtual ~EdgeProposer() {}
        const GraphMove proposeMove() const override;
        virtual const GraphMove proposeRawMove() const = 0;
        virtual const double getLogProposalProbRatio(const GraphMove &move) const = 0;
        const GraphMove getReverseMove(const GraphMove &move) const
        {
            return {move.addedEdges, move.removedEdges};
        }
        // virtual void setUp( const RandomGraph& randomGraph ) { clear(); setUpFromGraph(randomGraph.getGraph()); }
        virtual void setUpWithGraph(const MultiGraph &graph)
        {
            clear();
            m_graphPtr = &graph;
        }
        virtual void setUpWithPrior(const RandomGraph &prior);
        virtual void applyGraphMove(const GraphMove &move){};
        // virtual void applyBlockMove(const BlockMove& move) {};
        const bool &allowSelfLoops() const { return m_allowSelfLoops; }
        const bool &allowMultiEdges() const { return m_allowMultiEdges; }

        const bool isTrivialMove(const GraphMove &move) const
        {
            IntMap<BaseGraph::Edge> counter;
            for (const auto edge : move.addedEdges)
                counter.increment(edge);
            for (const auto edge : move.removedEdges)
                counter.decrement(edge);
            for (const auto key : counter)
                if (key.second != 0)
                    return false;
            return true;
        }

        virtual void clear() override { m_graphPtr = nullptr; }
        bool isSafe() const override
        {
            return (m_graphPtr != nullptr);
        }
        void checkSelfSafety() const override
        {
            if (m_graphPtr == nullptr)
                throw SafetyError("EdgeProposer: unsafe usage since `m_graphPtr` is NULL");
        }
    };

}

#endif
