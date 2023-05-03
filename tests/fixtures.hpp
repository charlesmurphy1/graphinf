#ifndef FASTMIDYNET_GRAPH_FIXTURES_HPP
#define FASTMIDYNET_GRAPH_FIXTURES_HPP

#include "GraphInf/types.h"

#include "GraphInf/graph/likelihood/likelihood.hpp"
#include "GraphInf/graph/prior/edge_count.h"
#include "GraphInf/graph/prior/block_count.h"
#include "GraphInf/graph/prior/block.h"
#include "GraphInf/graph/prior/label_graph.h"
#include "GraphInf/graph/random_graph.hpp"
#include "GraphInf/graph/prior/nested_block.h"
#include "GraphInf/graph/sbm.h"
#include "GraphInf/graph/hsbm.h"
#include "GraphInf/graph/erdosrenyi.h"

#include "GraphInf/data/dynamics/sis.h"

namespace GraphInf
{

    static MultiGraph getUndirectedHouseMultiGraph()
    {
        //     /*
        //      * (0)      (1)
        //      * ||| \   / | \
    //      * |||  \ /  |  \
    //      * |||   X   |  (4)
        //      * |||  / \  |  /
        //      * ||| /   \ | /
        //      * (2)------(3)-----(5)
        //      *
        //      *      (6)
        //      */
        // STATE = {0,0,0,1,1,1,2}
        // NEIGHBORS_STATE = {{3, 1, 0}, {1, 2, 0}, {4, 1, 0}, {3, 1, 1}, {1, 1, 0}, {0, 1, 0}, {0, 0, 0}}
        GraphInf::MultiGraph graph(7);
        graph.addMultiedge(0, 2, 3);
        graph.addEdge(0, 3);
        graph.addEdge(1, 2);
        graph.addEdge(1, 3);
        graph.addEdge(1, 4);
        graph.addEdge(2, 3);
        graph.addEdge(3, 4);
        graph.addEdge(3, 5);

        return graph;
    }

    static void doMetropolisHastingsSweepForGraph(RandomGraph &randomGraph, size_t numIteration = 10, bool verbose = false)
    {
        std::uniform_real_distribution<double> dist(0, 1);
        for (size_t it = 0; it < numIteration; ++it)
        {
            auto move = randomGraph.proposeGraphMove();
            if (not randomGraph.isValidGraphMove(move))
                continue;
            double logLikelihood = randomGraph.getLogLikelihoodRatioFromGraphMove(move);
            double logPrior = randomGraph.getLogPriorRatioFromGraphMove(move);
            double logProp = randomGraph.getLogProposalRatioFromGraphMove(move);
            bool accepted = false;
            if (dist(rng) < exp(logLikelihood + logPrior + logProp))
            {
                randomGraph.applyGraphMove(move);
                accepted = true;
            }
            if (verbose)
            {
                std::cout << "Iteration " << it << ": move=" << move;
                std::cout << ", logLikelihood=" << logLikelihood;
                std::cout << ", logPrior=" << logPrior;
                std::cout << ", logProp=" << logProp;
                std::cout << ", isAccepted=" << (int)accepted << std::endl;
            }
            randomGraph.checkConsistency();
        }
    }

    static void doMetropolisHastingsSweepForLabels(VertexLabeledRandomGraph<BlockIndex> &randomGraph, size_t numIteration = 10, bool verbose = false)
    {
        std::uniform_real_distribution<double> dist(0, 1);
        for (size_t it = 0; it < numIteration; ++it)
        {
            auto move = randomGraph.proposeLabelMove();
            if (not randomGraph.isValidLabelMove(move))
                continue;
            double logLikelihood = randomGraph.getLogLikelihoodRatioFromLabelMove(move);
            double logPrior = randomGraph.getLogPriorRatioFromLabelMove(move);
            double logProp = randomGraph.getLogProposalRatioFromLabelMove(move);
            bool accepted = false;
            if (dist(rng) < exp(logLikelihood + logPrior + logProp))
            {
                randomGraph.applyLabelMove(move);
                accepted = true;
            }
            if (verbose)
            {
                std::cout << "Iteration " << it << ": move=" << move;
                std::cout << ", logLikelihood=" << logLikelihood;
                std::cout << ", logPrior=" << logPrior;
                std::cout << ", logProp=" << logProp;
                std::cout << ", isAccepted=" << (int)accepted << std::endl;
            }
            randomGraph.checkConsistency();
        }
    }

    class DummyGraphLikelihood : public GraphLikelihoodModel
    {
    public:
        const MultiGraph sample() const { return generateErdosRenyi(*m_sizePtr, *m_edgeCountPtr); }
        const double getLogLikelihood() const { return 0; }
        const double getLogLikelihoodRatioFromGraphMove(const GraphMove &) const { return 0; }
        size_t *m_sizePtr = nullptr;
        size_t *m_edgeCountPtr = nullptr;
    };

    class DummyRandomGraph : public RandomGraph
    {
        size_t m_edgeCount;
        HingeFlipUniformProposer m_edgeProposer;
        DummyGraphLikelihood likelihood;
        void setUpLikelihood() override
        {
            likelihood.m_sizePtr = &m_size;
            likelihood.m_edgeCountPtr = &m_edgeCount;
        }

    public:
        using RandomGraph::RandomGraph;
        DummyRandomGraph(size_t size) : RandomGraph(size, likelihood)
        {
            setEdgeProposer(m_edgeProposer);
        }

        void setState(const MultiGraph &state) override
        {
            RandomGraph::setState(state);
            m_edgeCount = state.getTotalEdgeNumber();
        }

        const size_t getEdgeCount() const override { return m_edgeCount; }
    };

    class DummyDynamics : public Dynamics
    {
    public:
        DummyDynamics(RandomGraph &graphPrior, size_t numStates = 2, size_t length = 10) : Dynamics(graphPrior, numStates, length) {}

        const double getTransitionProb(
            const VertexState &prevVertexState,
            const VertexState &nextVertexState,
            const VertexNeighborhoodState &vertexNeighborhoodState) const { return 1. / getNumStates(); }

        void updateNeighborsStateFromEdgeMove(
            BaseGraph::Edge edge,
            int direction,
            std::map<BaseGraph::VertexIndex, VertexNeighborhoodStateSequence> &prev,
            std::map<BaseGraph::VertexIndex, VertexNeighborhoodStateSequence> &next) const { Dynamics::updateNeighborsStateFromEdgeMove(edge, direction, prev, next); }
    };

    class DummySISDynamics : public SISDynamics
    {
    public:
        DummySISDynamics(RandomGraph &graphPrior, size_t length = 10, double infection = 0.1) : SISDynamics(graphPrior, length, infection) {}
    };

    class DummyLabeledSISDynamics : public SISDynamics
    {
    public:
        DummyLabeledSISDynamics(VertexLabeledRandomGraph<BlockIndex> &graphPrior, size_t length = 10, double infection = 0.1) : SISDynamics(graphPrior, length, infection) {}
    };

    class DummyNestedSISDynamics : public SISDynamics
    {
    public:
        DummyNestedSISDynamics(NestedVertexLabeledRandomGraph<BlockIndex> &graphPrior, size_t length = 10, double infection = 0.1) : SISDynamics(graphPrior, length, infection) {}
    };

}
#endif
