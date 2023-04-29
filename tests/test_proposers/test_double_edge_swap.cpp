#include "gtest/gtest.h"
#include "GraphInf/graph/proposer/edge/double_edge_swap.h"
#include "GraphInf/graph/proposer/movetypes.h"
#include "../fixtures.hpp"

namespace GraphInf
{

    class DummyDoubleEdgeSwapProposer : public DoubleEdgeSwapProposer
    {
    public:
        using DoubleEdgeSwapProposer::DoubleEdgeSwapProposer;
        const EdgeSampler &getEdgeSampler() { return m_edgeSampler; }
    };

    class TestDoubleEdgeSwapProposer : public ::testing::Test
    {
    public:
        MultiGraph graph = getUndirectedHouseMultiGraph();
        MultiGraph toyGraph = getToyMultiGraph();
        DummyDoubleEdgeSwapProposer proposer;
        void SetUp()
        {
            proposer.setUpWithGraph(graph);
            proposer.checkSafety();
        }
        void TearDown()
        {
            proposer.checkConsistency();
        }
        const MultiGraph getToyMultiGraph()
        {
            /*
            0 === 1<>
            |     |
            |     |
            2 --- 3<>
            */
            MultiGraph graph(4);

            graph.addEdge(0, 1);
            graph.addEdge(0, 1);
            graph.addEdge(1, 1);
            graph.addEdge(0, 2);
            graph.addEdge(1, 3);
            graph.addEdge(2, 3);
            graph.addEdge(3, 3);
            return graph;
        }
    };

    TEST_F(TestDoubleEdgeSwapProposer, setup_anyGraph_samplerContainsAllEdges)
    {
        EXPECT_EQ(graph.getTotalEdgeNumber(), proposer.getEdgeSampler().getTotalWeight());
        EXPECT_EQ(graph.getEdgeNumber(), proposer.getEdgeSampler().getSize());
    }

    TEST_F(TestDoubleEdgeSwapProposer, setup_anyGraph_samplerHasOnlyOrderedEdges)
    {
        for (const auto &edge : graph.edges())
        {
            auto mult = graph.getEdgeMultiplicity(edge.first, edge.second);
            if (edge.first <= edge.second)
                EXPECT_EQ(round(proposer.getEdgeSampler().getEdgeWeight(edge)), mult);
            else
                EXPECT_EQ(round(proposer.getEdgeSampler().getEdgeWeight(edge)), 0);
        }
    }

    TEST_F(TestDoubleEdgeSwapProposer, applyGraphMove_addExistentEdge_edgeWeightIncreased)
    {
        BaseGraph::Edge edge = {0, 2};
        GraphMove move = {{}, {edge}};
        proposer.applyGraphMove(move);
        auto mult = graph.getEdgeMultiplicity(edge.first, edge.second);
        EXPECT_EQ(proposer.getEdgeSampler().getEdgeWeight(edge), mult + 1);
    }

    TEST_F(TestDoubleEdgeSwapProposer, applyGraphMove_addInexistentMultiEdge_edgeWeightIncreased)
    {
        BaseGraph::Edge edge = {0, 1};
        GraphMove move = {{}, {edge, edge}};
        proposer.applyGraphMove(move);
        auto mult = graph.getEdgeMultiplicity(edge.first, edge.second);
        EXPECT_EQ(proposer.getEdgeSampler().getEdgeWeight(edge), mult + 2);
    }

    TEST_F(TestDoubleEdgeSwapProposer, applyGraphMove_addInexistentEdge_edgeWeightIncreased)
    {
        BaseGraph::Edge edge = {0, 1};
        BaseGraph::Edge reversedEdge = {1, 0};
        GraphMove move = {{}, {edge}};
        proposer.applyGraphMove(move);
        auto mult = graph.getEdgeMultiplicity(edge.first, edge.second);
        EXPECT_EQ(proposer.getEdgeSampler().getEdgeWeight(edge), mult + 1);
        EXPECT_EQ(proposer.getEdgeSampler().getEdgeWeight(reversedEdge), 0);
    }

    TEST_F(TestDoubleEdgeSwapProposer, applyGraphMove_removeEdge_edgeWeightDecreased)
    {
        BaseGraph::Edge edge = {0, 2};
        GraphMove move = {{edge}, {}};
        proposer.applyGraphMove(move);
        auto mult = graph.getEdgeMultiplicity(edge.first, edge.second);
        EXPECT_EQ(proposer.getEdgeSampler().getEdgeWeight(edge), mult - 1);
    }

    TEST_F(TestDoubleEdgeSwapProposer, applyGraphMove_removeMultiEdge_edgeWeightDecreased)
    {
        BaseGraph::Edge edge = {0, 2};
        GraphMove move = {{edge, edge}, {}};
        proposer.applyGraphMove(move);
        auto mult = graph.getEdgeMultiplicity(edge.first, edge.second);
        EXPECT_EQ(proposer.getEdgeSampler().getEdgeWeight(edge), mult - 2);
    }

    TEST_F(TestDoubleEdgeSwapProposer, applyGraphMove_removeAllEdges_edgeRemovedFromSamplableSet)
    {
        BaseGraph::Edge edge = {0, 2};
        GraphMove move = {{edge, edge, edge}, {}};
        proposer.applyGraphMove(move);
        EXPECT_EQ(proposer.getEdgeSampler().getEdgeWeight(edge), 0);
    }

    TEST_F(TestDoubleEdgeSwapProposer, getLogProposalProbRatio_forNormalGraphMove_returnCorrectValue)
    {
        proposer.setUpWithGraph(toyGraph);
        GraphMove move;

        move = {{{0, 2}, {1, 3}}, {{0, 1}, {2, 3}}};
        double w02 = toyGraph.getEdgeMultiplicity(0, 2), w13 = toyGraph.getEdgeMultiplicity(1, 3);
        double w01 = toyGraph.getEdgeMultiplicity(0, 1), w23 = toyGraph.getEdgeMultiplicity(2, 3);
        EXPECT_FLOAT_EQ(proposer.getLogProposalProbRatio(move), log(w01 + 1) + log(w23 + 1) - log(w02) - log(w13));

        move = {{{0, 2}, {1, 3}}, {{0, 3}, {1, 2}}};
        double w03 = toyGraph.getEdgeMultiplicity(0, 3), w12 = toyGraph.getEdgeMultiplicity(1, 2);
        EXPECT_FLOAT_EQ(proposer.getLogProposalProbRatio(move), log(w03 + 1) + log(w12 + 1) - log(w02) - log(w13));
    }

    TEST_F(TestDoubleEdgeSwapProposer, getLogProposalProbRatio_forDoubleLoopyGraphMove_returnCorrectValue)
    {
        proposer.setUpWithGraph(toyGraph);
        GraphMove move;

        move = {{{1, 1}, {3, 3}}, {{1, 3}, {1, 3}}};

        double w11 = toyGraph.getEdgeMultiplicity(1, 1), w33 = toyGraph.getEdgeMultiplicity(3, 3);
        double w13 = toyGraph.getEdgeMultiplicity(1, 3);
        EXPECT_FLOAT_EQ(proposer.getLogProposalProbRatio(move), log(w13 + 2) + log(w13 + 1) - log(w11) - log(w33) - log(4));
    }

    TEST_F(TestDoubleEdgeSwapProposer, getLogProposalProbRatio_forSingleLoopyGraphMove_returnCorrectValue)
    {
        proposer.setUpWithGraph(toyGraph);
        GraphMove move;

        move = {{{1, 1}, {0, 2}}, {{0, 1}, {1, 2}}};
        double w11 = toyGraph.getEdgeMultiplicity(1, 1), w02 = toyGraph.getEdgeMultiplicity(0, 2);
        double w01 = toyGraph.getEdgeMultiplicity(0, 1), w12 = toyGraph.getEdgeMultiplicity(1, 2);
        EXPECT_FLOAT_EQ(proposer.getLogProposalProbRatio(move), log(w01 + 1) + log(w12 + 1) - log(w11) - log(w02) - log(2));
    }

    TEST_F(TestDoubleEdgeSwapProposer, getLogProposalProbRatio_forDoubleEdgeGraphMove_returnCorrectValue)
    {
        proposer.setUpWithGraph(toyGraph);
        GraphMove move;

        move = {{{0, 1}, {0, 1}}, {{0, 1}, {0, 1}}};
        double w01 = toyGraph.getEdgeMultiplicity(0, 1);
        EXPECT_FLOAT_EQ(proposer.getLogProposalProbRatio(move), 0);

        move = {{{0, 1}, {0, 1}}, {{0, 0}, {1, 1}}};
        double w00 = toyGraph.getEdgeMultiplicity(0, 0), w11 = toyGraph.getEdgeMultiplicity(1, 1);
        EXPECT_FLOAT_EQ(proposer.getLogProposalProbRatio(move), log(4) + log(w11 + 1) + log(w00 + 1) - log(w01) - log(w01 - 1));
    }

    TEST_F(TestDoubleEdgeSwapProposer, getLogProposalProbRatio_forHingeGraphMove_returnCorrectValue)
    {
        proposer.setUpWithGraph(toyGraph);
        GraphMove move;

        move = {{{1, 3}, {2, 3}}, {{1, 3}, {2, 3}}};
        double w13 = toyGraph.getEdgeMultiplicity(1, 3), w23 = toyGraph.getEdgeMultiplicity(2, 3);
        EXPECT_FLOAT_EQ(proposer.getLogProposalProbRatio(move), 0);

        move = {{{1, 3}, {2, 3}}, {{1, 2}, {3, 3}}};
        double w12 = toyGraph.getEdgeMultiplicity(1, 2), w33 = toyGraph.getEdgeMultiplicity(3, 3);
        EXPECT_FLOAT_EQ(proposer.getLogProposalProbRatio(move), log(2) + log(w12 + 1) + log(w33 + 1) - log(w13) - log(w23));
    }

}
