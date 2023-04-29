#include "gtest/gtest.h"

#include "SamplableSet.hpp"
#include "hash_specialization.hpp"
#include "GraphInf/graph/prior/edge_count.h"
#include "GraphInf/graph/erdosrenyi.h"
#include "GraphInf/graph/proposer/edge/hinge_flip.h"
#include "GraphInf/graph/proposer/movetypes.h"
#include "GraphInf/rng.h"
#include "../fixtures.hpp"

namespace GraphInf
{

    class DummyEdgeProposer : public HingeFlipUniformProposer
    {
    public:
        using HingeFlipUniformProposer::HingeFlipUniformProposer;
        const EdgeSampler &getEdgeSampler() { return m_edgeSampler; }
    };

    class TestHingeFlipUniformProposer : public ::testing::Test
    {
    public:
        ErdosRenyiModel randomGraph = ErdosRenyiModel(3, 3);
        DummyEdgeProposer proposer;
        MultiGraph graph;
        MultiGraph toyGraph = getToyMultiGraph();
        bool expectInconsistency = false;

        void SetUp()
        {
            randomGraph.sample();
            graph = randomGraph.getState();
            randomGraph.setState(graph);
            proposer.setUpWithGraph(graph);
            proposer.checkSafety();
        }
        void TearDown()
        {
            if (not expectInconsistency)
                proposer.checkConsistency();
        }

        const MultiGraph getToyMultiGraph()
        {
            /*
            0 --- 1<>
            ||
            ||
            ||
            2     3
            */
            MultiGraph graph(4);

            graph.addEdge(0, 1);
            graph.addEdge(1, 1);
            graph.addEdge(0, 2);
            graph.addEdge(0, 2);
            return graph;
        }
    };

    TEST_F(TestHingeFlipUniformProposer, setup_anyGraph_edgeSamplableSetContainsAllEdges)
    {
        EXPECT_EQ(graph.getTotalEdgeNumber(), proposer.getEdgeSampler().getTotalWeight());
        EXPECT_EQ(graph.getEdgeNumber(), proposer.getEdgeSampler().getSize());
    }

    TEST_F(TestHingeFlipUniformProposer, setup_anyGraph_samplableSetHasOnlyOrderedEdges)
    {
        for (const auto &edge : graph.edges())
        {
            auto mult = graph.getEdgeMultiplicity(edge.first, edge.second);
            if (edge.first <= edge.second)
                EXPECT_EQ(round(proposer.getEdgeSampler().getEdgeWeight(edge)), mult);
            else
                EXPECT_EQ(proposer.getEdgeSampler().getEdgeWeight(edge), 0);
        }
    }

    TEST_F(TestHingeFlipUniformProposer, applyGraphMove_addExistentEdge_edgeWeightIncreased)
    {
        BaseGraph::Edge edge = {0, 2};
        GraphMove move = {{}, {edge}};
        proposer.applyGraphMove(move);
        auto mult = graph.getEdgeMultiplicity(edge.first, edge.second);
        EXPECT_EQ(proposer.getEdgeSampler().getEdgeWeight(edge), mult + 1);
    }

    TEST_F(TestHingeFlipUniformProposer, applyGraphMove_addMultiEdge_edgeWeightIncreased)
    {
        BaseGraph::Edge edge = {0, 1};
        GraphMove move = {{}, {edge, edge}};
        proposer.applyGraphMove(move);
        auto mult = graph.getEdgeMultiplicity(edge.first, edge.second);
        EXPECT_EQ(proposer.getEdgeSampler().getEdgeWeight(edge), mult + 2);
        expectInconsistency = true;
    }

    TEST_F(TestHingeFlipUniformProposer, applyGraphMove_addInexistentEdge_edgeWeightIncreased)
    {
        BaseGraph::Edge edge = {0, 1};
        BaseGraph::Edge reversedEdge = {1, 0};
        GraphMove move = {{}, {edge}};
        proposer.applyGraphMove(move);
        auto mult = graph.getEdgeMultiplicity(edge.first, edge.second);
        EXPECT_EQ(proposer.getEdgeSampler().getEdgeWeight(edge), mult + 1);
        EXPECT_EQ(proposer.getEdgeSampler().getEdgeWeight(reversedEdge), 0);
    }

    TEST_F(TestHingeFlipUniformProposer, applyGraphMove_removeEdge_edgeWeightDecreased)
    {
        BaseGraph::Edge edge;
        size_t weight;
        while (true)
        {
            edge = proposer.getEdgeSampler().sample();
            weight = graph.getEdgeMultiplicity(edge.first, edge.second);
            if (edge.first != edge.second)
                break;
        }

        size_t edgeMult = graph.getEdgeMultiplicity(edge.first, edge.second);
        GraphMove move = {{edge}, {}};
        proposer.applyGraphMove(move);
        if (edgeMult > 1)
            EXPECT_EQ(proposer.getEdgeSampler().getEdgeWeight(edge), graph.getEdgeMultiplicity(edge.first, edge.second) - 1);
        expectInconsistency = true;
    }
    TEST_F(TestHingeFlipUniformProposer, applyGraphMove_removeSelfLoop_edgeWeightDecreased)
    {
        BaseGraph::Edge edge;
        size_t weight;
        while (true)
        {
            SetUp();
            edge = proposer.getEdgeSampler().sample();
            weight = graph.getEdgeMultiplicity(edge.first, edge.second);
            if (edge.first == edge.second)
                break;
        }

        size_t edgeMult = graph.getEdgeMultiplicity(edge.first, edge.second);
        GraphMove move = {{edge}, {}};
        proposer.applyGraphMove(move);
        if (edgeMult > 1)
            EXPECT_EQ(proposer.getEdgeSampler().getEdgeWeight(edge), edgeMult - 1);
        expectInconsistency = true;
    }

    TEST_F(TestHingeFlipUniformProposer, getLogProposalProbRatio_forNormalMove_returnCorrectValue)
    {
        proposer.setUpWithGraph(toyGraph);

        GraphMove normalMove1 = {{{0, 1}}, {{0, 3}}};
        EXPECT_FLOAT_EQ(proposer.getLogProposalProbRatio(normalMove1), 0);

        GraphMove normalMove2 = {{{0, 2}}, {{0, 1}}};
        EXPECT_FLOAT_EQ(proposer.getLogProposalProbRatio(normalMove2), 0);

        GraphMove normalMove3 = {{{0, 2}}, {{0, 3}}};
        EXPECT_FLOAT_EQ(proposer.getLogProposalProbRatio(normalMove3), -log(2));
    }

    TEST_F(TestHingeFlipUniformProposer, getLogProposalProbRatio_forLoopyMove_returnCorrectValue)
    {
        proposer.setUpWithGraph(toyGraph);

        GraphMove loopyMove1 = {{{1, 1}}, {{1, 3}}};
        EXPECT_FLOAT_EQ(proposer.getLogProposalProbRatio(loopyMove1), -log(2));

        GraphMove loopyMove2 = {{{1, 1}}, {{1, 0}}};
        EXPECT_FLOAT_EQ(proposer.getLogProposalProbRatio(loopyMove2), 0);
    }

    TEST_F(TestHingeFlipUniformProposer, getLogProposalProbRatio_forSelfieMove_returnCorrectValue)
    {
        proposer.setUpWithGraph(toyGraph);

        GraphMove selfieMove1 = {{{0, 1}}, {{0, 0}}};
        EXPECT_FLOAT_EQ(proposer.getLogProposalProbRatio(selfieMove1), log(2));

        GraphMove selfieMove2 = {{{1, 0}}, {{1, 0}}};
        EXPECT_FLOAT_EQ(proposer.getLogProposalProbRatio(selfieMove2), 0);

        GraphMove selfieMove3 = {{{1, 0}}, {{1, 1}}};
        EXPECT_FLOAT_EQ(proposer.getLogProposalProbRatio(selfieMove3), 2 * log(2));

        GraphMove selfieMove4 = {{{0, 1}}, {{0, 1}}};
        EXPECT_FLOAT_EQ(proposer.getLogProposalProbRatio(selfieMove4), 0);
    }

    TEST_F(TestHingeFlipUniformProposer, getLogProposalProbRatio_forLoopySelfieMove_returnCorrectValue)
    {
        proposer.setUpWithGraph(toyGraph);

        GraphMove selfieMove1 = {{{1, 1}}, {{1, 1}}};
        EXPECT_FLOAT_EQ(proposer.getLogProposalProbRatio(selfieMove1), 0);
    }

}
