#include "gtest/gtest.h"

#include "GraphInf/graph/prior/edge_count.h"
#include "GraphInf/graph/erdosrenyi.h"
#include "GraphInf/graph/proposer/edge/single_edge.h"
#include "GraphInf/graph/proposer/movetypes.h"
#include "GraphInf/utility/functions.h"
#include "../fixtures.hpp"

namespace GraphInf
{

    class TestSingleEdgeUniformProposer : public ::testing::Test
    {
    public:
        ErdosRenyiModel randomGraph = ErdosRenyiModel(10, 10);
        SingleEdgeUniformProposer proposer;
        MultiGraph graph;
        BaseGraph::Edge inexistentEdge = {0, 1};
        BaseGraph::Edge singleEdge = {0, 2};
        BaseGraph::Edge doubleEdge = {0, 3};
        void SetUp()
        {
            randomGraph.sample();
            graph = randomGraph.getState();
            graph.setEdgeMultiplicity(inexistentEdge.first, inexistentEdge.second, 0);
            graph.setEdgeMultiplicity(singleEdge.first, singleEdge.second, 1);
            graph.setEdgeMultiplicity(doubleEdge.first, doubleEdge.second, 2);
            randomGraph.setState(graph);
            proposer.setUpWithGraph(graph);
            proposer.checkSafety();
        }
        void TearDown()
        {
            proposer.checkConsistency();
        }
    };

    TEST_F(TestSingleEdgeUniformProposer, getLogProposalProbRatio_addEdge_return0)
    {
        GraphInf::GraphMove move = {{}, {inexistentEdge}};
        proposer.applyGraphMove(move);
        EXPECT_EQ(proposer.getLogProposalProbRatio(move), -log(0.5));

        move = {{}, {singleEdge}};
        proposer.applyGraphMove(move);
        EXPECT_EQ(proposer.getLogProposalProbRatio(move), 0);
    }

    TEST_F(TestSingleEdgeUniformProposer, getLogProposalProbRatio_removeEdgeWithMultiplicity2_return0)
    {
        GraphInf::GraphMove move = {{doubleEdge}, {}};
        proposer.applyGraphMove(move);
        EXPECT_EQ(proposer.getLogProposalProbRatio(move), 0);
    }

    TEST_F(TestSingleEdgeUniformProposer, getLogProposalProbRatio_removeEdgeWithMultiplicity1_returnCorrectRatio)
    {
        GraphInf::GraphMove move = {{singleEdge}, {}};
        proposer.applyGraphMove(move);
        EXPECT_EQ(proposer.getLogProposalProbRatio(move), log(.5));
    }

    class TestSingleEdgeDegreeProposer : public ::testing::Test
    {
    public:
        EdgeCountDeltaPrior edgeCountPrior = {10};
        ErdosRenyiModel randomGraph = ErdosRenyiModel(10, 10);
        SingleEdgeDegreeProposer proposer;
        MultiGraph graph;
        BaseGraph::Edge inexistentEdge = {0, 1};
        BaseGraph::Edge singleEdge = {0, 2};
        BaseGraph::Edge doubleEdge = {0, 3};
        void SetUp()
        {
            randomGraph.sample();
            graph = randomGraph.getState();

            graph.setEdgeMultiplicity(inexistentEdge.first, inexistentEdge.second, 0);
            graph.setEdgeMultiplicity(singleEdge.first, singleEdge.second, 1);
            graph.setEdgeMultiplicity(doubleEdge.first, doubleEdge.second, 2);
            randomGraph.setState(graph);
            proposer.setUpWithGraph(graph);
            proposer.checkSafety();
        }
        void TearDown()
        {
            proposer.checkConsistency();
        }
    };

    TEST_F(TestSingleEdgeDegreeProposer, getLogProposalProbRatio_addEdge_return0)
    {
        GraphInf::GraphMove move = {{}, {inexistentEdge}};
        proposer.applyGraphMove(move);
    }

    TEST_F(TestSingleEdgeDegreeProposer, getLogProposalProbRatio_removeEdgeWithMultiplicity2_return0)
    {
        GraphInf::GraphMove move = {{doubleEdge}, {}};
        // proposer.applyGraphMove(move);
    }

    TEST_F(TestSingleEdgeDegreeProposer, getLogProposalProbRatio_removeEdgeWithMultiplicity1_returnCorrectRatio)
    {
        GraphInf::GraphMove move = {{singleEdge}, {}};
        // proposer.applyGraphMove(move);
    }

}
