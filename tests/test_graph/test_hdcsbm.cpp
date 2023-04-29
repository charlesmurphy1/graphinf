#include "gtest/gtest.h"
#include <list>
#include <algorithm>
#include <string>

#include "../fixtures.hpp"
#include "GraphInf/graph/prior/block_count.h"
#include "GraphInf/graph/prior/nested_block.h"
#include "GraphInf/graph/prior/nested_label_graph.h"
#include "GraphInf/graph/prior/labeled_degree.h"
#include "GraphInf/graph/hdcsbm.h"
#include "GraphInf/types.h"
#include "GraphInf/utility/functions.h"
#include "BaseGraph/types.h"

using namespace std;
using namespace GraphInf;

class HDCSBMParametrizedTest : public ::testing::TestWithParam<bool>
{
public:
    const size_t NUM_VERTICES = 50, NUM_EDGES = 100;
    const bool canonical = false;
    NestedDegreeCorrectedStochasticBlockModelFamily randomGraph = NestedDegreeCorrectedStochasticBlockModelFamily(
        NUM_VERTICES, NUM_EDGES, GetParam(), canonical);
    BaseGraph::VertexIndex vertexIdx = 4;

    BaseGraph::Edge findEdge()
    {
        const auto &graph = randomGraph.getState();
        BaseGraph::Edge edge;
        BaseGraph::VertexIndex neighborIdx;
        for (auto idx : graph)
        {
            if (graph.getDegree(idx) > 0)
            {
                auto neighbor = *graph.getOutNeighbours(idx).begin();
                edge = {idx, neighbor};
                return edge;
            }
        }
        throw std::invalid_argument("State of randomGraph has no edge.");
    }

    GraphInf::BlockIndex findBlockMove(BaseGraph::VertexIndex idx)
    {
        GraphInf::BlockIndex blockIdx = randomGraph.getLabel(idx);
        if (blockIdx == randomGraph.getVertexCounts().size() - 1)
            return blockIdx - 1;
        else
            return blockIdx + 1;
    }

    void SetUp()
    {
        while (randomGraph.getLabelCount() > 30)
            randomGraph.sample();
    }
};

TEST_P(HDCSBMParametrizedTest, sampleState_graphChanges)
{
    for (size_t i = 0; i < 10; i++)
    {
        auto prevGraph = randomGraph.getState();
        randomGraph.sample();
        auto nextGraph = randomGraph.getState();
        EXPECT_FALSE(prevGraph == nextGraph);
    }
}

TEST_P(HDCSBMParametrizedTest, getLogLikelihood_returnNonZeroValue)
{
    EXPECT_TRUE(randomGraph.getLogLikelihood() < 0);
}

TEST_P(HDCSBMParametrizedTest, applyGraphMove_forAddedEdge)
{
    BaseGraph::Edge addedEdge = {0, 2};
    size_t addedEdgeMultBefore;
    if (randomGraph.getState().hasEdge(addedEdge.first, addedEdge.second))
        addedEdgeMultBefore = randomGraph.getState().getEdgeMultiplicity(addedEdge.first, addedEdge.second);
    else
        addedEdgeMultBefore = 0;

    GraphInf::GraphMove move = {{}, {addedEdge}};
    randomGraph.applyGraphMove(move);
    size_t addedEdgeMultAfter = randomGraph.getState().getEdgeMultiplicity(addedEdge.first, addedEdge.second);
    EXPECT_EQ(addedEdgeMultAfter - 1, addedEdgeMultBefore);
}

TEST_P(HDCSBMParametrizedTest, applyGraphMove_forAddedSelfLoop)
{
    BaseGraph::Edge addedEdge = {0, 0};
    size_t addedEdgeMultBefore = randomGraph.getState().getEdgeMultiplicity(addedEdge.first, addedEdge.second);
    GraphInf::GraphMove move = {{}, {addedEdge}};
    randomGraph.applyGraphMove(move);
    size_t addedEdgeMultAfter = randomGraph.getState().getEdgeMultiplicity(addedEdge.first, addedEdge.second);
    EXPECT_EQ(addedEdgeMultAfter - 1, addedEdgeMultBefore);
}

TEST_P(HDCSBMParametrizedTest, applyGraphMove_forRemovedEdge)
{
    BaseGraph::Edge removedEdge = findEdge();
    size_t removedEdgeMultBefore = randomGraph.getState().getEdgeMultiplicity(removedEdge.first, removedEdge.second);
    GraphInf::GraphMove move = {{removedEdge}, {}};
    randomGraph.applyGraphMove(move);
    size_t removedEdgeMultAfter = randomGraph.getState().getEdgeMultiplicity(removedEdge.first, removedEdge.second);
    EXPECT_EQ(removedEdgeMultAfter + 1, removedEdgeMultBefore);
}

TEST_P(HDCSBMParametrizedTest, applyGraphMove_forRemovedEdgeAndAddedEdge)
{
    BaseGraph::Edge removedEdge = findEdge();
    BaseGraph::Edge addedEdge = {20, 21};
    size_t removedEdgeMultBefore = randomGraph.getState().getEdgeMultiplicity(removedEdge.first, removedEdge.second);
    size_t addedEdgeMultBefore = randomGraph.getState().getEdgeMultiplicity(addedEdge.first, addedEdge.second);
    GraphInf::GraphMove move = {{removedEdge}, {addedEdge}};
    randomGraph.applyGraphMove(move);
    size_t removedEdgeMultAfter = randomGraph.getState().getEdgeMultiplicity(removedEdge.first, removedEdge.second);
    size_t addedEdgeMultAfter = randomGraph.getState().getEdgeMultiplicity(addedEdge.first, addedEdge.second);
    EXPECT_EQ(removedEdgeMultAfter + 1, removedEdgeMultBefore);
    EXPECT_EQ(addedEdgeMultAfter - 1, addedEdgeMultBefore);
}

TEST_P(HDCSBMParametrizedTest, applyGraphMove_forNoEdgesAddedOrRemoved)
{
    GraphInf::GraphMove move = {{}, {}};
    randomGraph.applyGraphMove(move);
}

// TEST_P(HDCSBMParametrizedTest, applyMove_forIdentityBlockMove_doNothing){
//     GraphInf::BlockIndex prevBlockIdx = randomGraph.getLabel(vertexIdx);
//     GraphInf::BlockIndex nextBlockIdx = prevBlockIdx;
//
//     GraphInf::BlockMove move = {vertexIdx, prevBlockIdx, nextBlockIdx};
//     randomGraph.applyLabelMove(move);
// }
//
// TEST_P(HDCSBMParametrizedTest, applyMove_forBlockMoveWithNoBlockCreation_changeBlockIdx){
//     GraphInf::BlockIndex prevBlockIdx = randomGraph.getLabel(vertexIdx);
//     GraphInf::BlockIndex nextBlockIdx = prevBlockIdx;
//     if (prevBlockIdx == randomGraph.getVertexCounts().size() - 1) nextBlockIdx --;
//     else nextBlockIdx ++;
//
//     GraphInf::BlockMove move = {vertexIdx, prevBlockIdx, nextBlockIdx};
//     EXPECT_EQ(randomGraph.getLabel(vertexIdx), prevBlockIdx);
//     randomGraph.applyLabelMove(move);
//     EXPECT_EQ(randomGraph.getLabel(vertexIdx), nextBlockIdx);
// }
//
// TEST_P(HDCSBMParametrizedTest, applyMove_forBlockMoveWithBlockCreation_changeBlockIdxAndBlockCount){
//     GraphInf::BlockIndex prevBlockIdx = randomGraph.getLabel(vertexIdx);
//     GraphInf::BlockIndex nextBlockIdx = randomGraph.getVertexCounts().size();
//     GraphInf::BlockMove move = {vertexIdx, prevBlockIdx, nextBlockIdx};
//     EXPECT_EQ(randomGraph.getLabel(vertexIdx), prevBlockIdx);
//     randomGraph.applyLabelMove(move);
//     EXPECT_EQ(randomGraph.getLabel(vertexIdx), nextBlockIdx);
// }
//
// // TEST_P(HDCSBMParametrizedTest, applyMove_forBlockMoveWithBlockDestruction_changeBlockIdxAndBlockCount){
// //     GraphInf::BlockIndex prevBlockIdx = randomGraph.getVertexCounts().size();
// //     GraphInf::BlockIndex nextBlockIdx = randomGraph.getLabel(vertexIdx);
// //     GraphInf::BlockMove move = {vertexIdx, nextBlockIdx, prevBlockIdx};
// //     randomGraph.applyLabelMove(move); // creating block before destroying it
// //     move = {vertexIdx, prevBlockIdx, nextBlockIdx};
// //     EXPECT_EQ(randomGraph.getLabel(vertexIdx), prevBlockIdx);
// //     randomGraph.applyLabelMove(move);
// //     EXPECT_EQ(randomGraph.getLabel(vertexIdx), nextBlockIdx);
// // }
//
TEST_P(HDCSBMParametrizedTest, getLogLikelihoodRatio_forAddedSelfLoop_returnCorrectLogLikelihoodRatio)
{
    GraphInf::GraphMove move = {{}, {{0, 0}}};
    double actualLogLikelihoodRatio = randomGraph.getLogLikelihoodRatioFromGraphMove(move);

    double logLikelihoodBefore = randomGraph.getLogLikelihood();
    randomGraph.applyGraphMove(move);
    double logLikelihoodAfter = randomGraph.getLogLikelihood();

    EXPECT_NEAR(actualLogLikelihoodRatio, logLikelihoodAfter - logLikelihoodBefore, 1E-6);
}

TEST_P(HDCSBMParametrizedTest, getLogLikelihoodRatio_forRemovedSelfLoop_returnCorrectLogLikelihoodRatio)
{
    randomGraph.applyGraphMove({{}, {{0, 0}}});
    GraphInf::GraphMove move = {{{0, 0}}, {}};
    double actualLogLikelihoodRatio = randomGraph.getLogLikelihoodRatioFromGraphMove(move);
    double logLikelihoodBefore = randomGraph.getLogLikelihood();
    randomGraph.applyGraphMove(move);
    double logLikelihoodAfter = randomGraph.getLogLikelihood();

    EXPECT_NEAR(actualLogLikelihoodRatio, logLikelihoodAfter - logLikelihoodBefore, 1E-6);
}

TEST_P(HDCSBMParametrizedTest, getLogLikelihoodRatio_forAddedEdge_returnCorrectLogLikelihoodRatio)
{
    GraphInf::GraphMove move = {{}, {{0, 2}}};
    double actualLogLikelihoodRatio = randomGraph.getLogLikelihoodRatioFromGraphMove(move);
    double logLikelihoodBefore = randomGraph.getLogLikelihood();
    randomGraph.applyGraphMove(move);
    double logLikelihoodAfter = randomGraph.getLogLikelihood();

    EXPECT_NEAR(actualLogLikelihoodRatio, logLikelihoodAfter - logLikelihoodBefore, 1E-6);
}

TEST_P(HDCSBMParametrizedTest, getLogLikelihoodRatio_forRemovedEdge_returnCorrectLogLikelihoodRatio)
{
    randomGraph.applyGraphMove({{}, {{0, 2}}});
    GraphInf::GraphMove move = {{{0, 2}}, {}};
    double actualLogLikelihoodRatio = randomGraph.getLogLikelihoodRatioFromGraphMove(move);
    double logLikelihoodBefore = randomGraph.getLogLikelihood();
    randomGraph.applyGraphMove(move);
    double logLikelihoodAfter = randomGraph.getLogLikelihood();

    EXPECT_NEAR(actualLogLikelihoodRatio, logLikelihoodAfter - logLikelihoodBefore, 1E-6);
}

TEST_P(HDCSBMParametrizedTest, getLogLikelihoodRatio_forRemovedAndAddedEdges_returnCorrectLogLikelihoodRatio)
{
    randomGraph.applyGraphMove({{}, {{0, 2}}});
    GraphInf::GraphMove move = {{{0, 2}}, {{0, 0}}};
    double actualLogLikelihoodRatio = randomGraph.getLogLikelihoodRatioFromGraphMove(move);
    double logLikelihoodBefore = randomGraph.getLogLikelihood();
    randomGraph.applyGraphMove(move);
    double logLikelihoodAfter = randomGraph.getLogLikelihood();

    EXPECT_NEAR(actualLogLikelihoodRatio, logLikelihoodAfter - logLikelihoodBefore, 1E-6);
}

// TEST_P(HDCSBMParametrizedTest, getLogLikelihoodRatio_forIdentityBlockMove_return0){
//
//     GraphInf::BlockIndex prevBlockIdx = randomGraph.getLabel(vertexIdx);
//     GraphInf::BlockIndex nextBlockIdx = prevBlockIdx;
//
//     GraphInf::BlockMove move = {vertexIdx, prevBlockIdx, nextBlockIdx};
//     EXPECT_NEAR(randomGraph.getLogLikelihoodRatioFromLabelMove(move), 0, 1E-6);
// }
//
// TEST_P(HDCSBMParametrizedTest, getLogLikelihoodRatio_forBlockMove_returnCorrectLogLikelihoodRatio){
//
//     GraphInf::BlockIndex prevBlockIdx = randomGraph.getLabel(vertexIdx);
//     GraphInf::BlockIndex nextBlockIdx = prevBlockIdx;
//     if (prevBlockIdx == randomGraph.getVertexCounts().size() - 1) nextBlockIdx --;
//     else nextBlockIdx ++;
//     GraphInf::BlockMove move = {vertexIdx, prevBlockIdx, nextBlockIdx};
//     double actualLogLikelihoodRatio = randomGraph.getLogLikelihoodRatioFromLabelMove(move);
//     double logLikelihoodBefore = randomGraph.getLogLikelihood();
//     randomGraph.applyLabelMove(move);
//     double logLikelihoodAfter = randomGraph.getLogLikelihood();
//
//     EXPECT_NEAR(actualLogLikelihoodRatio, logLikelihoodAfter - logLikelihoodBefore, 1E-6);
// }
//
// TEST_P(HDCSBMParametrizedTest, getLogLikelihoodRatio_forBlockMoveWithBlockCreation_returnCorrectLogLikelihoodRatio){
//
//     GraphInf::BlockIndex prevBlockIdx = randomGraph.getLabel(vertexIdx);
//     GraphInf::BlockIndex nextBlockIdx = randomGraph.getVertexCounts().size();
//
//     GraphInf::BlockMove move = {vertexIdx, prevBlockIdx, nextBlockIdx};
//     double actualLogLikelihoodRatio = randomGraph.getLogLikelihoodRatioFromLabelMove(move);
//
//     double logLikelihoodBefore = randomGraph.getLogLikelihood();
//     randomGraph.applyLabelMove(move);
//     double logLikelihoodAfter = randomGraph.getLogLikelihood();
//
//     EXPECT_NEAR(actualLogLikelihoodRatio, logLikelihoodAfter - logLikelihoodBefore, 1E-6);
// }
//
// TEST_P(HDCSBMParametrizedTest, getLogLikelihoodRatio_forBlockMoveWithBlockDestruction_returnCorrectLogLikelihoodRatio){
//
//     GraphInf::BlockIndex prevBlockIdx = randomGraph.getLabel(vertexIdx);
//     GraphInf::BlockIndex nextBlockIdx = randomGraph.getVertexCounts().size();
//
//     GraphInf::BlockMove move = {vertexIdx, prevBlockIdx, nextBlockIdx};
//     randomGraph.applyLabelMove(move);
//     move = {vertexIdx, nextBlockIdx, prevBlockIdx};
//     double actualLogLikelihoodRatio = randomGraph.getLogLikelihoodRatioFromLabelMove(move);
//     double logLikelihoodBefore = randomGraph.getLogLikelihood();
//     randomGraph.applyLabelMove(move);
//     double logLikelihoodAfter = randomGraph.getLogLikelihood();
//     EXPECT_NEAR(actualLogLikelihoodRatio, logLikelihoodAfter - logLikelihoodBefore, 1E-6);
// }
//
//
TEST_P(HDCSBMParametrizedTest, isCompatible_forGraphSampledFromSBM_returnTrue)
{
    randomGraph.sample();
    EXPECT_NO_THROW(randomGraph.checkConsistency());
    auto g = randomGraph.getState();
    EXPECT_TRUE(randomGraph.isCompatible(g));
}

TEST_P(HDCSBMParametrizedTest, isCompatible_forEmptyGraph_returnFalse)
{
    MultiGraph g(0);
    EXPECT_FALSE(randomGraph.isCompatible(g));
}

TEST_P(HDCSBMParametrizedTest, doingMetropolisHastingsWithGraph_expectNoConsistencyError)
{
    EXPECT_NO_THROW(doMetropolisHastingsSweepForGraph(randomGraph));
}

TEST_P(HDCSBMParametrizedTest, doingMetropolisHastingsWithLabels_expectNoConsistencyError)
{
    EXPECT_NO_THROW(doMetropolisHastingsSweepForLabels(randomGraph));
}

INSTANTIATE_TEST_SUITE_P(
    NestedDegreeCorrectedStochasticBlockModelFamilyTests,
    HDCSBMParametrizedTest,
    ::testing::Values(false, true));
