#include "gtest/gtest.h"
#include <list>
#include <algorithm>
#include <string>

#include "../fixtures.hpp"
#include "GraphInf/graph/prior/block.h"
#include "GraphInf/graph/prior/label_graph.h"
#include "GraphInf/graph/prior/labeled_degree.h"
#include "GraphInf/graph/dcsbm.h"
#include "GraphInf/types.h"
#include "GraphInf/utility/functions.h"
#include "BaseGraph/types.h"

using namespace std;
using namespace GraphInf;

class DCSBMParametrizedTest : public ::testing::TestWithParam<std::tuple<bool, bool, bool>>
{
public:
    const size_t NUM_VERTICES = 5, NUM_EDGES = 10, NUM_BLOCKS = 3;
    const bool canonical = false;
    DegreeCorrectedStochasticBlockModelFamily randomGraph = DegreeCorrectedStochasticBlockModelFamily(
        NUM_VERTICES, NUM_EDGES, NUM_BLOCKS, std::get<0>(GetParam()), std::get<1>(GetParam()), std::get<2>(GetParam()), canonical);

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
        GraphInf::BlockIndex block = randomGraph.getLabel(idx);
        if (block == randomGraph.getVertexCounts().size() - 1)
            return block - 1;
        else
            return block + 1;
    }

    void SetUp()
    {
        randomGraph.sample();
    }
};

TEST_P(DCSBMParametrizedTest, sampleState_graphChanges)
{
    for (size_t i = 0; i < 10; i++)
    {
        auto prevGraph = randomGraph.getState();
        randomGraph.sample();
        auto nextGraph = randomGraph.getState();
        EXPECT_FALSE(prevGraph == nextGraph);
    }
}

TEST_P(DCSBMParametrizedTest, sample_graphChanges)
{
    for (size_t i = 0; i < 10; i++)
    {
        auto prevGraph = randomGraph.getState();
        randomGraph.sample();
        auto nextGraph = randomGraph.getState();
        EXPECT_FALSE(prevGraph == nextGraph);
    }
}

TEST_P(DCSBMParametrizedTest, getLogLikelihood_returnNonZeroValue)
{
    EXPECT_TRUE(randomGraph.getLogLikelihood() <= 1e-14);
}

TEST_P(DCSBMParametrizedTest, applyMove_forAddedEdge)
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

TEST_P(DCSBMParametrizedTest, applyMove_forAddedSelfLoop)
{
    BaseGraph::Edge addedEdge = {0, 0};
    size_t addedEdgeMultBefore = randomGraph.getState().getEdgeMultiplicity(addedEdge.first, addedEdge.second);
    GraphInf::GraphMove move = {{}, {addedEdge}};
    randomGraph.applyGraphMove(move);
    size_t addedEdgeMultAfter = randomGraph.getState().getEdgeMultiplicity(addedEdge.first, addedEdge.second);
    EXPECT_EQ(addedEdgeMultAfter - 1, addedEdgeMultBefore);
}

TEST_P(DCSBMParametrizedTest, applyMove_forRemovedEdge)
{
    BaseGraph::Edge removedEdge = findEdge();
    size_t removedEdgeMultBefore = randomGraph.getState().getEdgeMultiplicity(removedEdge.first, removedEdge.second);
    GraphInf::GraphMove move = {{removedEdge}, {}};
    randomGraph.applyGraphMove(move);
    size_t removedEdgeMultAfter = randomGraph.getState().getEdgeMultiplicity(removedEdge.first, removedEdge.second);
    EXPECT_EQ(removedEdgeMultAfter + 1, removedEdgeMultBefore);
}

TEST_P(DCSBMParametrizedTest, applyMove_forRemovedEdgeAndAddedEdge)
{
    BaseGraph::Edge removedEdge = findEdge();
    BaseGraph::Edge addedEdge = {1, 1};
    while (removedEdge == addedEdge)
    {
        SetUp();
        removedEdge = findEdge();
    }
    size_t removedEdgeMultBefore = randomGraph.getState().getEdgeMultiplicity(removedEdge.first, removedEdge.second);
    size_t addedEdgeMultBefore = randomGraph.getState().getEdgeMultiplicity(addedEdge.first, addedEdge.second);
    GraphInf::GraphMove move = {{removedEdge}, {addedEdge}};
    randomGraph.applyGraphMove(move);
    size_t removedEdgeMultAfter = randomGraph.getState().getEdgeMultiplicity(removedEdge.first, removedEdge.second);
    size_t addedEdgeMultAfter = randomGraph.getState().getEdgeMultiplicity(addedEdge.first, addedEdge.second);
    EXPECT_EQ(removedEdgeMultAfter + 1, removedEdgeMultBefore);
    EXPECT_EQ(addedEdgeMultAfter - 1, addedEdgeMultBefore);
}

TEST_P(DCSBMParametrizedTest, applyMove_forNoEdgesAddedOrRemoved)
{
    GraphInf::GraphMove move = {{}, {}};
    randomGraph.applyGraphMove(move);
}

TEST_P(DCSBMParametrizedTest, applyMove_forIdentityBlockMove_doNothing)
{
    GraphInf::BlockIndex prevBlockIdx = randomGraph.getLabel(vertexIdx);
    GraphInf::BlockIndex nextBlockIdx = prevBlockIdx;

    GraphInf::BlockMove move = {vertexIdx, prevBlockIdx, nextBlockIdx};
    randomGraph.applyLabelMove(move);
}

TEST_P(DCSBMParametrizedTest, applyMove_forBlockMoveWithNoBlockCreation_changeBlockIdx)
{
    randomGraph.sample();
    GraphInf::BlockIndex prevBlockIdx = randomGraph.getLabel(vertexIdx);
    GraphInf::BlockIndex nextBlockIdx = prevBlockIdx;
    if (prevBlockIdx == randomGraph.getVertexCounts().size() - 1)
        nextBlockIdx--;
    else
        nextBlockIdx++;

    GraphInf::BlockMove move = {vertexIdx, prevBlockIdx, nextBlockIdx};
    EXPECT_EQ(randomGraph.getLabel(vertexIdx), prevBlockIdx);
    randomGraph.applyLabelMove(move);
    EXPECT_EQ(randomGraph.getLabel(vertexIdx), nextBlockIdx);
}

TEST_P(DCSBMParametrizedTest, applyMove_forBlockMoveWithBlockCreation_changeBlockIdxAndBlockCount)
{
    GraphInf::BlockIndex prevBlockIdx = randomGraph.getLabel(vertexIdx);
    GraphInf::BlockIndex nextBlockIdx = randomGraph.getVertexCounts().size();
    GraphInf::BlockMove move = {vertexIdx, prevBlockIdx, nextBlockIdx};
    EXPECT_EQ(randomGraph.getLabel(vertexIdx), prevBlockIdx);
    randomGraph.applyLabelMove(move);
    EXPECT_EQ(randomGraph.getLabel(vertexIdx), nextBlockIdx);
}

// TEST_P(DCSBMParametrizedTest, applyMove_forBlockMoveWithBlockDestruction_changeBlockIdxAndBlockCount){
//     GraphInf::BlockIndex prevBlockIdx = randomGraph.getVertexCounts().size();
//     GraphInf::BlockIndex nextBlockIdx = randomGraph.getLabel(vertexIdx);
//     GraphInf::BlockMove move = {vertexIdx, nextBlockIdx, prevBlockIdx};
//     randomGraph.applyLabelMove(move); // creating block before destroying it
//     move = {vertexIdx, prevBlockIdx, nextBlockIdx};
//     EXPECT_EQ(randomGraph.getLabel(vertexIdx), prevBlockIdx);
//     randomGraph.applyLabelMove(move);
//     EXPECT_EQ(randomGraph.getLabel(vertexIdx), nextBlockIdx);
// }

TEST_P(DCSBMParametrizedTest, getLogLikelihoodRatio_forAddedSelfLoop_returnCorrectLogLikelihoodRatio)
{
    GraphInf::GraphMove move = {{}, {{0, 0}}};
    double actualLogLikelihoodRatio = randomGraph.getLogLikelihoodRatioFromGraphMove(move);
    double logLikelihoodBefore = randomGraph.getLogLikelihood();
    randomGraph.applyGraphMove(move);
    double logLikelihoodAfter = randomGraph.getLogLikelihood();

    EXPECT_NEAR(actualLogLikelihoodRatio, logLikelihoodAfter - logLikelihoodBefore, 1E-6);
}

TEST_P(DCSBMParametrizedTest, getLogLikelihoodRatio_forRemovedSelfLoop_returnCorrectLogLikelihoodRatio)
{
    randomGraph.applyGraphMove({{}, {{0, 0}}});
    GraphInf::GraphMove move = {{{0, 0}}, {}};
    double actualLogLikelihoodRatio = randomGraph.getLogLikelihoodRatioFromGraphMove(move);
    double logLikelihoodBefore = randomGraph.getLogLikelihood();
    randomGraph.applyGraphMove(move);
    double logLikelihoodAfter = randomGraph.getLogLikelihood();

    EXPECT_NEAR(actualLogLikelihoodRatio, logLikelihoodAfter - logLikelihoodBefore, 1E-6);
}

TEST_P(DCSBMParametrizedTest, getLogLikelihoodRatio_forAddedEdge_returnCorrectLogLikelihoodRatio)
{
    GraphInf::GraphMove move = {{}, {{0, 2}}};
    double actualLogLikelihoodRatio = randomGraph.getLogLikelihoodRatioFromGraphMove(move);
    double logLikelihoodBefore = randomGraph.getLogLikelihood();
    randomGraph.applyGraphMove(move);
    double logLikelihoodAfter = randomGraph.getLogLikelihood();

    EXPECT_NEAR(actualLogLikelihoodRatio, logLikelihoodAfter - logLikelihoodBefore, 1E-6);
}

TEST_P(DCSBMParametrizedTest, getLogLikelihoodRatio_forRemovedEdge_returnCorrectLogLikelihoodRatio)
{
    randomGraph.applyGraphMove({{}, {{0, 2}}});
    GraphInf::GraphMove move = {{{0, 2}}, {}};
    double actualLogLikelihoodRatio = randomGraph.getLogLikelihoodRatioFromGraphMove(move);
    double logLikelihoodBefore = randomGraph.getLogLikelihood();
    randomGraph.applyGraphMove(move);
    double logLikelihoodAfter = randomGraph.getLogLikelihood();

    EXPECT_NEAR(actualLogLikelihoodRatio, logLikelihoodAfter - logLikelihoodBefore, 1E-6);
}

TEST_P(DCSBMParametrizedTest, getLogLikelihoodRatio_forRemovedAndAddedEdges_returnCorrectLogLikelihoodRatio)
{
    randomGraph.applyGraphMove({{}, {{0, 2}}});
    GraphInf::GraphMove move = {{{0, 2}}, {{0, 0}}};
    double actualLogLikelihoodRatio = randomGraph.getLogLikelihoodRatioFromGraphMove(move);
    double logLikelihoodBefore = randomGraph.getLogLikelihood();
    randomGraph.applyGraphMove(move);
    double logLikelihoodAfter = randomGraph.getLogLikelihood();

    EXPECT_NEAR(actualLogLikelihoodRatio, logLikelihoodAfter - logLikelihoodBefore, 1E-6);
}

TEST_P(DCSBMParametrizedTest, getLogLikelihoodRatio_forIdentityBlockMove_return0)
{

    GraphInf::BlockIndex prevBlockIdx = randomGraph.getLabel(vertexIdx);
    GraphInf::BlockIndex nextBlockIdx = prevBlockIdx;

    GraphInf::BlockMove move = {vertexIdx, prevBlockIdx, nextBlockIdx};
    EXPECT_NEAR(randomGraph.getLogLikelihoodRatioFromLabelMove(move), 0, 1E-6);
}

TEST_P(DCSBMParametrizedTest, getLogLikelihoodRatio_forBlockMove_returnCorrectLogLikelihoodRatio)
{

    GraphInf::BlockIndex prevBlockIdx = randomGraph.getLabel(vertexIdx);
    GraphInf::BlockIndex nextBlockIdx = prevBlockIdx;
    if (prevBlockIdx == randomGraph.getVertexCounts().size() - 1)
        nextBlockIdx--;
    else
        nextBlockIdx++;
    GraphInf::BlockMove move = {vertexIdx, prevBlockIdx, nextBlockIdx};
    double actualLogLikelihoodRatio = randomGraph.getLogLikelihoodRatioFromLabelMove(move);
    double logLikelihoodBefore = randomGraph.getLogLikelihood();
    randomGraph.applyLabelMove(move);
    double logLikelihoodAfter = randomGraph.getLogLikelihood();

    EXPECT_NEAR(actualLogLikelihoodRatio, logLikelihoodAfter - logLikelihoodBefore, 1E-6);
}

TEST_P(DCSBMParametrizedTest, getLogLikelihoodRatio_forBlockMoveWithBlockCreation_returnCorrectLogLikelihoodRatio)
{

    GraphInf::BlockIndex prevBlockIdx = randomGraph.getLabel(vertexIdx);
    GraphInf::BlockIndex nextBlockIdx = randomGraph.getVertexCounts().size();

    GraphInf::BlockMove move = {vertexIdx, prevBlockIdx, nextBlockIdx};
    double actualLogLikelihoodRatio = randomGraph.getLogLikelihoodRatioFromLabelMove(move);

    double logLikelihoodBefore = randomGraph.getLogLikelihood();
    randomGraph.applyLabelMove(move);
    double logLikelihoodAfter = randomGraph.getLogLikelihood();

    EXPECT_NEAR(actualLogLikelihoodRatio, logLikelihoodAfter - logLikelihoodBefore, 1E-6);
}

TEST_P(DCSBMParametrizedTest, getLogLikelihoodRatio_forBlockMoveWithBlockDestruction_returnCorrectLogLikelihoodRatio)
{

    GraphInf::BlockIndex prevBlockIdx = randomGraph.getLabel(vertexIdx);
    GraphInf::BlockIndex nextBlockIdx = randomGraph.getVertexCounts().size();

    GraphInf::BlockMove move = {vertexIdx, prevBlockIdx, nextBlockIdx};
    randomGraph.applyLabelMove(move);
    move = {vertexIdx, nextBlockIdx, prevBlockIdx};
    double actualLogLikelihoodRatio = randomGraph.getLogLikelihoodRatioFromLabelMove(move);
    double logLikelihoodBefore = randomGraph.getLogLikelihood();
    randomGraph.applyLabelMove(move);
    double logLikelihoodAfter = randomGraph.getLogLikelihood();
    EXPECT_NEAR(actualLogLikelihoodRatio, logLikelihoodAfter - logLikelihoodBefore, 1E-6);
}

TEST_P(DCSBMParametrizedTest, isCompatible_forGraphSampledFromSBM_returnTrue)
{
    randomGraph.sample();
    auto g = randomGraph.getState();
    EXPECT_TRUE(randomGraph.isCompatible(g));
}

TEST_P(DCSBMParametrizedTest, isCompatible_forEmptyGraph_returnFalse)
{
    MultiGraph g(0);
    EXPECT_FALSE(randomGraph.isCompatible(g));
}

TEST_P(DCSBMParametrizedTest, doingMetropolisHastingsWithGraph_expectNoConsistencyError)
{
    EXPECT_NO_THROW(doMetropolisHastingsSweepForGraph(randomGraph));
}

TEST_P(DCSBMParametrizedTest, doingMetropolisHastingsWithLabels_expectNoConsistencyError)
{
    EXPECT_NO_THROW(doMetropolisHastingsSweepForLabels(randomGraph));
}

TEST_P(DCSBMParametrizedTest, enumeratingAllGraphs_likelihoodIsNormalized)
{
    size_t N = 4, E = 4, B = 0;
    DegreeCorrectedStochasticBlockModelFamily g(
        N, E, B,
        std::get<0>(GetParam()),
        std::get<1>(GetParam()),
        std::get<2>(GetParam()),
        false);

    std::list<double> s;
    for (auto gg : enumerateAllGraphs(N, E))
    {
        g.setState(gg);
        s.push_back(g.getLogJoint());
    }
    if (not std::get<1>(GetParam()))
        EXPECT_NEAR(logSumExp(s) - g.getLabelLogJoint(), 0, 1e-6);
}

INSTANTIATE_TEST_SUITE_P(
    DegreeCorrectedStochasticBlockModelFamilyTests,
    DCSBMParametrizedTest,
    ::testing::Values(
        std::make_tuple(false, false, false),
        std::make_tuple(false, true, false),
        std::make_tuple(true, false, false),
        std::make_tuple(true, true, false),
        std::make_tuple(false, false, true),
        std::make_tuple(false, true, true),
        std::make_tuple(true, false, true),
        std::make_tuple(true, true, true)));
