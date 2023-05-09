#include "gtest/gtest.h"
#include <list>
#include <algorithm>
#include <string>
#include <cmath>

#include "../fixtures.hpp"
#include "GraphInf/graph/prior/block.h"
#include "GraphInf/graph/prior/label_graph.h"
#include "GraphInf/graph/sbm.h"
#include "GraphInf/types.h"
#include "GraphInf/utility/functions.h"
#include "GraphInf/rng.h"
#include "BaseGraph/types.h"

using namespace std;
using namespace GraphInf;

class SBMParametrizedTest : public ::testing::TestWithParam<std::tuple<bool, bool, bool>>
{
public:
    const size_t NUM_VERTICES = 50, NUM_EDGES = 100, NUM_BLOCKS = 3;
    const bool canonical = false;
    StochasticBlockModelFamily randomGraph = StochasticBlockModelFamily(
        NUM_VERTICES,
        NUM_EDGES,
        NUM_BLOCKS,
        std::get<0>(GetParam()),
        std::get<1>(GetParam()),
        canonical,
        std::get<2>(GetParam()));

    BaseGraph::VertexIndex vertex = 4;

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

    void SetUp() {}
};

TEST_P(SBMParametrizedTest, sampleState_graphChanges)
{
    for (size_t i = 0; i < 10; i++)
    {
        auto prevGraph = randomGraph.getState();
        randomGraph.sample();
        auto nextGraph = randomGraph.getState();
        EXPECT_FALSE(prevGraph == nextGraph);
    }
}

TEST_P(SBMParametrizedTest, sample_graphChanges)
{
    for (size_t i = 0; i < 10; i++)
    {
        auto prevGraph = randomGraph.getState();
        randomGraph.sample();
        auto nextGraph = randomGraph.getState();
        EXPECT_FALSE(prevGraph == nextGraph);
    }
}

TEST_P(SBMParametrizedTest, getLogLikelihood_returnNonZeroValue)
{
    EXPECT_TRUE(randomGraph.getLogLikelihood() < 0);
}

TEST_P(SBMParametrizedTest, applyMove_forAddedEdge)
{
    BaseGraph::Edge addedEdge = {0, 2};
    size_t addedEdgeMultBefore = randomGraph.getState().getEdgeMultiplicity(addedEdge.first, addedEdge.second);
    GraphInf::GraphMove move = {{}, {addedEdge}};
    randomGraph.applyGraphMove(move);
    size_t addedEdgeMultAfter = randomGraph.getState().getEdgeMultiplicity(addedEdge.first, addedEdge.second);
    EXPECT_EQ(addedEdgeMultAfter - 1, addedEdgeMultBefore);
}

TEST_P(SBMParametrizedTest, applyMove_forAddedSelfLoop)
{
    BaseGraph::Edge addedEdge = {0, 0};
    size_t addedEdgeMultBefore = randomGraph.getState().getEdgeMultiplicity(addedEdge.first, addedEdge.second);
    GraphInf::GraphMove move = {{}, {addedEdge}};
    randomGraph.applyGraphMove(move);
    size_t addedEdgeMultAfter = randomGraph.getState().getEdgeMultiplicity(addedEdge.first, addedEdge.second);
    EXPECT_EQ(addedEdgeMultAfter - 1, addedEdgeMultBefore);
}

TEST_P(SBMParametrizedTest, applyMove_forRemovedEdge)
{
    BaseGraph::Edge removedEdge = findEdge();
    size_t removedEdgeMultBefore = randomGraph.getState().getEdgeMultiplicity(removedEdge.first, removedEdge.second);
    GraphInf::GraphMove move = {{removedEdge}, {}};
    randomGraph.applyGraphMove(move);
    size_t removedEdgeMultAfter = randomGraph.getState().getEdgeMultiplicity(removedEdge.first, removedEdge.second);
    EXPECT_EQ(removedEdgeMultAfter + 1, removedEdgeMultBefore);
}

TEST_P(SBMParametrizedTest, applyMove_forRemovedEdgeAndAddedEdge)
{
    BaseGraph::Edge addedEdge = {10, 11};
    BaseGraph::Edge removedEdge = findEdge();
    while (addedEdge == removedEdge)
        removedEdge = findEdge();
    size_t removedEdgeMultBefore = randomGraph.getState().getEdgeMultiplicity(removedEdge.first, removedEdge.second);
    size_t addedEdgeMultBefore = randomGraph.getState().getEdgeMultiplicity(addedEdge.first, addedEdge.second);
    GraphInf::GraphMove move = {{removedEdge}, {addedEdge}};
    randomGraph.applyGraphMove(move);
    size_t removedEdgeMultAfter = randomGraph.getState().getEdgeMultiplicity(removedEdge.first, removedEdge.second);
    size_t addedEdgeMultAfter = randomGraph.getState().getEdgeMultiplicity(addedEdge.first, addedEdge.second);
    EXPECT_EQ(removedEdgeMultAfter + 1, removedEdgeMultBefore);
    EXPECT_EQ(addedEdgeMultAfter - 1, addedEdgeMultBefore);
}

TEST_P(SBMParametrizedTest, applyMove_forNoEdgesAddedOrRemoved)
{
    GraphInf::GraphMove move = {{}, {}};
    randomGraph.applyGraphMove(move);
}

TEST_P(SBMParametrizedTest, applyMove_forIdentityBlockMove_doNothing)
{
    GraphInf::BlockIndex prevLabel = randomGraph.getLabel(vertex);
    GraphInf::BlockIndex nextLabel = prevLabel;

    GraphInf::BlockMove move = {vertex, prevLabel, nextLabel};
    randomGraph.applyLabelMove(move);
}

TEST_P(SBMParametrizedTest, applyMove_forBlockMoveWithNoBlockCreation_changeBlockIdx)
{
    GraphInf::BlockIndex prevLabel = randomGraph.getLabel(vertex);
    GraphInf::BlockIndex nextLabel = findBlockMove(vertex);

    GraphInf::BlockMove move = {vertex, prevLabel, nextLabel};
    randomGraph.applyLabelMove(move);
    EXPECT_NE(randomGraph.getLabel(vertex), prevLabel);
    EXPECT_EQ(randomGraph.getLabel(vertex), nextLabel);
}

TEST_P(SBMParametrizedTest, applyMove_forBlockMoveWithBlockCreation_changeBlockIdxAndBlockCount)
{
    GraphInf::BlockIndex prevLabel = randomGraph.getLabel(vertex);
    GraphInf::BlockIndex nextLabel = randomGraph.getVertexCounts().size();
    GraphInf::BlockMove move = {vertex, prevLabel, nextLabel};
    randomGraph.applyLabelMove(move);
    EXPECT_NE(randomGraph.getLabel(vertex), prevLabel);
    EXPECT_EQ(randomGraph.getLabel(vertex), nextLabel);
}

TEST_P(SBMParametrizedTest, applyMove_forBlockMoveWithBlockDestruction_changeBlockIdxAndBlockCount)
{
    GraphInf::BlockIndex prevLabel = randomGraph.getVertexCounts().size();
    GraphInf::BlockIndex nextLabel = randomGraph.getLabel(vertex);
    GraphInf::BlockMove move = {vertex, nextLabel, prevLabel};
    randomGraph.applyLabelMove(move); // creating block before destroying it
    move = {vertex, prevLabel, nextLabel};
    randomGraph.applyLabelMove(move);
    EXPECT_EQ(randomGraph.getLabel(vertex), nextLabel);
    EXPECT_NE(randomGraph.getLabel(vertex), prevLabel);
}

TEST_P(SBMParametrizedTest, getLogLikelihoodRatio_forAddedSelfLoop_returnCorrectLogLikelihoodRatio)
{
    GraphInf::GraphMove move = {{}, {{0, 0}}};
    double actualLogLikelihoodRatio = randomGraph.getLogLikelihoodRatioFromGraphMove(move);
    double logLikelihoodBefore = randomGraph.getLogLikelihood();
    randomGraph.applyGraphMove(move);
    double logLikelihoodAfter = randomGraph.getLogLikelihood();

    EXPECT_NEAR(actualLogLikelihoodRatio, logLikelihoodAfter - logLikelihoodBefore, 1E-6);
}

TEST_P(SBMParametrizedTest, getLogLikelihoodRatio_forRemovedSelfLoop_returnCorrectLogLikelihoodRatio)
{
    randomGraph.applyGraphMove({{}, {{0, 0}}});
    GraphInf::GraphMove move = {{{0, 0}}, {}};
    double actualLogLikelihoodRatio = randomGraph.getLogLikelihoodRatioFromGraphMove(move);
    double logLikelihoodBefore = randomGraph.getLogLikelihood();
    randomGraph.applyGraphMove(move);
    double logLikelihoodAfter = randomGraph.getLogLikelihood();

    EXPECT_NEAR(actualLogLikelihoodRatio, logLikelihoodAfter - logLikelihoodBefore, 1E-6);
}

TEST_P(SBMParametrizedTest, getLogLikelihoodRatio_forAddedEdge_returnCorrectLogLikelihoodRatio)
{
    GraphInf::GraphMove move = {{}, {{0, 2}}};
    double actualLogLikelihoodRatio = randomGraph.getLogLikelihoodRatioFromGraphMove(move);
    double logLikelihoodBefore = randomGraph.getLogLikelihood();
    randomGraph.applyGraphMove(move);
    double logLikelihoodAfter = randomGraph.getLogLikelihood();

    EXPECT_NEAR(actualLogLikelihoodRatio, logLikelihoodAfter - logLikelihoodBefore, 1E-6);
}

TEST_P(SBMParametrizedTest, getLogLikelihoodRatio_forRemovedEdge_returnCorrectLogLikelihoodRatio)
{
    randomGraph.applyGraphMove({{}, {{0, 2}}});
    GraphInf::GraphMove move = {{{0, 2}}, {}};
    double actualLogLikelihoodRatio = randomGraph.getLogLikelihoodRatioFromGraphMove(move);
    double logLikelihoodBefore = randomGraph.getLogLikelihood();
    randomGraph.applyGraphMove(move);
    double logLikelihoodAfter = randomGraph.getLogLikelihood();

    EXPECT_NEAR(actualLogLikelihoodRatio, logLikelihoodAfter - logLikelihoodBefore, 1E-6);
}

TEST_P(SBMParametrizedTest, getLogLikelihoodRatio_forRemovedAndAddedEdges_returnCorrectLogLikelihoodRatio)
{
    randomGraph.applyGraphMove({{}, {{0, 2}}});
    GraphInf::GraphMove move = {{{0, 2}}, {{0, 0}}};
    double actualLogLikelihoodRatio = randomGraph.getLogLikelihoodRatioFromGraphMove(move);
    double logLikelihoodBefore = randomGraph.getLogLikelihood();
    randomGraph.applyGraphMove(move);
    double logLikelihoodAfter = randomGraph.getLogLikelihood();

    EXPECT_NEAR(actualLogLikelihoodRatio, logLikelihoodAfter - logLikelihoodBefore, 1E-6);
}

TEST_P(SBMParametrizedTest, getLogLikelihoodRatio_forIdentityBlockMove_return0)
{

    GraphInf::BlockIndex prevLabel = randomGraph.getLabel(vertex);
    GraphInf::BlockIndex nextLabel = prevLabel;

    GraphInf::BlockMove move = {vertex, prevLabel, nextLabel};

    EXPECT_NEAR(randomGraph.getLogLikelihoodRatioFromLabelMove(move), 0, 1E-6);
}

TEST_P(SBMParametrizedTest, getLogLikelihoodRatio_forBlockMove_returnCorrectLogLikelihoodRatio)
{
    GraphInf::BlockIndex prevLabel = randomGraph.getLabel(vertex);
    GraphInf::BlockIndex nextLabel = findBlockMove(vertex);
    GraphInf::BlockMove move = {vertex, prevLabel, nextLabel};
    double actualLogLikelihoodRatio = randomGraph.getLogLikelihoodRatioFromLabelMove(move);
    double logLikelihoodBefore = randomGraph.getLogLikelihood();
    randomGraph.applyLabelMove(move);
    double logLikelihoodAfter = randomGraph.getLogLikelihood();

    EXPECT_NEAR(actualLogLikelihoodRatio, logLikelihoodAfter - logLikelihoodBefore, 1E-6);
}

TEST_P(SBMParametrizedTest, getLogLikelihoodRatio_forBlockMoveWithBlockCreation_returnCorrectLogLikelihoodRatio)
{

    GraphInf::BlockIndex prevLabel = randomGraph.getLabel(vertex);
    GraphInf::BlockIndex nextLabel = randomGraph.getVertexCounts().size();

    GraphInf::BlockMove move = {vertex, prevLabel, nextLabel, 1};

    double actualLogLikelihoodRatio = randomGraph.getLogLikelihoodRatioFromLabelMove(move);
    double logLikelihoodBefore = randomGraph.getLogLikelihood();
    randomGraph.applyLabelMove(move);
    double logLikelihoodAfter = randomGraph.getLogLikelihood();
    EXPECT_NEAR(actualLogLikelihoodRatio, logLikelihoodAfter - logLikelihoodBefore, 1E-6);
}

TEST_P(SBMParametrizedTest, getLogLikelihoodRatio_forBlockMoveWithBlockDestruction_returnCorrectLogLikelihoodRatio)
{

    GraphInf::BlockIndex prevLabel = randomGraph.getLabel(vertex);
    GraphInf::BlockIndex nextLabel = randomGraph.getVertexCounts().size();

    GraphInf::BlockMove move = {vertex, prevLabel, nextLabel, -1};
    randomGraph.applyLabelMove(move);
    move = {vertex, nextLabel, prevLabel};

    double actualLogLikelihoodRatio = randomGraph.getLogLikelihoodRatioFromLabelMove(move);

    double logLikelihoodBefore = randomGraph.getLogLikelihood();
    randomGraph.applyLabelMove(move);
    double logLikelihoodAfter = randomGraph.getLogLikelihood();
    EXPECT_NEAR(actualLogLikelihoodRatio, logLikelihoodAfter - logLikelihoodBefore, 1E-6);
}

TEST_P(SBMParametrizedTest, isCompatible_forGraphSampledFromSBM_returnTrue)
{
    randomGraph.sample();
    auto g = randomGraph.getState();
    EXPECT_TRUE(randomGraph.isCompatible(g));
}

TEST_P(SBMParametrizedTest, isCompatible_forEmptyGraph_returnFalse)
{
    MultiGraph g(0);
    EXPECT_FALSE(randomGraph.isCompatible(g));
}

TEST_P(SBMParametrizedTest, setLabels_forSomeRandomLabels_returnConsistentState)
{
    size_t N = randomGraph.getSize();
    size_t B = randomGraph.getLabelCount();
    std::vector<BlockIndex> newLabels(N);
    std::uniform_int_distribution<BlockIndex> dist(0, B - 1);
    for (size_t v = 0; v < N; ++v)
        newLabels[v] = dist(rng);
    randomGraph.setLabels(newLabels, false);
    EXPECT_EQ(randomGraph.getLabels(), newLabels);
    EXPECT_NO_THROW(randomGraph.checkConsistency());
}

TEST_P(SBMParametrizedTest, doingMetropolisHastingsWithGraph_expectNoConsistencyError)
{
    EXPECT_NO_THROW(doMetropolisHastingsSweepForGraph(randomGraph));
}

TEST_P(SBMParametrizedTest, doingMetropolisHastingsWithLabels_expectNoConsistencyError)
{
    EXPECT_NO_THROW(doMetropolisHastingsSweepForLabels(randomGraph));
}

TEST_P(SBMParametrizedTest, enumeratingAllGraphs_likelihoodIsNormalized)
{
    size_t N = 4, E = 4, B = 0;
    StochasticBlockModelFamily g(
        N, E, B,
        std::get<0>(GetParam()),
        std::get<1>(GetParam()),
        false,
        std::get<2>(GetParam()));

    std::list<double> s;
    for (auto gg : enumerateAllGraphs(N, E))
    {
        g.setState(gg);
        s.push_back(g.getLogJoint());
    }
    EXPECT_NEAR(logSumExp(s) - g.getLabelLogJoint(), 0, 1e-6);
}

INSTANTIATE_TEST_SUITE_P(
    StochasticBlockModelFamilyTests,
    SBMParametrizedTest,
    ::testing::Values(
        std::make_tuple(false, false, false),
        std::make_tuple(false, true, false),
        std::make_tuple(true, false, false),
        std::make_tuple(true, true, false),
        std::make_tuple(false, false, true),
        std::make_tuple(false, true, true),
        std::make_tuple(true, false, true),
        std::make_tuple(true, true, true)));

// TEST(SBMTest, sampleSBM_return)
// {
//     seedWithTime();
//     StochasticBlockModelFamily graph = {5, 5, 0, true, false, false, false, false, true};
//     EXPECT_EQ(graph.getEdgeCount(), 5);
// }

TEST(SBMTest, construction_returnSafeObject)
{
    std::vector<size_t> sizes = {10, 20, 30};
    size_t edgeCount = 100;
    double assortativity = 0.8;
    std::vector<BlockIndex> blocks = getPlantedBlocks(sizes);
    LabelGraph labelGraph = getPlantedLabelGraph(sizes.size(), edgeCount);
    StochasticBlockModel randomGraph = StochasticBlockModel(blocks, labelGraph);
    EXPECT_NO_THROW(randomGraph.checkSafety());
    randomGraph.sample();
    EXPECT_NO_THROW(randomGraph.checkConsistency());
}

TEST(UniformSBMTest, construction_returnSafeObject)
{
    seedWithTime();
    for (size_t i = 0; i < 1; ++i)
    {
        StochasticBlockModelFamily randomGraph(100, 250, 0, true, true);
        EXPECT_NO_THROW(randomGraph.checkSafety());
        randomGraph.sample();
        EXPECT_NO_THROW(randomGraph.checkConsistency());
    }
}

TEST(PlantedPartitionModelTest, constructor1_noThrow)
{
    size_t edgeCount = 100;
    double assortativity = 0.8;
    PlantedPartitionModel randomGraph = PlantedPartitionModel({10, 20, 30}, edgeCount, assortativity);
    randomGraph.sample();
    EXPECT_NO_THROW(randomGraph.checkConsistency());
}

TEST(PlantedPartitionModelTest, constructor2_noThrow)
{
    size_t edgeCount = 100;
    double assortativity = 0.8;
    PlantedPartitionModel randomGraph = PlantedPartitionModel(60, edgeCount, 3, assortativity);
    randomGraph.sample();
    EXPECT_NO_THROW(randomGraph.checkConsistency());
}
