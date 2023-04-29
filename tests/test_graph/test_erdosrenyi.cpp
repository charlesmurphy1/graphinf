#include "gtest/gtest.h"
#include <list>
#include <algorithm>
#include <string>

#include "../fixtures.hpp"
#include "GraphInf/graph/prior/edge_count.h"
#include "GraphInf/graph/erdosrenyi.h"
#include "GraphInf/types.h"
#include "GraphInf/utility/functions.h"
#include "BaseGraph/types.h"

using namespace std;
using namespace GraphInf;

class ErdosRenyiModelTest : public ::testing::Test
{
public:
    const size_t NUM_VERTICES = 50, NUM_EDGES = 50;
    ErdosRenyiModel randomGraph = ErdosRenyiModel(NUM_VERTICES, NUM_EDGES);
    void SetUp()
    {
        randomGraph.sample();
    }
};

TEST_F(ErdosRenyiModelTest, sample_getGraphWithCorrectNumberOfEdges)
{
    randomGraph.sample();
    EXPECT_EQ(randomGraph.getState().getTotalEdgeNumber(), randomGraph.getEdgeCount());
}

TEST_F(ErdosRenyiModelTest, getLogLikelihoodRatioFromGraphMove_forAddedEdge_returnCorrectLogLikelihoodRatio)
{
    auto graph = randomGraph.getState();

    GraphMove move = {};
    for (auto vertex : graph)
    {
        if (graph.getEdgeMultiplicity(0, vertex) == 0)
        {
            move.addedEdges.push_back({0, vertex});
            break;
        }
    }
    double actualLogLikelihoodRatio = randomGraph.getLogLikelihoodRatioFromGraphMove(move);
    double logLikelihoodBefore = randomGraph.getLogLikelihood();
    randomGraph.applyGraphMove(move);
    double logLikelihoodAfter = randomGraph.getLogLikelihood();
    EXPECT_NEAR(actualLogLikelihoodRatio, logLikelihoodAfter - logLikelihoodBefore, 1E-6);
}

TEST_F(ErdosRenyiModelTest, getLogLikelihoodRatioFromGraphMove_forRemovedEdge_returnCorrectLogLikelihoodRatio)
{
    auto graph = randomGraph.getState();

    GraphMove move = {};
    for (auto neighbor : graph.getOutNeighbours(0))
    {
        move.removedEdges.push_back({0, neighbor});
        break;
    }

    double actualLogLikelihoodRatio = randomGraph.getLogLikelihoodRatioFromGraphMove(move);
    double logLikelihoodBefore = randomGraph.getLogLikelihood();
    randomGraph.applyGraphMove(move);
    double logLikelihoodAfter = randomGraph.getLogLikelihood();
    EXPECT_NEAR(actualLogLikelihoodRatio, logLikelihoodAfter - logLikelihoodBefore, 1E-6);
}

TEST_F(ErdosRenyiModelTest, isCompatible_forGraphSampledFromSBM_returnTrue)
{
    randomGraph.sample();
    auto g = randomGraph.getState();
    EXPECT_TRUE(randomGraph.isCompatible(g));
}

TEST_F(ErdosRenyiModelTest, isCompatible_forEmptyGraph_returnFalse)
{
    MultiGraph g(0);
    EXPECT_FALSE(randomGraph.isCompatible(g));
}

TEST_F(ErdosRenyiModelTest, doingMetropolisHastingsWithGraph_expectNoConsistencyError)
{
    EXPECT_NO_THROW(doMetropolisHastingsSweepForGraph(randomGraph));
}

TEST_F(ErdosRenyiModelTest, enumeratingAllGraphs_likelihoodIsNormalized)
{
    ErdosRenyiModel g(3, 3);

    std::list<double> s;
    for (auto gg : enumerateAllGraphs(3, 3))
    {
        g.setState(gg);
        s.push_back(g.getLogJoint());
    }
    EXPECT_NEAR(logSumExp(s), 0, 1e-6);
}
