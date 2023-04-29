#include "gtest/gtest.h"
#include <list>
#include <algorithm>
#include <string>

#include "../fixtures.hpp"
#include "GraphInf/graph/prior/edge_count.h"
#include "GraphInf/graph/prior/degree.h"
#include "GraphInf/graph/configuration.h"
#include "GraphInf/types.h"
#include "GraphInf/utility/functions.h"
#include "GraphInf/utility/integer_partition.h"
#include "BaseGraph/types.h"

using namespace std;
using namespace GraphInf;

class CMParametrizedTest : public ::testing::TestWithParam<bool>
{
public:
    const size_t NUM_VERTICES = 5, NUM_EDGES = 10;
    ConfigurationModelFamily randomGraph = ConfigurationModelFamily(NUM_VERTICES, NUM_EDGES, GetParam());
    void SetUp()
    {
        randomGraph.sample();
    }
};

TEST_P(CMParametrizedTest, sample_getStateWithCorrectNumberOfEdges)
{
    EXPECT_EQ(randomGraph.getState().getTotalEdgeNumber(), randomGraph.getEdgeCount());
}

TEST_P(CMParametrizedTest, getLogLikelihood_returnNonPositiveValue)
{
    EXPECT_LE(randomGraph.getLogLikelihood(), 0);
}

TEST_P(CMParametrizedTest, getLogLikelihoodRatioFromGraphMove_forAddedEdge_returnCorrectValue)
{
    GraphMove move = {{}, {{0, 1}}};

    double actualLogLikelihoodRatio = randomGraph.getLogLikelihoodRatioFromGraphMove(move);
    double logLikelihoodBefore = randomGraph.getLogLikelihood();
    randomGraph.applyGraphMove(move);
    double logLikelihoodAfter = randomGraph.getLogLikelihood();

    EXPECT_NEAR(actualLogLikelihoodRatio, logLikelihoodAfter - logLikelihoodBefore, 1e-6);
}

TEST_P(CMParametrizedTest, getLogLikelihoodRatioFromGraphMove_forAddedSelfLoop_returnCorrectValue)
{
    GraphMove move = {{}, {{0, 0}}};

    double actualLogLikelihoodRatio = randomGraph.getLogLikelihoodRatioFromGraphMove(move);
    double logLikelihoodBefore = randomGraph.getLogLikelihood();
    randomGraph.applyGraphMove(move);
    double logLikelihoodAfter = randomGraph.getLogLikelihood();

    EXPECT_NEAR(actualLogLikelihoodRatio, logLikelihoodAfter - logLikelihoodBefore, 1e-6);
}

TEST_P(CMParametrizedTest, getLogLikelihoodRatioFromGraphMove_forRemovedEdge_returnCorrectValue)
{
    GraphMove move = {{{0, 1}}, {}};
    randomGraph.applyGraphMove({{}, {{0, 1}}});
    double actualLogLikelihoodRatio = randomGraph.getLogLikelihoodRatioFromGraphMove(move);
    double logLikelihoodBefore = randomGraph.getLogLikelihood();
    randomGraph.applyGraphMove(move);
    double logLikelihoodAfter = randomGraph.getLogLikelihood();

    EXPECT_NEAR(actualLogLikelihoodRatio, logLikelihoodAfter - logLikelihoodBefore, 1e-6);
}

TEST_P(CMParametrizedTest, getLogLikelihoodRatioFromGraphMove_forRemovedSelfLoop_returnCorrectValue)
{
    GraphMove move = {{{0, 0}}, {}};
    randomGraph.applyGraphMove({{}, {{0, 0}}});
    double actualLogLikelihoodRatio = randomGraph.getLogLikelihoodRatioFromGraphMove(move);
    double logLikelihoodBefore = randomGraph.getLogLikelihood();
    randomGraph.applyGraphMove(move);
    double logLikelihoodAfter = randomGraph.getLogLikelihood();

    EXPECT_NEAR(actualLogLikelihoodRatio, logLikelihoodAfter - logLikelihoodBefore, 1e-6);
}

TEST_P(CMParametrizedTest, isCompatible_forGraphSampledFromSBM_returnTrue)
{
    randomGraph.sample();
    auto g = randomGraph.getState();
    EXPECT_TRUE(randomGraph.isCompatible(g));
}

TEST_P(CMParametrizedTest, isCompatible_forEmptyGraph_returnFalse)
{
    MultiGraph g(0);
    EXPECT_FALSE(randomGraph.isCompatible(g));
}

TEST_P(CMParametrizedTest, doingMetropolisHastingsWithGraph_expectNoConsistencyError)
{
    EXPECT_NO_THROW(doMetropolisHastingsSweepForGraph(randomGraph));
}

TEST_P(CMParametrizedTest, enumeratingAllGraphs_likelihoodIsNormalized)
{
    ConfigurationModelFamily g(3, 3, GetParam());

    std::list<double> s;
    for (auto gg : enumerateAllGraphs(3, 3))
    {
        g.setState(gg);
        double logPrior = g.getLogPrior();
        if (GetParam()) // if GetParam() [using hyperprior], then compute logPrior with exact value of number of partitions
            logPrior = -logMultinomialCoefficient(g.getDegreePrior().getDegreeCounts().getValues()) - log_q(2 * g.getEdgeCount(), g.getSize(), true);
        s.push_back(g.getLogLikelihood() + logPrior);
    }
    EXPECT_NEAR(logSumExp(s), 0, 1e-6);
}

INSTANTIATE_TEST_SUITE_P(
    ConfigurationModelFamilyTests,
    CMParametrizedTest,
    ::testing::Values(false, true));

TEST(CMTests, instanciateConfigurationModel_forRegularSequence)
{
    std::vector<size_t> degrees(100, 5);
    ConfigurationModel graph(degrees);
    EXPECT_EQ(graph.getSize(), 100);
    EXPECT_EQ(graph.getDegrees(), degrees);
}
