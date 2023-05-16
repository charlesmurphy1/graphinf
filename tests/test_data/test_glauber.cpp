#include "gtest/gtest.h"
#include <list>

#include "GraphInf/data/dynamics/glauber.h"
#include "GraphInf/graph/erdosrenyi.h"
#include "GraphInf/graph/proposer/edge/hinge_flip.h"
#include "../fixtures.hpp"

namespace GraphInf
{

    class TestGlauberDynamics : public ::testing::Test
    {
    public:
        const double COUPLING = 0.0001;
        const std::list<std::vector<VertexState>> NEIGHBOR_STATES = {{1, 3}, {2, 2}, {3, 1}};
        const size_t LENGTH = 20;
        double avgk = 5;
        ErdosRenyiModel randomGraph = ErdosRenyiModel(100, 250);
        GlauberDynamics dynamics = GraphInf::GlauberDynamics(
            randomGraph, LENGTH, COUPLING);

        void SetUp()
        {
            dynamics.acceptSelfLoops(false);
        }
    };

    TEST_F(TestGlauberDynamics, getActivationProb_forEachStateTransition_returnCorrectProbability)
    {
        for (auto neighborState : NEIGHBOR_STATES)
        {
            EXPECT_EQ(
                sigmoid(2 * COUPLING * ((int)neighborState[1] - (int)neighborState[0])),
                dynamics.getActivationProb(neighborState));
        }
    }

    TEST_F(TestGlauberDynamics, getDeactivationProb_forEachStateTransition_returnCorrectProbability)
    {
        for (auto neighborState : NEIGHBOR_STATES)
        {
            EXPECT_EQ(sigmoid(
                          2 * COUPLING * ((int)neighborState[0] - (int)neighborState[1])),
                      dynamics.getDeactivationProb(neighborState));
        }
    }

    TEST_F(TestGlauberDynamics, getLogLikelihoodRatioFromGraphMove_forAddedEdge_returnCorrectValue)
    {
        dynamics.sample();
        GraphMove move = {{}, {{0, 1}}};
        double actual = dynamics.getLogLikelihoodRatioFromGraphMove(move);
        double logLikelihoodBefore = dynamics.getLogLikelihood();
        dynamics.applyGraphMove(move);
        double logLikelihoodAfter = dynamics.getLogLikelihood();
        EXPECT_NEAR(actual, logLikelihoodAfter - logLikelihoodBefore, 1e-6);
    }

    TEST_F(TestGlauberDynamics, getLogLikelihoodRatioFromGraphMove_forRemovedEdge_returnCorrectValue)
    {
        dynamics.sample();
        GraphMove move = {{{0, 1}}, {}};
        GraphMove reversedMove = {{}, {{0, 1}}};
        dynamics.applyGraphMove(reversedMove);
        double actual = dynamics.getLogLikelihoodRatioFromGraphMove(move);
        double logLikelihoodBefore = dynamics.getLogLikelihood();
        dynamics.applyGraphMove(move);
        double logLikelihoodAfter = dynamics.getLogLikelihood();
        EXPECT_NEAR(actual, logLikelihoodAfter - logLikelihoodBefore, 1e-6);
    }

    TEST_F(TestGlauberDynamics, afterSample_getCorrectNeighborState)
    {
        dynamics.sample();
        dynamics.checkConsistency();
    }

    TEST_F(TestGlauberDynamics, getLogLikelihood_returnCorrectLogLikelikehood)
    {
        dynamics.sample();
        auto past = dynamics.getPastStates();
        auto future = dynamics.getFutureStates();
        auto neighborState = dynamics.getNeighborsPastStates();

        double expected = dynamics.getLogLikelihood();
        double actual = 0;
        for (size_t t = 0; t < dynamics.getLength(); ++t)
        {
            for (auto vertex : dynamics.getGraph())
            {
                actual += log(dynamics.getTransitionProb(past[vertex][t], future[vertex][t], neighborState[vertex][t]));
            }
        }
        EXPECT_NEAR(expected, actual, 1E-6);
    }

    TEST_F(TestGlauberDynamics, getLogLikelihoodRatio_forSomeGraphMove_returnLogJointRatio)
    {
        dynamics.sample();
        auto graphMove = randomGraph.proposeGraphMove();
        double ratio = dynamics.getLogLikelihoodRatioFromGraphMove(graphMove);
        double logLikelihoodBefore = dynamics.getLogLikelihood();
        dynamics.applyGraphMove(graphMove);
        double logLikelihoodAfter = dynamics.getLogLikelihood();

        EXPECT_NEAR(ratio, logLikelihoodAfter - logLikelihoodBefore, 1e-6);
    }

    TEST_F(TestGlauberDynamics, metropolisGraphStep_forAcceptedMove_noConsistencyError)
    {
        dynamics.sample();
        MCMCSummary summary = {"none", 0, false};
        while (not summary.isAccepted)
            summary = dynamics.metropolisGraphStep();
        dynamics.checkConsistency();
    }

    TEST_F(TestGlauberDynamics, metropolisParamStep_noConsistencyError)
    {
        dynamics.sample();
        MCMCSummary summary = {"none", 0, false};
        while (not summary.isAccepted)
            summary = dynamics.metropolisParamStep();
        dynamics.checkConsistency();
    }
}
