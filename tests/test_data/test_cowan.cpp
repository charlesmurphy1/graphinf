#include "gtest/gtest.h"
#include <list>

#include "GraphInf/data/dynamics/cowan.h"
#include "GraphInf/data/types.h"
#include "GraphInf/graph/erdosrenyi.h"
#include "GraphInf/graph/proposer/edge/hinge_flip.h"
#include "../fixtures.hpp"

namespace GraphInf
{

    class TestCowanDynamics : public ::testing::Test
    {
    public:
        const double A = 1., NU = 7., MU = 1., ETA = 0.5;
        const size_t LENGTH = 20;
        const std::list<std::vector<VertexState>> NEIGHBOR_STATES = {{1, 3}, {2, 2}, {3, 1}, {2, 0}};
        ErdosRenyiModel randomGraph = ErdosRenyiModel(10, 10);
        GraphInf::CowanDynamics dynamics = GraphInf::CowanDynamics(
            randomGraph, LENGTH, NU, A, MU, ETA, 0, 0);
    };

    TEST_F(TestCowanDynamics, getActivationProb_forEachStateTransition_returnCorrectProbability)
    {
        for (const auto &neighbor_state : NEIGHBOR_STATES)
            EXPECT_EQ(sigmoid(A * (NU * neighbor_state[1] - MU)), dynamics.getActivationProb(neighbor_state));
    }

    TEST_F(TestCowanDynamics, getDeactivationProb_forEachStateTransition_returnCorrectProbability)
    {
        for (auto neighbor_state : NEIGHBOR_STATES)
            EXPECT_EQ(ETA, dynamics.getDeactivationProb(neighbor_state));
    }

    TEST_F(TestCowanDynamics, afterSample_getCorrectNeighborState)
    {
        dynamics.sample();
        dynamics.checkConsistency();
    }

    TEST_F(TestCowanDynamics, getLogLikelihood_returnCorrectLogLikelikehood)
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

    TEST_F(TestCowanDynamics, getLogLikelihoodRatio_forSomeGraphMove_returnLogJointRatio)
    {
        dynamics.sample();
        auto graphMove = randomGraph.proposeGraphMove();
        double ratio = dynamics.getLogLikelihoodRatioFromGraphMove(graphMove);
        double logLikelihoodBefore = dynamics.getLogLikelihood();
        dynamics.applyGraphMove(graphMove);
        double logLikelihoodAfter = dynamics.getLogLikelihood();

        EXPECT_NEAR(ratio, logLikelihoodAfter - logLikelihoodBefore, 1e-6);
    }

} /* GraphInf */
