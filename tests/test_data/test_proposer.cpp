
#include <gtest/gtest.h>
#include "GraphInf/data/proposer.h"

namespace GraphInf
{

    class TestStepParamProposer : public ::testing::Test
    {
    public:
        double stepSize = 0.1, p = 0.5;
        StepParamProposer proposer = StepParamProposer(stepSize, p);

        void SetUp() {}
        void TearDown() {}
    };

    TEST_F(TestStepParamProposer, proposeMove_returnMove)
    {
        auto move = proposer.proposeMove();
    }

    TEST_F(TestStepParamProposer, logProposal_forSomeRandomMove_returnProb)
    {
        auto move = proposer.proposeMove();
        double prob = proposer.logProposal(move);
        if (move > 0)
            EXPECT_EQ(prob, log(p));
        else
            EXPECT_EQ(prob, log(1 - p));
    }

    TEST_F(TestStepParamProposer, logProposalRatio_forSomeRandomMove_returnRatio)
    {
        auto move = proposer.proposeMove();
        double ratio = proposer.logProposalRatio(move);
        if (p == 0.5)
            EXPECT_EQ(ratio, 0);
    }

    class TestGaussianParamProposer : public ::testing::Test
    {
    public:
        double mean = 0, stddev = 1;
        GaussianParamProposer proposer = GaussianParamProposer(mean, stddev);

        void SetUp() {}
        void TearDown() {}
    };

    TEST_F(TestGaussianParamProposer, proposeMove_returnMove)
    {
        auto move = proposer.proposeMove();
    }

    TEST_F(TestGaussianParamProposer, logProposal_forSomeRandomMove_returnProb)
    {
        auto move = proposer.proposeMove();
        double prob = proposer.logProposal(move);
        EXPECT_EQ(prob, -pow((move - mean) / stddev, 2) - 0.5 * log(2 * PI * stddev * stddev));
    }

    TEST_F(TestGaussianParamProposer, logProposalRatio_forSomeRandomMove_returnRatio)
    {
        auto move = proposer.proposeMove();
        double ratio = proposer.logProposalRatio(move);
        if (mean == 0)
            EXPECT_EQ(ratio, 0);
    }
    class TestMultiParamProposer : public ::testing::Test
    {
    public:
        MultiParamProposer proposer = MultiParamProposer();

        void SetUp()
        {
            proposer.insertGaussianProposer("x", 1., 0., 0.1);
            proposer.insertStepProposer("y", 1, 0.1, 0.5);
        }
        void TearDown() {}
    };

    TEST_F(TestMultiParamProposer, proposeMove_returnMove)
    {
        auto move = proposer.proposeMove();
    }

    TEST_F(TestMultiParamProposer, logProposalRatio_forSomeRandomMove_returnRatio)
    {
        auto move = proposer.proposeMove("x");
        double ratio = proposer.logProposalRatio(move);
        EXPECT_EQ(ratio, 0);

        move = proposer.proposeMove("y");
        ratio = proposer.logProposalRatio(move);
        EXPECT_EQ(ratio, 0);

        move = proposer.proposeMove();
        ratio = proposer.logProposalRatio(move);
        EXPECT_EQ(ratio, 0);
    }
}