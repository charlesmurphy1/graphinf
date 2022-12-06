#include "gtest/gtest.h"
#include <cmath>
#include <random>
#include <time.h>

#include "GraphInf/mcmc/mcmc.h"
#include "GraphInf/rng.h"
#include "../fixtures.hpp"


namespace GraphInf{

class TestMCMC: public::testing::Test{
public:
    DummyMCMC mcmc = DummyMCMC();
    bool expectConsistencyError = false;
    void SetUp(){
        seed(time(NULL));
    }
    void TearDown(){
        if (not expectConsistencyError)
            mcmc.checkConsistency();
    }

};

TEST_F(TestMCMC, doMHSweep_for42Burn_mcmcStateIsUpdated){
    mcmc.doMHSweep(42);
    EXPECT_EQ(mcmc.getNumSteps(), 42);
    EXPECT_EQ(mcmc.getNumSweeps(), 1);

    mcmc.doMHSweep(42);
    EXPECT_EQ(mcmc.getNumSteps(), 84);
    EXPECT_EQ(mcmc.getNumSweeps(), 2);

    EXPECT_EQ(mcmc.getLastLogAcceptance(), -log(2));
    EXPECT_EQ(mcmc.getLastLogJointRatio(), 0);
}


} // GraphInf
