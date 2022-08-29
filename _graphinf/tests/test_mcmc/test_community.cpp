#include "gtest/gtest.h"

#include "GraphInf/proposer/label/uniform.hpp"
#include "GraphInf/proposer/nested_label/uniform.hpp"
#include "GraphInf/mcmc/community.hpp"
#include "GraphInf/mcmc/callbacks/collector.hpp"
#include "GraphInf/mcmc/callbacks/action.h"
#include "GraphInf/rng.h"
#include "../fixtures.hpp"

using namespace std;

namespace GraphInf{


class TestPartitionReconstructionMCMC: public::testing::Test{
    size_t numSteps=10;
public:
    StochasticBlockModelFamily randomGraph = StochasticBlockModelFamily(10, 10, 3);
    PartitionReconstructionMCMC mcmc = PartitionReconstructionMCMC(randomGraph);
    CheckConsistencyOnSweep callback;
    bool expectConsistencyError = false;
    void SetUp(){
        seed(1);
        randomGraph.sample();

        mcmc.insertCallBack("check_consistency", callback);
        mcmc.checkSafety();
    }
    void TearDown(){
        if (not expectConsistencyError)
            mcmc.checkConsistency();
    }
};

TEST_F(TestPartitionReconstructionMCMC, doMetropolisHastingsStep){
    mcmc.doMetropolisHastingsStep();
}

TEST_F(TestPartitionReconstructionMCMC, doMHSweep2){
    mcmc.doMHSweep(1000);


}

TEST_F(TestPartitionReconstructionMCMC, setLabels_noThrow){
    size_t N = randomGraph.getSize();
    size_t B = randomGraph.getLabelCount();
    std::vector<BlockIndex> newLabels(N);
    std::uniform_int_distribution<BlockIndex> dist(0, B-1);
    for (size_t v=0; v<N; ++v)
        newLabels[v] = dist(rng);
    mcmc.setLabels(newLabels);
    // EXPECT_EQ(mcmc.getLabels(), newLabels);
    EXPECT_NO_THROW(mcmc.checkConsistency());
}


class TestNestedPartitionReconstructionMCMC: public::testing::Test{
    size_t numSteps=10;
public:
    NestedStochasticBlockModelFamily randomGraph = NestedStochasticBlockModelFamily(10, 10);
    NestedPartitionReconstructionMCMC mcmc = NestedPartitionReconstructionMCMC(randomGraph);
    CheckConsistencyOnStep callback;
    bool expectConsistencyError = false;
    void SetUp(){
        // seed(2);
        seedWithTime();
        randomGraph.sample();

        mcmc.insertCallBack("check_consistency", callback);
        mcmc.checkSafety();
    }
    void TearDown(){
        if (not expectConsistencyError)
            mcmc.checkConsistency();
    }
};

TEST_F(TestNestedPartitionReconstructionMCMC, doMetropolisHastingsStep){
    mcmc.doMetropolisHastingsStep();
}

TEST_F(TestNestedPartitionReconstructionMCMC, doMHSweep){
    mcmc.doMHSweep(1000);
}

}
