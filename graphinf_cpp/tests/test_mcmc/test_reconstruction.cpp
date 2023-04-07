#include "gtest/gtest.h"
#include <cmath>
#include <random>
#include <time.h>

#include "GraphInf/data/dynamics/sis.hpp"
#include "GraphInf/random_graph/proposer/edge/hinge_flip.h"
#include "GraphInf/random_graph/proposer/label/uniform.hpp"
#include "GraphInf/random_graph/proposer/nested_label/uniform.hpp"
#include "GraphInf/mcmc/reconstruction.hpp"
#include "GraphInf/rng.h"
#include "../fixtures.hpp"

using namespace std;

namespace GraphInf{

class TestGraphReconstructionMCMC: public::testing::Test{
    size_t numSteps=10;
public:
    ErdosRenyiModel randomGraph = ErdosRenyiModel(10, 10);
    DummySISDynamics dynamics = DummySISDynamics(randomGraph);
    GraphReconstructionMCMC<RandomGraph> mcmc = GraphReconstructionMCMC<RandomGraph>(dynamics);
    bool expectConsistencyError = false;
    void SetUp(){
        seed(1);
        dynamics.sample();
        mcmc.checkSafety();
    }
    void TearDown(){
        if (not expectConsistencyError)
            mcmc.checkConsistency();
    }
};

TEST_F(TestGraphReconstructionMCMC, doMetropolisHastingsStep){
    mcmc.doMetropolisHastingsStep();
}

TEST_F(TestGraphReconstructionMCMC, doMHSweep){
    mcmc.doMHSweep(10);
}

class TestVertexLabeledGraphReconstructionMCMC: public::testing::Test{
    size_t numSteps=10;
public:
    StochasticBlockModelFamily graphPrior = StochasticBlockModelFamily(10, 10, 3);
    DummyLabeledSISDynamics dynamics = DummyLabeledSISDynamics(graphPrior);
    VertexLabeledGraphReconstructionMCMC<BlockIndex> mcmc = VertexLabeledGraphReconstructionMCMC<BlockIndex>(dynamics);
    bool expectConsistencyError = false;
    void SetUp(){
        seedWithTime();
        dynamics.sample();
        mcmc.checkSafety();
    }
    void TearDown(){
        if (not expectConsistencyError)
            mcmc.checkConsistency();
    }
};

TEST_F(TestVertexLabeledGraphReconstructionMCMC, doMetropolisHastingsStep){
    mcmc.doMetropolisHastingsStep();
}

TEST_F(TestVertexLabeledGraphReconstructionMCMC, doMHSweep){
    mcmc.doMHSweep(10);
}


class TestNestedVertexLabeledGraphReconstructionMCMC: public::testing::Test{
    size_t numSteps=10;
public:
    NestedStochasticBlockModelFamily graphPrior = NestedStochasticBlockModelFamily(10, 10);
    DummyNestedSISDynamics dynamics = DummyNestedSISDynamics(graphPrior);
    NestedVertexLabeledGraphReconstructionMCMC<BlockIndex> mcmc = NestedVertexLabeledGraphReconstructionMCMC<BlockIndex>(dynamics);
    bool expectConsistencyError = false;
    void SetUp(){
        seedWithTime();
        dynamics.sample();
        mcmc.checkSafety();
    }
    void TearDown(){
        if (not expectConsistencyError)
            mcmc.checkConsistency();
    }
};

TEST_F(TestNestedVertexLabeledGraphReconstructionMCMC, doMetropolisHastingsStep){
    mcmc.doMetropolisHastingsStep();
}

TEST_F(TestNestedVertexLabeledGraphReconstructionMCMC, doMHSweep){
    mcmc.doMHSweep(10);
}


} // GraphInf