#include "gtest/gtest.h"
#include <list>

#include "GraphInf/data/dynamics/glauber.hpp"
#include "GraphInf/random_graph/erdosrenyi.h"
#include "GraphInf/random_graph/proposer/edge/hinge_flip.h"
#include "../fixtures.hpp"

namespace GraphInf{


class TestGlauberDynamics: public::testing::Test{
public:
    const double COUPLING_CONSTANT = 0.0001;
    const std::list<std::vector<VertexState>> NEIGHBOR_STATES = {{1, 3}, {2, 2}, {3, 1}};
    const size_t NUM_STEPS=20;
    double avgk = 5;
    ErdosRenyiModel randomGraph = ErdosRenyiModel(100, 250);
    GlauberDynamics<RandomGraph> dynamics = GraphInf::GlauberDynamics<RandomGraph>(
        randomGraph, NUM_STEPS, COUPLING_CONSTANT, 0, 0, false, true, -1
    );

    void SetUp(){
        dynamics.acceptSelfLoops(false);
    }
};


TEST_F(TestGlauberDynamics, getActivationProb_forEachStateTransition_returnCorrectProbability) {
    for (auto neighborState: NEIGHBOR_STATES){
        EXPECT_EQ(
            sigmoid( 2 * COUPLING_CONSTANT/avgk * ((int)neighborState[1]-(int)neighborState[0]) ),
            dynamics.getActivationProb(neighborState)
        );
    }
}

TEST_F(TestGlauberDynamics, getDeactivationProb_forEachStateTransition_returnCorrectProbability) {
    for (auto neighborState: NEIGHBOR_STATES){
        EXPECT_EQ(sigmoid(
            2*COUPLING_CONSTANT/avgk*((int)neighborState[0]-(int)neighborState[1])),
            dynamics.getDeactivationProb(neighborState)
        );
    }
}

TEST_F(TestGlauberDynamics, getLogLikelihoodRatioFromGraphMove_forAddedEdge_returnCorrectValue) {
    dynamics.sample();
    GraphMove move = {{}, {{0, 1}}};
    double actual = dynamics.getLogLikelihoodRatioFromGraphMove(move);
    double logLikelihoodBefore = dynamics.getLogLikelihood();
    dynamics.applyGraphMove(move);
    double logLikelihoodAfter = dynamics.getLogLikelihood();
    EXPECT_NEAR(actual, logLikelihoodAfter - logLikelihoodBefore, 1e-6);
}

TEST_F(TestGlauberDynamics, getLogLikelihoodRatioFromGraphMove_forRemovedEdge_returnCorrectValue) {
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

TEST_F(TestGlauberDynamics, afterSample_getCorrectNeighborState){
    dynamics.sample();
    dynamics.checkConsistency();
    // std::vector<int> s(dynamics.getNumSteps(), 0);
    // std::cout << "s = [";
    // for (auto t=0; t<dynamics.getNumSteps(); ++t){
    //     int s = 0;
    //     for (auto i=0; i<dynamics.getSize(); ++i){
    //         s += dynamics.getPastStates()[i][t];
    //     }
    //     std::cout << " " << s << " ";
    // }
    // std::cout << "]" << std::endl;;

}

TEST_F(TestGlauberDynamics, getLogLikelihood_returnCorrectLogLikelikehood){
    dynamics.sample();
    auto past = dynamics.getPastStates();
    auto future = dynamics.getFutureStates();
    auto neighborState = dynamics.getNeighborsPastStates();

    double expected = dynamics.getLogLikelihood();
    double actual = 0;
    for(size_t t=0; t<dynamics.getNumSteps(); ++t){
        for (auto vertex : dynamics.getGraph()){
            actual += log(dynamics.getTransitionProb(past[vertex][t], future[vertex][t], neighborState[vertex][t]));
        }
    }
    EXPECT_NEAR(expected, actual, 1E-6);
}

TEST_F(TestGlauberDynamics, getLogLikelihoodRatio_forSomeGraphMove_returnLogJointRatio){
    dynamics.sample();
    auto graphMove = randomGraph.proposeGraphMove();
    double ratio = dynamics.getLogLikelihoodRatioFromGraphMove(graphMove);
    double logLikelihoodBefore = dynamics.getLogLikelihood();
    dynamics.applyGraphMove(graphMove);
    double logLikelihoodAfter = dynamics.getLogLikelihood();

    EXPECT_NEAR(ratio, logLikelihoodAfter - logLikelihoodBefore, 1e-6);
}
//
// TEST(ExtraGlauberTest, testing_tests){
//     seedWithTime();
//     ErdosRenyiModel g = {5, 5};
//     GlauberDynamics<RandomGraph> m = {g, 100, 10};
//     m.sample();
//     std::vector<std::vector<size_t>> s;
//     for (size_t t=0; t<100; ++t){
//         s.push_back({});
//         for (auto v : m.getGraph()){
//             s[t].push_back(m.getPastStates()[v][t]);
//         }
//     }
//     displayMatrix(s, "s", true);
//
//     m.setState(s[0]);
//     auto p = m.getTransitionProbs(0);
//     std::cout<< displayVector(s[0], "s0") << "->" << displayVector(s[1], "s1") << std::endl;
//
//     for (auto v : m.getGraph()){
//         std::cout << "\t" << displayVector(m.getTransitionProbs(v), "p[" + std::to_string(v) + "]") << std::endl;
//     }
//
// }

TEST(ExtraGlauberTest, testing_tests){
    seedWithTime();
    ErdosRenyiModel g = {5, 5};
    GlauberDynamics<RandomGraph> m = {g, 100, 10};
    m.sample();

    MultiGraph gg(5);
    gg.addEdgeIdx(0, 1);
    gg.addEdgeIdx(0, 1);
    gg.addEdgeIdx(0, 3);
    gg.addEdgeIdx(0, 4);
    gg.addEdgeIdx(2, 3);

    std::vector<size_t> s0 = {0, 1, 0, 1, 1};

    m.setGraph(gg);
    m.sampleState(s0);
    std::vector<std::vector<size_t>> s;
    for (size_t t=0; t<100; ++t){
        s.push_back({});
        for (auto v : m.getGraph()){
            s[t].push_back(m.getPastStates()[v][t]);
        }
    }
    displayMatrix(s, "s", true);

    m.setState(s[0]);
    auto p = m.getTransitionProbs(0);
    std::cout<< displayVector(s[0], "s0") << "-> " << displayVector(s[1], "s1") << std::endl;
    for (auto v : m.getGraph()){
        std::cout << "\t" << displayVector(m.getTransitionProbs(v), "p[" + std::to_string(v) + "]") << std::endl;
    }

}

/*

[0 1 0 1 1] -> [0 0 0 0 0] : [1e-15, 0.999999999999999], n=[0, 4]
Undirected graph of size: 5
Neighbours of:
0: (1, 2)(3, 1)(4, 1)
1: (0, 2)
2: (3, 1)
3: (0, 1)(2, 1)
4: (0, 1)


*/

}
