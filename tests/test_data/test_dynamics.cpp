#include "gtest/gtest.h"
#include <list>
#include <algorithm>
#include <iostream>

#include "GraphInf/data/dynamics/dynamics.h"
#include "GraphInf/types.h"
#include "GraphInf/utility/functions.h"
#include "BaseGraph/types.h"
#include "../fixtures.hpp"

using namespace std;
using namespace GraphInf;

namespace GraphInf
{

    class DynamicsParametrizedTest : public ::testing::TestWithParam<size_t>
    {
    public:
        const int NUM_VERTICES = 7;
        const int LENGTH = 20;
        const int PAST_LENGTH = GetParam();
        const int NUM_STATES = 3;
        const vector<int> ALL_VERTEX_STATES = {0, 1, 2};
        const State STATE = {0, 0, 0, 1, 1, 2, 0};
        const NeighborsState NEIGHBORS_STATE = {
            {3, 1, 0}, {1, 2, 0}, {4, 1, 0}, {3, 1, 1}, {1, 1, 0}, {0, 1, 0}, {0, 0, 0}};
        const GraphMove GRAPH_MOVE = {{{0, 2}}, {{0, 5}}};
        MultiGraph GRAPH = getUndirectedHouseMultiGraph();
        DummyRandomGraph randomGraph = DummyRandomGraph(NUM_VERTICES);
        DummyDynamics dynamics = DummyDynamics(randomGraph, NUM_STATES, LENGTH);
        MultiGraph graph = GRAPH;
        State state = STATE;
        bool expectConsistencyError = true;
        void SetUp()
        {
            dynamics.acceptSelfLoops(false);
            dynamics.setGraph(graph);
            dynamics.setCurrentState(state);
            // dynamics.checkSafety();
        }
        void TearDown()
        {
            if (not expectConsistencyError)
                dynamics.checkConsistency();
        }
    };

    TEST_P(DynamicsParametrizedTest, getState_returnState)
    {
        auto x = dynamics.getState();
        EXPECT_EQ(x, STATE);
    }

    TEST_P(DynamicsParametrizedTest, getNeighborsState_returnState)
    {

        auto n = dynamics.getNeighborsState();
        EXPECT_EQ(n, dynamics.computeNeighborsState(STATE));
    }

    TEST_P(DynamicsParametrizedTest, getGraph_returnGraph)
    {
        auto g = dynamics.getGraph();
        EXPECT_EQ(g, GRAPH);
    }

    TEST_P(DynamicsParametrizedTest, getSize_returnGraphSize)
    {
        auto n = dynamics.getSize();
        EXPECT_EQ(n, STATE.size());
        EXPECT_EQ(n, NUM_VERTICES);
    }

    TEST_P(DynamicsParametrizedTest, getNumStates_returnNumStates)
    {
        auto s = dynamics.getNumStates();
        EXPECT_EQ(s, NUM_STATES);
    }

    TEST_P(DynamicsParametrizedTest, getRandomState_returnRandomState)
    {
        auto x = dynamics.getRandomState();
        EXPECT_EQ(x.size(), NUM_VERTICES);
        for (auto xx : x)
        {
            EXPECT_TRUE(xx <= NUM_STATES - 1);
            EXPECT_TRUE(xx >= 0);
        }
    }

    TEST_P(DynamicsParametrizedTest, computeNeighborsState_forSomeState_returnThatNeighborState)
    {
        auto neighborsState = dynamics.computeNeighborsState(STATE);
        EXPECT_EQ(neighborsState.size(), NUM_VERTICES);
        int i = 0, j = 0;
        for (auto vertexNeighborState : neighborsState)
        {
            j = 0;
            EXPECT_EQ(vertexNeighborState.size(), NUM_STATES);
            for (auto l : vertexNeighborState)
            {
                EXPECT_EQ(l, NEIGHBORS_STATE[i][j]);
                j++;
            }
            i++;
        }
    }

    TEST_P(DynamicsParametrizedTest, getTransitionProbs_forEachVertexState_returnTransitionProbVector)
    {
        vector<double> probs;
        for (auto in_state : ALL_VERTEX_STATES)
        {
            probs = dynamics.getTransitionProbs(in_state, NEIGHBORS_STATE[3]);
            EXPECT_EQ(probs.size(), NUM_STATES);
            for (auto out_state : ALL_VERTEX_STATES)
            {
                EXPECT_EQ(probs[out_state], dynamics.getTransitionProb(in_state, out_state, NEIGHBORS_STATE[3]));
            }
        }
    }

    TEST_P(DynamicsParametrizedTest, sampleState_forSomeNumSteps_returnNothing)
    {
        dynamics.sampleState();
        auto n = dynamics.getNeighborsState();
        EXPECT_EQ(n, dynamics.computeNeighborsState(dynamics.getState()));
    }

    TEST_P(DynamicsParametrizedTest, getPastStates_returnPastStates)
    {
        dynamics.sampleState();
        StateSequence past_states = dynamics.getPastStates();
        EXPECT_EQ(past_states.size(), NUM_VERTICES);
        for (auto state : past_states)
        {
            EXPECT_EQ(state.size(), LENGTH);
        }
    }

    TEST_P(DynamicsParametrizedTest, getFutureStates_returnFutureStates)
    {
        dynamics.sampleState();
        StateSequence future_states = dynamics.getFutureStates();
        EXPECT_EQ(future_states.size(), NUM_VERTICES);
        for (auto state : future_states)
        {
            EXPECT_EQ(state.size(), LENGTH);
        }
    }

    TEST_P(DynamicsParametrizedTest, getLogJointRatio_forSomeGraphMove_returnLogJointRatio)
    {
        dynamics.sampleState();
        double ratio = dynamics.getLogJointRatioFromGraphMove(GRAPH_MOVE);
        EXPECT_EQ(ratio, 0.);
    }

    TEST_P(DynamicsParametrizedTest, applyMove_forSomeGraphMove_expectChangesInTheGraph)
    {
        dynamics.sampleState();
        auto past = dynamics.getPastStates();
        dynamics.applyGraphMove(GRAPH_MOVE);
        auto expected = dynamics.getNeighborsPastStates();
        auto graph = dynamics.getGraph();
        EXPECT_EQ(graph.getEdgeMultiplicity(0, 2), 2);
        EXPECT_EQ(graph.getEdgeMultiplicity(0, 5), 1);
        for (size_t t = 0; t < dynamics.getLength(); ++t)
        {
            for (const auto vertex : graph)
            {
                std::vector<size_t> actual(dynamics.getNumStates(), 0);
                for (auto neighbor : graph.getOutNeighbours(vertex))
                {
                    size_t edgeMult = graph.getEdgeMultiplicity(vertex, neighbor);
                    if (neighbor == vertex)
                    {
                        if (dynamics.acceptSelfLoops())
                            edgeMult *= 2;
                        else
                            continue;
                    }

                    actual[past[neighbor][t]] += edgeMult;
                }
                for (size_t s = 0; s < dynamics.getNumStates(); ++s)
                {
                    EXPECT_EQ(expected[vertex][t][s], actual[s]);
                }
            }
        }
        EXPECT_EQ(dynamics.getNeighborsState(), dynamics.computeNeighborsState(dynamics.getState()));
    }

    TEST_P(DynamicsParametrizedTest, updateNeighborsStateFromEdgeMove_fromAddedEdge_expectCorrectionInNeighborState)
    {
        dynamics.sampleState();
        BaseGraph::Edge edge = GRAPH_MOVE.addedEdges[0];
        map<BaseGraph::VertexIndex, VertexNeighborhoodStateSequence> actualBefore, actualAfter;

        dynamics.updateNeighborsStateFromEdgeMove(edge, 1, actualBefore, actualAfter);

        auto expectedBefore = dynamics.getNeighborsPastStates();
        dynamics.applyGraphMove({{}, {edge}});
        auto expectedAfter = dynamics.getNeighborsPastStates();

        for (auto actual : actualBefore)
            for (size_t t = 0; t < dynamics.getLength(); ++t)
                for (size_t s = 0; s < dynamics.getNumStates(); ++s)
                    EXPECT_EQ(expectedBefore[actual.first][t][s], actual.second[t][s]);

        for (auto actual : actualAfter)
            for (size_t t = 0; t < dynamics.getLength(); ++t)
                for (size_t s = 0; s < dynamics.getNumStates(); ++s)
                    EXPECT_EQ(expectedAfter[actual.first][t][s], actual.second[t][s]);
    }

    TEST_P(DynamicsParametrizedTest, updateNeighborsStateFromEdgeMove_fromRemovedEdge_expectCorrectionInNeighborState)
    {
        dynamics.sampleState();
        BaseGraph::Edge edge = GRAPH_MOVE.removedEdges[0];
        map<BaseGraph::VertexIndex, VertexNeighborhoodStateSequence> actualBefore, actualAfter;

        dynamics.updateNeighborsStateFromEdgeMove(edge, -1, actualBefore, actualAfter);

        auto expectedBefore = dynamics.getNeighborsPastStates();
        dynamics.applyGraphMove({{edge}, {}});
        auto expectedAfter = dynamics.getNeighborsPastStates();

        for (auto actual : actualBefore)
            for (size_t t = 0; t < dynamics.getLength(); ++t)
                for (size_t s = 0; s < dynamics.getNumStates(); ++s)
                    EXPECT_EQ(expectedBefore[actual.first][t][s], actual.second[t][s]);

        for (auto actual : actualAfter)
            for (size_t t = 0; t < dynamics.getLength(); ++t)
                for (size_t s = 0; s < dynamics.getNumStates(); ++s)
                    EXPECT_EQ(expectedAfter[actual.first][t][s], actual.second[t][s]);
    }

    INSTANTIATE_TEST_SUITE_P(
        DynamicsBaseClassTests,
        DynamicsParametrizedTest,
        ::testing::Values(0, 10));

} /* GraphInf */
