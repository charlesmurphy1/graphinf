#include "gtest/gtest.h"
#include <list>
#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <iostream>

#include "GraphInf/graph/random_graph.hpp"
#include "GraphInf/generators.h"
#include "GraphInf/types.h"
#include "GraphInf/graph/delta.h"
#include "BaseGraph/types.h"
#include "../fixtures.hpp"

using namespace std;
using namespace GraphInf;

static const int NUM_VERTICES = 7;
static const GraphMove GRAPH_MOVE = {{{0, 3}}, {{0, 5}}};
static MultiGraph GRAPH = getUndirectedHouseMultiGraph();

class TestRandomGraphBaseClass : public ::testing::Test
{
public:
    DummyRandomGraph randomGraph = {NUM_VERTICES};
    MultiGraph graph = GRAPH;

    void SetUp()
    {
        randomGraph.setState(graph);
    }
};

// void enumerateAllGraphs() const;

TEST_F(TestRandomGraphBaseClass, getState_returnHouseMultigraphGraph)
{
    MultiGraph graph = randomGraph.getState();
    EXPECT_EQ(graph.getSize(), NUM_VERTICES);
    EXPECT_EQ(graph.getTotalEdgeNumber(), 10);
}

TEST_F(TestRandomGraphBaseClass, getState_differentFromHouseGraph)
{
    MultiGraph graph = randomGraph.getState();
    graph.addEdge(0, 0);
    randomGraph.setState(graph);
    EXPECT_EQ(graph.getSize(), NUM_VERTICES);
    EXPECT_EQ(graph.getTotalEdgeNumber(), 11);
}

TEST_F(TestRandomGraphBaseClass, getSize_returnCorrectGraphSize)
{
    MultiGraph graph = randomGraph.getState();
    EXPECT_EQ(graph.getSize(), randomGraph.getSize());
    EXPECT_EQ(NUM_VERTICES, randomGraph.getSize());
}

TEST_F(TestRandomGraphBaseClass, getLogJoint_return0)
{
    EXPECT_EQ(randomGraph.getLogJoint(), 0);
}

TEST_F(TestRandomGraphBaseClass, applyMove_forSomeGraphMove)
{
    randomGraph.applyGraphMove(GRAPH_MOVE);
    auto removed = GRAPH_MOVE.removedEdges[0], added = GRAPH_MOVE.addedEdges[0];
    EXPECT_FALSE(randomGraph.getState().hasEdge(removed.first, removed.second));
    EXPECT_TRUE(randomGraph.getState().hasEdge(added.first, added.second));
}

TEST_F(TestRandomGraphBaseClass, applyMove_forNonExistingEdgeRemoved_throwRuntimeError)
{
    GraphMove move = {{{0, 0}}, {}}; // non-existing edge, throw logic_error
    EXPECT_THROW(randomGraph.applyGraphMove(move), std::runtime_error);
}

TEST(TestDeltaGraph, forSomeGraph_constructDeltaGraphAndSample_returnOriginalgraph)
{

    const auto g = generateErdosRenyi(10, 10);
    DeltaGraph model = {g};
    model.sample();
}