#include "gtest/gtest.h"

#include "../fixtures.hpp"
#include "GraphInf/types.h"
#include "GraphInf/generators.h"

namespace GraphInf
{

    MultiGraph constructLabelGraph()
    {
        MultiGraph labelGraph(3);

        labelGraph.addMultiedge(0, 0, 2);
        labelGraph.addMultiedge(0, 1, 1);
        labelGraph.addMultiedge(0, 2, 3);
        labelGraph.addMultiedge(1, 2, 2);
        labelGraph.addMultiedge(2, 2, 3);
        return labelGraph;
    }

    static const MultiGraph LABEL_GRAPH = constructLabelGraph();
    static const GraphInf::BlockSequence VERTEX_BLOCKS = {
        0, 0, 0, 0,
        1, 1, 1,
        2, 2, 2, 2, 2};
    static const GraphInf::DegreeSequence DEGREES = {
        4, 2, 2, 0,
        2, 0, 1,
        0, 4, 4, 2, 1};

    static LabelGraph getLabelGraph(const GraphInf::MultiGraph &graph, const std::vector<BlockIndex> &vertexBlocks)
    {
        size_t blockNumber = 1;
        for (auto block : vertexBlocks)
            if (block >= blockNumber)
                blockNumber = block + 1;

        LabelGraph labelGraph(blockNumber);
        size_t r, s, mult;

        for (const auto &edge : graph.edges())
        {
            r = vertexBlocks[edge.first];
            s = vertexBlocks[edge.second];
            mult = graph.getEdgeMultiplicity(edge.first, edge.second);
            labelGraph.addMultiedge(r, s, mult);
        }
        return labelGraph;
    }
    static const size_t numberOfGeneratedGraphs = 10;

    static const GraphInf::DegreeSequence convertDegrees(const std::vector<size_t> &basegraphDegrees)
    {
        GraphInf::DegreeSequence degrees;
        for (auto d : basegraphDegrees)
            degrees.push_back(d);
        return degrees;
    }

    TEST(TESTSampleRandomNeighbor, sampleRandomNeighbor_forMultipleSample_sampleAccordingToMultiplicity)
    {
        seedWithTime();
        size_t numSamples = 1000;
        MultiGraph graph = getUndirectedHouseMultiGraph();
        graph.addMultiedge(1, 2, 2);
        graph.addMultiedge(1, 3, 1);
        graph.addMultiedge(1, 1, 2);

        CounterMap<BaseGraph::VertexIndex> counter;
        for (size_t i = 0; i < numSamples; ++i)
        {
            counter.increment(sampleRandomNeighbor(graph, 1, true));
        }

        for (const auto &c : counter)
        {
            size_t m = ((c.first == 1) ? 2 : 1) * graph.getEdgeMultiplicity(1, c.first);
            EXPECT_EQ(round(((double)c.second * 10) / numSamples), m);
        }
    }

    TEST(TestDCSBMGenerator, generateDCSBM_givenLabelGraphAndDegrees_generatedGraphsRespectLabelGraphAndDegrees)
    {
        for (size_t i = 0; i < numberOfGeneratedGraphs; i++)
        {
            auto randomGraph = GraphInf::generateDCSBM(VERTEX_BLOCKS, LABEL_GRAPH, DEGREES);
            EXPECT_EQ(LABEL_GRAPH, getLabelGraph(randomGraph, VERTEX_BLOCKS));
            EXPECT_EQ(DEGREES, convertDegrees(randomGraph.getDegrees()));
        }
    }

    TEST(TestSBMGenerator, generateStubLabeledSBM_givenLabelGraph_generatedGraphsRespectLabelGraph)
    {
        for (size_t i = 0; i < numberOfGeneratedGraphs; i++)
        {
            auto randomGraph = GraphInf::generateStubLabeledSBM(VERTEX_BLOCKS, LABEL_GRAPH);
            EXPECT_EQ(LABEL_GRAPH, getLabelGraph(randomGraph, VERTEX_BLOCKS));
        }
    }

    TEST(TestSBMGenerator, generateMultiGraphSBM_givenLabelGraph_generatedGraphsRespectLabelGraph)
    {
        for (size_t i = 0; i < numberOfGeneratedGraphs; i++)
        {
            auto randomGraph = GraphInf::generateMultiGraphSBM(VERTEX_BLOCKS, LABEL_GRAPH);
            EXPECT_EQ(LABEL_GRAPH, getLabelGraph(randomGraph, VERTEX_BLOCKS));
        }
    }

    TEST(TestSBMGenerator, generateSBM_ggivenLabelGraph_returnGraphsWithCorrectLabelGraph)
    {

        auto graph = GraphInf::generateSBM(VERTEX_BLOCKS, LABEL_GRAPH);
        EXPECT_EQ(LABEL_GRAPH, getLabelGraph(graph, VERTEX_BLOCKS));
    }

}
