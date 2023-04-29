#include "GraphInf/utility/functions.h"
#include "GraphInf/graph/proposer/edge/util.h"

namespace GraphInf
{

    // std::map<std::pair<BlockIndex,BlockIndex>, MultiGraph> getSubGraphOfLabelPair(const RandomGraph& randomGraph){
    //
    //     const MultiGraph& graph = randomGraph.getGraph();
    //     const std::vector<BlockIndex>& blocks = randomGraph.getBlocks();
    //     size_t blockCount = randomGraph.getBlockCount();
    //
    //     /* collecting edges */
    //     std::map<LabelPair, std::list<BaseGraph::Edge> > subEdges;
    //     for (auto vertex: graph){
    //         size_t r = blocks[vertex];
    //         for (auto neighbor: graph.getNeighboursOfIdx(vertex)){
    //             size_t s = blocks[neighbor.vertexIndex];
    //             BaseGraph::Edge edge = getOrderedEdge({vertex, neighbor.vertexIndex});
    //             LabelPair labelPair = {r, s};
    //             if (subEdges.count(labelPair) == 0)
    //                 subEdges.insert({labelPair, {edge}});
    //             else
    //                 subEdges[labelPair].push_back(edge);
    //         }
    //     }
    //
    //     /* constructing sub graphs */
    //     std::map<std::pair<BlockIndex,BlockIndex>, MultiGraph> subGraphs;
    //     for (size_t r=0; r<blockCount; ++r){
    //         for (size_t s=r; s<blockCount; ++s){
    //             LabelPair labelPair = {r, s};
    //             subGraphs.insert({labelPair, MultiGraph(graph.getSize())});
    //             for (auto e : subEdges[labelPair]){
    //                 subGraphs[labelPair].setEdgeMultiplicityIdx(e, graph.getEdgeMultiplicityIdx(e));
    //             }
    //         }
    //     }
    //
    //     for (auto g : subGraphs){
    //         std::cout << "Sugraph (" << g.first.first << ", " << g.first.second << ") : N=" << g.second.getSize() << ", E=" << g.second.getTotalEdgeNumber() << std::endl;
    //     }
    //
    //     return subGraphs;
    // }

    void checkEdgeSamplerConsistencyWithGraph(const std::string className, const MultiGraph &graph, const EdgeSampler &edgeSampler)
    {
        for (const auto &edge : graph.edges())
        {
            if (not edgeSampler.contains(edge))
                throw ConsistencyError(
                    className + ": edgeSampler is inconsistent with graph, edge (" + std::to_string(edge.first) + ", " + std::to_string(edge.second) + ") is not in sampler.");
            size_t expected = graph.getEdgeMultiplicity(edge.first, edge.second);
            size_t actual = edgeSampler.getEdgeWeight(edge);
            if (expected != actual)
                throw ConsistencyError(
                    className + ": edgeSampler is inconsistent with graph, edge (" + std::to_string(edge.first) + ", " + std::to_string(edge.second) + ") with weights " + std::to_string(expected) + "!=" + std::to_string(actual) + ".");
        }
    }

    void checkVertexSamplerConsistencyWithGraph(const std::string className, const MultiGraph &graph, const VertexSampler &vertexSampler)
    {
        for (auto u : graph)
        {
            if (not vertexSampler.contains(u))
                throw ConsistencyError(
                    className + ": vertexSampler is inconsistent with graph, vertex " + std::to_string(u) + " is not in sampler.");
        }
    }

}
