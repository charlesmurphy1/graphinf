#ifndef GRAPH_INF_EDGE_PROPOSER_UTIL_H
#define GRAPH_INF_EDGE_PROPOSER_UTIL_H

#include <map>
#include "GraphInf/graph/random_graph.hpp"
#include "GraphInf/graph/proposer/sampler/edge_sampler.h"
#include "GraphInf/graph/proposer/sampler/vertex_sampler.h"

namespace GraphInf
{

    // using LabelPair = std::pair<BlockIndex, BlockIndex>;
    // std::map<std::pair<BlockIndex,BlockIndex>, MultiGraph> getSubGraphOfLabelPair(const RandomGraph& randomGraph);

    void checkEdgeSamplerConsistencyWithGraph(const std::string, const MultiGraph &, const EdgeSampler &);
    void checkVertexSamplerConsistencyWithGraph(const std::string, const MultiGraph &, const VertexSampler &);

}

#endif
