#ifndef GRAPH_INF_TYPES_H
#define GRAPH_INF_TYPES_H

#include <map>
#include <random>
#include <vector>

#include "BaseGraph/undirected_multigraph.hpp"
#include "BaseGraph/types.h"
#include "GraphInf/utility/maps.hpp"

namespace GraphInf
{

    template <typename T>
    using Matrix = std::vector<std::vector<T>>;

    typedef std::mt19937_64 RNG;

    typedef BaseGraph::UndirectedMultigraph MultiGraph;
    typedef BaseGraph::UndirectedMultigraph LabelGraph;
    typedef int BlockIndex;
    typedef std::vector<size_t> DegreeSequence;
    typedef std::vector<BlockIndex> BlockSequence;
    typedef Matrix<size_t> EdgeMatrix;
    typedef CounterMap<std::pair<BlockIndex, size_t>> VertexLabeledDegreeCountsMap;
    typedef CounterMap<size_t> DegreeCountsMap;
    typedef int Level;

} // namespace GraphInf

#endif
