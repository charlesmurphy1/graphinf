#ifndef GRAPH_INF_GRAPH_DISTANCE_H
#define GRAPH_INF_GRAPH_DISTANCE_H

#include "GraphInf/types.h"

namespace GraphInf{

class GraphDistance{
public:
    virtual double compute(const MultiGraph& graph1, const MultiGraph& graph2) const = 0;
};

class HammingDistance: public GraphDistance{

public:
    using GraphDistance::GraphDistance;
    double compute(const MultiGraph& graph1, const MultiGraph& graph2) const override;
};

}


#endif
