#ifndef GRAPH_INF_DELTA_LIKELIHOOD_H
#define GRAPH_INF_DELTA_LIKELIHOOD_H

#include "GraphInf/graph/likelihood/likelihood.hpp"

namespace GraphInf
{
    class DeltaGraphLikelihood : public GraphLikelihoodModel
    {
    public:
        const double getLogLikelihood() const override { return 0; }
        const double getLogLikelihoodRatioFromGraphMove(const GraphMove &move) const override { return -INFINITY; }
        const MultiGraph sample() const override { return *m_statePtr; }
    };
}

#endif