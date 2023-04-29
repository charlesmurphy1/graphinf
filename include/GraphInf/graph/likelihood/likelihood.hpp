#ifndef GRAPH_INF_RANDOMGRAPH_LIKELIHOOD_H
#define GRAPH_INF_RANDOMGRAPH_LIKELIHOOD_H

#include "GraphInf/rv.hpp"
#include "GraphInf/graph/proposer/movetypes.h"

namespace GraphInf
{

    class GraphLikelihoodModel : public NestedRandomVariable
    {
    protected:
        GraphLikelihoodModel() {}

    public:
        virtual ~GraphLikelihoodModel() {}
        virtual const double getLogLikelihood() const = 0;
        virtual const double getLogLikelihoodRatioFromGraphMove(const GraphMove &) const = 0;
        virtual const MultiGraph sample() const = 0;
        const MultiGraph *m_statePtr = nullptr;
    };

    template <typename Label>
    class VertexLabeledGraphLikelihoodModel : public GraphLikelihoodModel
    {
    public:
        virtual const double getLogLikelihoodRatioFromLabelMove(const LabelMove<Label> &) const = 0;
        using GraphLikelihoodModel::m_statePtr;
    };

}

#endif
