#ifndef GRAPH_INF_DELTAGRAPH_H
#define GRAPH_INF_DELTAGRAPH_H

#include "GraphInf/graph/random_graph.hpp"
#include "GraphInf/graph/likelihood/delta.h"
#include "GraphInf/graph/util.h"

namespace GraphInf
{

    class DeltaGraph : public RandomGraph
    {
    private:
        DeltaGraphLikelihood m_likelihoodModel;
        void setUpLikelihood() override
        {
            m_likelihoodModel.m_statePtr = &m_state;
        }

    public:
        DeltaGraph(const MultiGraph graph) : RandomGraph(graph.getSize(), graph.getTotalEdgeNumber(), m_likelihoodModel)
        {
            m_state = graph;
            setUpLikelihood();
        }
    };

}

#endif