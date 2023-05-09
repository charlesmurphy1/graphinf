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
        std::unique_ptr<EdgeProposer> m_edgeProposerUPtr = nullptr;
        void setUpLikelihood() override
        {
            m_likelihoodModel.m_statePtr = &m_state;
        }

    public:
        DeltaGraph(const MultiGraph graph) : RandomGraph(graph.getSize(), m_likelihoodModel)
        {
            m_state = graph;
            m_edgeProposerUPtr = std::unique_ptr<EdgeProposer>(makeEdgeProposer());
            setEdgeProposer(*m_edgeProposerUPtr);
            setUpLikelihood();
        }
        const size_t getEdgeCount() const override { return m_state.getTotalEdgeNumber(); }
    };

}

#endif