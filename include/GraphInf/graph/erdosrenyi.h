#ifndef GRAPH_INF_ERDOSRENYI_H
#define GRAPH_INF_ERDOSRENYI_H

#include <map>
#include <utility>
#include <vector>

#include "BaseGraph/types.h"
#include "GraphInf/graph/random_graph.hpp"
#include "GraphInf/graph/likelihood/erdosrenyi.h"
#include "GraphInf/graph/util.h"
#include "GraphInf/generators.h"

namespace GraphInf
{

    class ErdosRenyiModelBase : public RandomGraph
    {
    protected:
        ErdosRenyiLikelihood m_likelihoodModel;
        const double _getLogPrior() const override { return m_edgeCountPriorPtr->getLogJoint(); }
        const double _getLogPriorRatioFromGraphMove(const GraphMove &move) const override
        {
            return m_edgeCountPriorPtr->getLogJointRatioFromGraphMove(move);
        }
        void sampleOnlyPrior(bool canonical = true) override
        {
            if (canonical)
                m_edgeCountPriorPtr->sample();
        }
        void setUpLikelihood() override
        {
            m_likelihoodModel.m_statePtr = &m_state;
            m_likelihoodModel.m_graphSizePtr = &m_size;
            m_likelihoodModel.m_edgeCountPriorPtrPtr = &m_edgeCountPriorPtr;
            m_likelihoodModel.m_withSelfLoopsPtr = &m_withSelfLoops;
            m_likelihoodModel.m_withParallelEdgesPtr = &m_withParallelEdges;
        }
        ErdosRenyiModelBase(size_t size, double edgeCount, bool canonical = false, bool withSelfLoops = true, bool withParallelEdges = true) : RandomGraph(size, edgeCount, m_likelihoodModel, canonical, withSelfLoops, withParallelEdges)
        {
            setUpLikelihood();
        }

        void computeConsistentState() override
        {
            m_edgeCountPriorPtr->setState(m_state.getTotalEdgeNumber());
        }

    public:
        // const EdgeCountPrior &getEdgeCountPrior() { return *m_edgeCountPriorPtr; }
        // void setEdgeCountPrior(EdgeCountPrior &edgeCountPrior) { m_edgeCountPriorPtr = &edgeCountPrior; }
        // void fromGraph(const MultiGraph &graph) override
        // {
        //     RandomGraph::fromGraph(graph);
        //     m_edgeCountPriorPtr->setState(graph.getTotalEdgeNumber());
        // }

        // const bool isCompatible(const MultiGraph& graph) const override{
        //     return RandomGraph::isCompatible(graph) and graph.getTotalEdgeNumber() == getEdgeCount();
        // }
        void computationFinished() const override
        {
            m_isProcessed = false;
            m_edgeCountPriorPtr->computationFinished();
        }
        void checkSelfSafety() const override
        {
            RandomGraph::checkSelfSafety();
            if (not m_edgeCountPriorPtr)
                throw SafetyError("ErdosRenyiModel", "m_edgeCountPriorPtr");
            m_edgeCountPriorPtr->checkSafety();
        }
    };

    class ErdosRenyiModel : public ErdosRenyiModelBase
    {
        // std::unique_ptr<EdgeCountPrior> m_edgeCountPriorUPtr = nullptr;

    public:
        ErdosRenyiModel(
            size_t size,
            double edgeCount,
            bool canonical = false,
            bool withSelfLoops = true,
            bool withParallelEdges = true) : ErdosRenyiModelBase(size, edgeCount, canonical, withSelfLoops, withParallelEdges)
        {
            // m_edgeCountPriorUPtr = std::unique_ptr<EdgeCountPrior>(makeEdgeCountPrior(edgeCount, canonical));
            // setEdgeCountPrior(*m_edgeCountPriorUPtr);
            // if (canonical)
            //     setGraphMoveType("canonical");
            // else
            //     setGraphMoveType("microcanonical");
            checkSafety();
            sample();
        }
    };

} // end GraphInf
#endif
