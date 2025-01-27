#ifndef GRAPH_INF_CONFIGURATION_H
#define GRAPH_INF_CONFIGURATION_H

#include <map>
#include <utility>
#include <vector>

#include "BaseGraph/types.h"
#include "GraphInf/graph/likelihood/configuration.h"
#include "GraphInf/graph/random_graph.hpp"
#include "GraphInf/graph/util.h"
#include "GraphInf/generators.h"

namespace GraphInf
{

    class ConfigurationModelBase : public RandomGraph
    {
    protected:
        ConfigurationModelLikelihood m_likelihoodModel;
        DegreePrior *m_degreePriorPtr;

        void _applyGraphMove(const GraphMove &move) override
        {
            m_degreePriorPtr->applyGraphMove(move);
            RandomGraph::_applyGraphMove(move);
        }
        const double _getLogPrior() const override { return m_degreePriorPtr->getLogJoint(); }
        const double _getLogPriorRatioFromGraphMove(const GraphMove &move) const override
        {
            return m_degreePriorPtr->getLogJointRatioFromGraphMove(move);
        }
        void sampleOnlyPrior(bool canonical = true) override
        {
            if (canonical)
                m_degreePriorPtr->sample();
            else
                m_degreePriorPtr->sampleState();
        }
        void setUpLikelihood() override
        {
            m_likelihoodModel.m_statePtr = &m_state;
            m_likelihoodModel.m_degreePriorPtrPtr = &m_degreePriorPtr;
        }

        ConfigurationModelBase(size_t size, double edgeCount, bool canonical = false) : RandomGraph(size, edgeCount, m_likelihoodModel, canonical, true, true) { setUpLikelihood(); }
        ConfigurationModelBase(size_t size, double edgeCount, DegreePrior &degreePrior, bool canonical = false) : RandomGraph(size, edgeCount, m_likelihoodModel, canonical, true, true), m_degreePriorPtr(&degreePrior)
        {
            setUpLikelihood();
            m_degreePriorPtr->isRoot(false);
            m_degreePriorPtr->setSize(m_size);
        }
        void computeConsistentState() override
        {
            m_degreePriorPtr->setGraph(m_state);
        }

    public:
        DegreePrior &getDegreePriorRef() const { return *m_degreePriorPtr; }
        const DegreePrior &getDegreePrior() const { return *m_degreePriorPtr; }
        void setDegreePrior(DegreePrior &prior)
        {
            m_degreePriorPtr = &prior;
            m_degreePriorPtr->isRoot(false);
            m_degreePriorPtr->setEdgeCountPrior(*m_edgeCountPriorPtr);
        }
        const size_t getDegree(BaseGraph::VertexIndex vertex) const { return m_degreePriorPtr->getDegree(vertex); }
        const std::vector<size_t> &getDegrees() const { return m_degreePriorPtr->getState(); }

        void computationFinished() const override
        {
            m_isProcessed = false;
            m_degreePriorPtr->computationFinished();
        }

        void checkSelfConsistency() const override
        {
            RandomGraph::checkSelfConsistency();
            checkGraphConsistencyWithDegreeSequence(
                "ConfigurationModelBase", "m_state", m_state, "m_degreePriorPtr", getDegrees());
        }

        void checkSelfSafety() const override
        {
            RandomGraph::checkSelfSafety();
            if (not m_degreePriorPtr)
                throw SafetyError("ConfigurationModelBase", "m_degreePriorPtr");
            m_degreePriorPtr->checkSafety();
        }
    };

    class ConfigurationModel : public ConfigurationModelBase
    {
    private:
        DegreeDeltaPrior m_degreeDeltaPrior;

    public:
        ConfigurationModel(std::vector<size_t> degrees) : ConfigurationModelBase(degrees.size(), 0, false), m_degreeDeltaPrior(degrees)
        {
            size_t edgeCount = 0;
            for (size_t i = 0; i < degrees.size(); i++)
            {
                edgeCount += degrees[i];
            }
            edgeCount /= 2;
            m_edgeCountPriorPtr->setState(edgeCount);
            setDegreePrior(m_degreeDeltaPrior);
            checkSafety();
            sample();
            setGraphMoveType("double_edge_swap");
        }
    };

    class ConfigurationModelFamily : public ConfigurationModelBase
    {
        std::unique_ptr<DegreePrior> m_degreePriorUPtr = nullptr;

    public:
        ConfigurationModelFamily(
            size_t size,
            double edgeCount,
            bool canonical = false,
            bool useDegreeHyperPrior = true) : ConfigurationModelBase(size, edgeCount, canonical)
        {
            m_degreePriorUPtr = std::unique_ptr<DegreePrior>(makeDegreePrior(size, *m_edgeCountPriorPtr, useDegreeHyperPrior));
            setDegreePrior(*m_degreePriorUPtr);
            checkSafety();
            sample();
        }
    };

} // end GraphInf
#endif
