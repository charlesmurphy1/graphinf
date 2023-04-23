#ifndef GRAPH_INF_DATAMODEL_H
#define GRAPH_INF_DATAMODEL_H

#include "GraphInf/rv.hpp"
#include "GraphInf/random_graph/random_graph.hpp"
#include "GraphInf/utility/mcmc.h"

namespace GraphInf
{

    class DataModel : public NestedRandomVariable
    {
    protected:
        RandomGraph *m_graphPriorPtr = nullptr;
        virtual void computeConsistentState(){};
        virtual void applyGraphMoveToSelf(const GraphMove &move) = 0;
        std::uniform_real_distribution<double> m_uniform;

    public:
        DataModel(RandomGraph &graphPrior) : m_uniform(0, 1) { setGraphPrior(graphPrior); }

        const MultiGraph &getGraph() const { return m_graphPriorPtr->getState(); }
        void setGraph(const MultiGraph &graph)
        {
            m_graphPriorPtr->setState(graph);
            computeConsistentState();
        }
        const RandomGraph &getGraphPrior() const { return *m_graphPriorPtr; }
        RandomGraph &getGraphPriorRef() const { return *m_graphPriorPtr; }
        void setGraphPrior(RandomGraph &prior)
        {
            m_graphPriorPtr = &prior;
            // m_graphPriorPtr->isRoot(false);
            computeConsistentState();
        }
        const size_t getSize() const { return m_graphPriorPtr->getSize(); }
        // virtual void sampleState() = 0;
        // void sample(){
        //     m_graphPriorPtr->sample();
        //     sampleState();
        //     computationFinished();
        //     #if DEBUG
        //     checkConsistency();
        //     #endif
        // }
        void samplePrior()
        {
            m_graphPriorPtr->sample();
            computeConsistentState();
            computationFinished();
#if DEBUG
            checkConsistency();
#endif
        }
        virtual const double getLogLikelihood() const = 0;
        const double getLogPrior() const
        {
            return NestedRandomVariable::processRecursiveFunction<double>([&]()
                                                                          { return m_graphPriorPtr->getLogJoint(); },
                                                                          0);
        }
        const double getLogJoint() const { return getLogPrior() + getLogLikelihood(); }
        virtual const double getLogLikelihoodRatioFromGraphMove(const GraphMove &move) const = 0;
        const double getLogPriorRatioFromGraphMove(const GraphMove &move) const
        {
            return NestedRandomVariable::processRecursiveConstFunction<double>([&]()
                                                                               { return m_graphPriorPtr->getLogJointRatioFromGraphMove(move); },
                                                                               0);
        }
        const double getLogJointRatioFromGraphMove(const GraphMove &move) const
        {
            return getLogPriorRatioFromGraphMove(move) + getLogLikelihoodRatioFromGraphMove(move);
        }
        const double getLogAcceptanceProbFromGraphMove(const GraphMove &move, double betaPrior = 1, double betaLikelihood = 1) const;
        virtual const MCMCSummary metropolisStep(const double samplePriorProb = 0.5, const double betaPrior = 1, const double betaLikelihood = 1);
        const int mcmcSweep(size_t numSteps, const double samplePriorProb = 0.5, const double betaPrior = 1, const double betaLikelihood = 1);

        void applyGraphMove(const GraphMove &move)
        {
            NestedRandomVariable::processRecursiveFunction([&]()
                                                           {
            applyGraphMoveToSelf(move);
            m_graphPriorPtr->applyGraphMove(move); });
#if DEBUG
            checkConsistency();
#endif
        }

        void computationFinished() const override
        {
            m_isProcessed = false;
            m_graphPriorPtr->computationFinished();
        }
        void checkSelfSafety() const override
        {
            if (m_graphPriorPtr == nullptr)
                throw SafetyError("DataModel", "m_graphPriorPtr");
            m_graphPriorPtr->checkSafety();
        }

        virtual bool isSafe() const override
        {
            return (m_graphPriorPtr != nullptr) and (m_graphPriorPtr->isSafe());
        }
    };

}
#endif
