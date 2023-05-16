#ifndef GRAPH_INF_DATAMODEL_H
#define GRAPH_INF_DATAMODEL_H

#include "GraphInf/rv.hpp"
#include "GraphInf/graph/random_graph.hpp"
#include "GraphInf/data/proposer.h"
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
        MultiParamProposer m_paramProposer;

    public:
        DataModel(RandomGraph &prior) : m_uniform(0, 1) { setGraphPrior(prior); }
        virtual ~DataModel() {}

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
            computeConsistentState();
        }
        const size_t getSize() const { return m_graphPriorPtr->getSize(); }
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
        virtual const double getLogLikelihoodRatioFromParaMove(const ParamMove &move)
        {
            ParamMove reverseMove = {move.key, -move.value};
            double logLikelihoodRatio = -getLogLikelihood();
            applyParamMove(move);
            logLikelihoodRatio += getLogLikelihood();
            applyParamMove(reverseMove);
            return logLikelihoodRatio;
        }
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
        virtual const MCMCSummary metropolisGraphStep(const double betaPrior = 1, const double betaLikelihood = 1);
        virtual const MCMCSummary metropolisParamStep()
        {
            if (m_paramProposer.size() == 0)
                return {"ParamMove()", 0, true};

            auto move = m_paramProposer.proposeMove();

            if (not isValidParamMove(move))
                return {move.display(), -INFINITY, false};

            double likelihoodRatio = getLogLikelihoodRatioFromParaMove(move);
            double proposalRatio = m_paramProposer.logProposalRatio(move);
            double acceptProb = exp(likelihoodRatio + proposalRatio);
            if (m_uniform(rng) < acceptProb)
            {
                applyParamMove(move);
                return {move.display(), acceptProb, true};
            }
            return {move.display(), acceptProb, false};
        }
        virtual const MCMCSummary metropolisPriorStep()
        {
            return m_graphPriorPtr->metropolisStep();
        }

        const int gibbsSweep(size_t numSteps, const double sampleGraphProb = 1., const double samplePriorProb = 0., const double sampleParamProb = 0., const double betaPrior = 1, const double betaLikelihood = 1);
        const int metropolisSweep(size_t numSteps, const double sampleGraphRate = 1., const double samplePriorRate = 0., const double sampleParamRate = 0., const double betaPrior = 1, const double betaLikelihood = 1);

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
        virtual void applyParamMove(const ParamMove &move) {}
        virtual bool isValidParamMove(const ParamMove &move) const { return true; }

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
