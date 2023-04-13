#include "GraphInf/data/data_model.h"

namespace GraphInf
{

    const double DataModel::getLogAcceptanceProbFromGraphMove(const GraphMove move, double betaPrior = 1, double betaLikelihood = 1) const
    {
        double logLikelihoodRatio, logPriorRatio;
        if (betaLikelihood == 0)
            logLikelihoodRatio = 0;
        else
            logLikelihoodRatio = betaLikelihood * getLogLikelihoodRatioFromGraphMove(move);
        if (betaPrior == 0)
            logPriorRatio = 0;
        else
            logPriorRatio = betaPrior * getLogPriorRatioFromGraphMove(move);
        double logProposalRatio = m_graphPriorPtr->getEdgeProposer().getLogProposalProbRatio(move);
        if (logLikelihoodRatio == -INFINITY or logPriorRatio == -INFINITY)
            return -INFINITY;
        double logJointRatio = logLikelihoodRatio + logPriorRatio;
        return logProposalRatio + logJointRatio;
    }

    const MCMCSummary DataModel::metropolisStep(const double samplePriorProb = 0.5, const double betaPrior = 1, const double betaLikelihood = 1)
    {
        if (m_uniform(rng) < samplePriorProb)
            return m_graphPriorPtr->metropolisStep();

        const auto move = m_graphPriorPtr->proposeGraphMove();
        if (m_graphPriorPtr->getEdgeProposer().isTrivialMove(move))
            return {"graph", 1.0, true};
        double acceptProb = exp(getLogAcceptanceProbFromGraphMove(move, betaPrior, betaLikelihood));
        bool isAccepted = false;
        if (m_uniform(rng) < acceptProb)
        {
            isAccepted = true;
            applyGraphMove(move);
        }
        return {"graph", acceptProb, isAccepted};
    }
    const int DataModel::mcmcSweep(size_t numSteps, const double samplePriorProb = 0.5, const double betaPrior = 1, const double betaLikelihood = 1)
    {
        int numSuccesses = 0;
        for (size_t i = 0; i < numSteps; i++)
        {
            auto summary = metropolisStep(samplePriorProb, betaPrior, betaLikelihood);
            if (summary.isAccepted)
                numSuccesses += 1;
        }
        return numSuccesses;
    }
}