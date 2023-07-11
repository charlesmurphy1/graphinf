#include "GraphInf/data/data_model.h"

namespace GraphInf
{

    const double DataModel::getLogAcceptanceProbFromGraphMove(const GraphMove &move, double betaPrior, double betaLikelihood) const
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

    const MCMCSummary DataModel::metropolisGraphStep(const double betaPrior, const double betaLikelihood)
    {
        const auto move = m_graphPriorPtr->proposeGraphMove();
        if (m_graphPriorPtr->getEdgeProposer().isTrivialMove(move))
            return {"GraphMove(trivial)", 1.0, true};
        double acceptProb = exp(getLogAcceptanceProbFromGraphMove(move, betaPrior, betaLikelihood));
        bool isAccepted = false;
        if (m_uniform(rng) < acceptProb)
        {
            isAccepted = true;
            applyGraphMove(move);
        }
        return {move.display(), acceptProb, isAccepted};
    }
    const int DataModel::gibbsSweep(size_t numSteps, const double betaPrior, const double betaLikelihood)
    {
        int numSuccesses = 0;
        for (size_t i = 0; i < numSteps; i++)
        {
            MCMCSummary summary;
            if (m_uniform(rng) < m_paramRate)
                summary = metropolisParamStep();
            if (m_uniform(rng) < m_graphPriorRate)
                summary = metropolisPriorStep();
            if (m_uniform(rng) < m_graphRate)
                summary = metropolisGraphStep(betaPrior, betaLikelihood);

            if (summary.isAccepted)
                numSuccesses += 1;
        }
        return numSuccesses;
    }
    const int DataModel::metropolisSweep(size_t numSteps, const double betaPrior, const double betaLikelihood)
    {
        int numSuccesses = 0;
        double z = m_graphRate + m_graphPriorRate + m_paramRate;
        auto dist = std::discrete_distribution<int>({m_graphRate / z, m_graphPriorRate / z, m_paramRate / z});
        for (size_t i = 0; i < numSteps; i++)
        {
            MCMCSummary summary;

            switch (dist(rng))
            {
            case 0:
                summary = metropolisGraphStep(betaPrior, betaLikelihood);
                break;
            case 1:
                summary = metropolisPriorStep();
                break;
            case 2:
                summary = metropolisParamStep();
                break;
            default:
                break;
            }

            if (summary.isAccepted)
                numSuccesses += 1;
        }
        return numSuccesses;
    }
    const int DataModel::metropolisGraphSweep(size_t numSteps, const double betaPrior, const double betaLikelihood)
    {
        int numSuccesses = 0;
        for (size_t i = 0; i < numSteps; i++)
        {
            MCMCSummary summary;
            summary = metropolisGraphStep(betaPrior, betaLikelihood);
            if (summary.isAccepted)
                numSuccesses += 1;
        }
        return numSuccesses;
    }
    const int DataModel::metropolisParamSweep(size_t numSteps, double betaPrior, double betaLikelihood)
    {
        int numSuccesses = 0;
        for (size_t i = 0; i < numSteps; i++)
        {
            MCMCSummary summary;
            summary = metropolisParamStep(betaLikelihood, betaPrior);
            if (summary.isAccepted)
                numSuccesses += 1;
        }
        return numSuccesses;
    }
}