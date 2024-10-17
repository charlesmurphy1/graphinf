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
        double logProposalRatio = m_graphPriorPtr->getLogProposalRatioFromGraphMove(move);
        if (logLikelihoodRatio == -INFINITY or logPriorRatio == -INFINITY)
            return -INFINITY;
        double logJointRatio = logLikelihoodRatio + logPriorRatio;
        return logProposalRatio + logJointRatio;
    }

    const StepResult<GraphMove> DataModel::metropolisGraphStep(const double betaPrior, const double betaLikelihood, bool debug)
    {
        const auto move = m_graphPriorPtr->proposeGraphMove();
        if (m_graphPriorPtr->isTrivialGraphMove(move))
            return {};
        auto logLikelihoodBefore = 0.0, logPriorBefore = 0.0;
        if (debug)
        {
            logLikelihoodBefore = getLogLikelihood();
            logPriorBefore = getLogPrior();
        }

        // Log likelihood ratio
        double logLikelihoodRatio = 0;
        if (betaLikelihood > 0)
            logLikelihoodRatio = betaLikelihood * getLogLikelihoodRatioFromGraphMove(move);

        // Log prior ratio
        double logPriorRatio = 0;
        if (betaPrior > 0)
            logPriorRatio = betaPrior * getLogPriorRatioFromGraphMove(move);

        // Log proposal ratio
        double logProposalRatio = m_graphPriorPtr->getLogProposalRatioFromGraphMove(move);

        // Acceptance probability
        double acceptProb = exp(logLikelihoodRatio + logPriorRatio + logProposalRatio);

        // Metropolis-Hastings step
        bool accepted = false;
        if (m_uniform(rng) < acceptProb)
        {
            accepted = true;
            applyGraphMove(move);
        }
        if (debug)
        {
            checkConsistency();
            auto logLikelihoodAfter = getLogLikelihood();
            auto logPriorAfter = getLogPrior();
            if (abs(logLikelihoodAfter - logLikelihoodBefore - logLikelihoodRatio) > 1e-6 && accepted == 1)
            {
                std::stringstream ss;
                ss << "DataModel: log likelihood mismatch with move " << move.display() << ": expected_ratio=" << logLikelihoodAfter - logLikelihoodBefore << ", actual_ratio=" << logLikelihoodRatio;
                throw std::runtime_error(ss.str());
            }
            if (abs(logPriorAfter - logPriorBefore - logPriorRatio) > 1e-6 && accepted == 1)
            {
                std::stringstream ss;
                ss << "DataModel: log prior mismatch with move " << move.display() << ": expected_ratio=" << logPriorAfter - logPriorBefore << ", actual_ratio=" << logPriorRatio;
                throw std::runtime_error(ss.str());
            }
        }
        return {move, logLikelihoodRatio + logPriorRatio, accepted};
    }

    const MCMCSummary DataModel::metropolisGraphSweep(size_t numSteps, const double betaPrior, const double betaLikelihood, int debugFrequency)
    {
        MCMCSummary summary = {};
        for (size_t i = 0; i < numSteps; i++)
        {
            summary.update(metropolisGraphStep(betaPrior, betaLikelihood, (debugFrequency > 0 && (i + 1) % debugFrequency == 0)));
            summary.update(m_graphPriorPtr->metropolisStep(betaPrior, betaLikelihood));
        }
        return summary;
    }
    const MCMCSummary DataModel::metropolisParamSweep(size_t numSteps, double betaPrior, double betaLikelihood)
    {
        MCMCSummary summary = {};
        for (size_t i = 0; i < numSteps; i++)
        {
            summary.update(metropolisParamStep(betaLikelihood, betaPrior));
        }
        return summary;
    }
}