#ifndef GRAPH_INF_VOTER_MODEL_H
#define GRAPH_INF_VOTER_MODEL_H

#include "GraphInf/data/dynamics/binary_dynamics.h"
#include "GraphInf/data/util.h"
#include "GraphInf/data/proposer.h"
#include "random"

namespace GraphInf
{

    class VoterDynamics : public BinaryDynamics
    {
        MultiParamProposer m_proposer;
        double m_random_flip_prob;

    public:
        VoterDynamics(
            RandomGraph &graphPrior,
            size_t numSteps,
            double random_flip_prob = 0.1,
            double autoActivationProb = 0,
            double autoDeactivationProb = 0,
            double activationStddev = 0.1,
            double deactivationStddev = 0.1,
            double randomFlipStddev = 0.1) : BinaryDynamics(graphPrior,
                                                            numSteps,
                                                            autoActivationProb,
                                                            autoDeactivationProb,
                                                            activationStddev,
                                                            deactivationStddev),
                                             m_random_flip_prob(random_flip_prob)
        {
            m_paramProposer.insertGaussianProposer("random_flip_prob", 1, 0.0, randomFlipStddev);
        }

        const double getRandomFlipProb() const { return m_random_flip_prob; }
        void setRandomFlipProb(double random_flip_prob) { m_random_flip_prob = random_flip_prob; }

        const double getActivationProb(const VertexNeighborhoodState &vertexNeighborState) const override
        {
            double p = (double)vertexNeighborState[1] / (double)(vertexNeighborState[0] + vertexNeighborState[1]);
            return m_random_flip_prob * 0.5 + (1 - m_random_flip_prob) * p;
        }
        const double getDeactivationProb(const VertexNeighborhoodState &vertexNeighborState) const override
        {
            double p = (double)vertexNeighborState[0] / (double)(vertexNeighborState[0] + vertexNeighborState[1]);
            return m_random_flip_prob * 0.5 + (1 - m_random_flip_prob) * p;
        }

        void applyParamMove(const ParamMove &move) override
        {
            if (move.key == "random_flip_prob")
                m_random_flip_prob += move.value;

            BinaryDynamics::applyParamMove(move);
        }

        bool isValidParamMove(const ParamMove &move) const override
        {
            if (move.key == "random_flip_prob")
                return 0 <= m_random_flip_prob + move.value && m_random_flip_prob + move.value <= 1;
            return BinaryDynamics::isValidParamMove(move);
        }
    };

} // namespace GraphInf

#endif
