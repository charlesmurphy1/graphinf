#ifndef GRAPH_INF_SIS_DYNAMICS_H
#define GRAPH_INF_SIS_DYNAMICS_H

#include <vector>
#include <map>
#include <cmath>

#include "GraphInf/data/dynamics/binary_dynamics.h"

namespace GraphInf
{

    class SISDynamics : public BinaryDynamics
    {

    public:
        explicit SISDynamics(
            RandomGraph &graphPrior,
            size_t numSteps,
            double infectionProb = 0.5,
            double recoveryProb = 0.5,
            double autoActivationProb = 1e-6,
            double autoDeactivationProb = 0,
            double infectionStddev = 0.1,
            double activationStddev = 0.1,
            double deactivationStddev = 0.1) : BinaryDynamics(graphPrior,
                                                              numSteps,
                                                              autoActivationProb,
                                                              autoDeactivationProb,
                                                              activationStddev,
                                                              deactivationStddev),
                                               m_infectionProb(infectionProb),
                                               m_recoveryProb(recoveryProb)
        {
            m_paramProposer.insertGaussianProposer("infection", 1.0, 0.0, infectionStddev);
        }

        const double getActivationProb(const VertexNeighborhoodState &vertexNeighborState) const override
        {
            return 1 - std::pow(1 - getInfectionProb(), vertexNeighborState[1]);
        }
        const double getDeactivationProb(const VertexNeighborhoodState &vertexNeighborState) const override
        {
            return m_recoveryProb;
        }

        const double getInfectionProb() const { return m_infectionProb; }
        void setInfectionProb(double infectionProb) { m_infectionProb = infectionProb; }
        const double getRecoveryProb() const { return m_recoveryProb; }
        void setRecoveryProb(double recoveryProb) { m_recoveryProb = recoveryProb; }
        void applyParamMove(const ParamMove &move) override
        {
            if (move.key == "infection")
                m_infectionProb += move.value;

            BinaryDynamics::applyParamMove(move);
        }
        bool isValidParamMove(const ParamMove &move) const override
        {
            if (move.key == "infection")
                return 0 <= m_infectionProb + move.value && m_infectionProb + move.value <= 1;
            return BinaryDynamics::isValidParamMove(move);
        }

    private:
        double m_infectionProb, m_recoveryProb;
        const double EPSILON = 1e-6;
    };

} // namespace GraphInf

#endif
