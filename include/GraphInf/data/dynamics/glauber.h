#ifndef GRAPH_INF_ISING_MODEL_H
#define GRAPH_INF_ISING_MODEL_H

#include "GraphInf/data/dynamics/binary_dynamics.h"
#include "GraphInf/data/util.h"
#include "GraphInf/data/proposer.h"
#include "random"

namespace GraphInf
{

    class GlauberDynamics : public BinaryDynamics
    {
        double m_coupling;
        MultiParamProposer m_proposer;

    public:
        GlauberDynamics(
            RandomGraph &graphPrior,
            size_t numSteps,
            double coupling = 1,
            double autoActivationProb = 0,
            double autoDeactivationProb = 0,
            double couplingStddev = 0.1,
            double activationStddev = 0.1,
            double deactivationStddev = 0.1) : BinaryDynamics(graphPrior,
                                                              numSteps,
                                                              autoActivationProb,
                                                              autoDeactivationProb,
                                                              activationStddev,
                                                              deactivationStddev),
                                               m_coupling(coupling)
        {
            m_paramProposer.insertGaussianProposer("coupling", 1, 0.0, couplingStddev);
        }

        const double getActivationProb(const VertexNeighborhoodState &vertexNeighborState) const override
        {
            double p = sigmoid(2 * getCoupling() * ((int)vertexNeighborState[1] - (int)vertexNeighborState[0]));
            return p;
        }
        const double getDeactivationProb(const VertexNeighborhoodState &vertexNeighborState) const override
        {
            double p = sigmoid(2 * getCoupling() * ((int)vertexNeighborState[0] - (int)vertexNeighborState[1]));
            return p;
        }
        const double getCoupling() const { return m_coupling; }
        void setCoupling(double coupling) { m_coupling = coupling; }
        void applyParamMove(const ParamMove &move) override
        {
            if (move.key == "coupling")
                m_coupling += move.value;

            BinaryDynamics::applyParamMove(move);
        }
        bool isValidParamMove(const ParamMove &move) const override
        {
            if (move.key == "coupling")
                return 0 <= m_coupling + move.value;
            return BinaryDynamics::isValidParamMove(move);
        }
    };

} // namespace GraphInf

#endif
