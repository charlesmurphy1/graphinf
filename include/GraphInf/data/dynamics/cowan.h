#ifndef GRAPH_INF_WILSON_COWAN_H
#define GRAPH_INF_WILSON_COWAN_H

#include "GraphInf/data/dynamics/binary_dynamics.h"
#include "GraphInf/data/util.h"

namespace GraphInf
{

    class CowanDynamics : public BinaryDynamics
    {
    private:
        double m_a;
        double m_nu;
        double m_mu;
        double m_eta;
        double MIN_A = 0, MAX_A = 10;
        double MIN_MU = 0, MAX_MU = 10;
        double MIN_NU = 0, MAX_NU = 10;
        double MIN_ETA = 0, MAX_ETA = 1;

    public:
        CowanDynamics(
            RandomGraph &graphPrior,
            size_t numSteps,
            double nu = 1,
            double a = 1,
            double mu = 1,
            double eta = 0.5,
            double autoActivationProb = 1e-6,
            double autoDeactivationProb = 0,
            double aStddev = 0.1,
            double nuStddev = 0.1,
            double muStddev = 0.1,
            double etaStddev = 0.1,
            double activationStddev = 0.1) : BinaryDynamics(graphPrior,
                                                            numSteps,
                                                            autoActivationProb,
                                                            autoDeactivationProb,
                                                            activationStddev,
                                                            0.0),
                                             m_a(a),
                                             m_nu(nu),
                                             m_mu(mu),
                                             m_eta(eta)
        {
            m_paramProposer.insertGaussianProposer("nu", 1.0, 0.0, nuStddev);
            m_paramProposer.insertGaussianProposer("mu", 1.0, 0.0, muStddev);
            m_paramProposer.insertGaussianProposer("eta", 1.0, 0.0, muStddev);
        }

        const double getActivationProb(const VertexNeighborhoodState &vertexNeighborState) const override
        {
            return sigmoid(m_a * (getNu() * vertexNeighborState[1] - m_mu));
        }
        const double getDeactivationProb(const VertexNeighborhoodState &vertexNeighborState) const override
        {
            return m_eta;
        }
        const double getA() const { return m_a; }
        void setA(double a) { m_a = a; }
        const double getNu() const { return m_nu; }
        void setNu(double nu) { m_nu = nu; }
        const double getMu() const { return m_mu; }
        void setMu(double mu) { m_mu = mu; }
        const double getEta() const { return m_eta; }
        void setEta(double eta) { m_eta = eta; }

        void applyParamMove(const ParamMove &move) override
        {
            if (move.key == "a")
                m_a += move.value;
            else if (move.key == "nu")
                m_nu += move.value;
            else if (move.key == "mu")
                m_mu += move.value;
            else if (move.key == "eta")
                m_eta += move.value;

            BinaryDynamics::applyParamMove(move);
        }
        bool isValidParamMove(const ParamMove &move) const override
        {
            if (move.key == "a")
                return MIN_A <= m_a + move.value && m_a + move.value <= MAX_A;
            else if (move.key == "nu")
                return MIN_NU <= m_nu + move.value && m_nu + move.value <= MAX_NU;
            else if (move.key == "mu")
                return MIN_MU <= m_mu + move.value && m_mu + move.value <= MAX_MU;
            else if (move.key == "eta")
                return MIN_ETA <= m_eta + move.value && m_eta + move.value <= MAX_ETA;
            return BinaryDynamics::isValidParamMove(move);
        }
    };

} // namespace GraphInf

#endif
