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

    public:
        CowanDynamics(
            RandomGraph &graphPrior,
            size_t numSteps,
            double nu = 1,
            double a = 1,
            double mu = 1,
            double eta = 0.5,
            double autoActivationProb = 1e-6,
            double autoDeactivationProb = 0) : BinaryDynamics(graphPrior,
                                                              numSteps,
                                                              autoActivationProb,
                                                              autoDeactivationProb),
                                               m_a(a),
                                               m_nu(nu),
                                               m_mu(mu),
                                               m_eta(eta) {}

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
    };

} // namespace GraphInf

#endif
