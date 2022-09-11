#ifndef GRAPH_INF_WILSON_COWAN_H
#define GRAPH_INF_WILSON_COWAN_H


#include "GraphInf/data/dynamics/binary_dynamics.hpp"
#include "GraphInf/data/util.h"


namespace GraphInf{

template<typename GraphPriorType=RandomGraph>
class CowanDynamics: public BinaryDynamics<GraphPriorType> {
private:
    double m_a;
    double m_nu;
    double m_mu;
    double m_eta;

public:
    using BaseClass = BinaryDynamics<GraphPriorType>;
    CowanDynamics(
            size_t numSteps,
            double nu=1,
            double a=1,
            double mu=1,
            double eta=0.5,
            double autoActivationProb=1e-6,
            double autoDeactivationProb=0):
        BaseClass(
            numSteps,
            autoActivationProb,
            autoDeactivationProb),
        m_a(a),
        m_nu(nu),
        m_mu(mu),
        m_eta(eta) {}
    CowanDynamics(
            GraphPriorType& graphPrior,
            size_t numSteps,
            double nu=1,
            double a=1,
            double mu=1,
            double eta=0.5,
            double autoActivationProb=1e-6,
            double autoDeactivationProb=0):
        BaseClass(
            graphPrior,
            numSteps,
            autoActivationProb,
            autoDeactivationProb),
        m_a(a),
        m_nu(nu),
        m_mu(mu),
        m_eta(eta) {}

    const double getActivationProb(const VertexNeighborhoodState& vertexNeighborState) const override {
        return sigmoid(m_a * ( getNu() * vertexNeighborState[1] - m_mu));
    }
    const double getDeactivationProb(const VertexNeighborhoodState& vertexNeighborState) const override{
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
