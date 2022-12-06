#ifndef GRAPH_INF_ISING_MODEL_H
#define GRAPH_INF_ISING_MODEL_H


#include "GraphInf/data/dynamics/binary_dynamics.hpp"
#include "GraphInf/data/util.h"


namespace GraphInf{


template<typename GraphPriorType=RandomGraph>
class GlauberDynamics: public BinaryDynamics<GraphPriorType> {
    double m_coupling;

    public:
        using BaseClass = BinaryDynamics<GraphPriorType>;

        GlauberDynamics(
                size_t numSteps,
                double coupling=1,
                double autoActivationProb=0,
                double autoDeactivationProb=0):
            BaseClass(
                numSteps,
                autoActivationProb,
                autoDeactivationProb),
            m_coupling(coupling) {}
        GlauberDynamics(
                GraphPriorType& graphPrior,
                size_t numSteps,
                double coupling=1,
                double autoActivationProb=0,
                double autoDeactivationProb=0):
            BaseClass(
                graphPrior,
                numSteps,
                autoActivationProb,
                autoDeactivationProb),
            m_coupling(coupling) {}

        const double getActivationProb(const VertexNeighborhoodState& vertexNeighborState) const override {
            double p = sigmoid(2 * getCoupling() * ((int)vertexNeighborState[1] - (int)vertexNeighborState[0]));
            return p;
        }
        const double getDeactivationProb(const VertexNeighborhoodState& vertexNeighborState) const override {
            double p = sigmoid(2 * getCoupling() * ((int)vertexNeighborState[0] - (int)vertexNeighborState[1]));
            return p;
        }
        const double getCoupling() const { return m_coupling; }
        void setCoupling(double coupling) { m_coupling = coupling; }
};

} // namespace GraphInf

#endif
