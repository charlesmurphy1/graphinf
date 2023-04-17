#ifndef FASTMIDYNET_DEGREE_DYNAMICS_H
#define FASTMIDYNET_DEGREE_DYNAMICS_H

#include "GraphInf/data/dynamics/binary_dynamics.h"

namespace GraphInf
{

    class DegreeDynamics : public BinaryDynamics
    {
        double m_C;

    public:
        DegreeDynamics(RandomGraph &graphPrior, size_t numSteps, double C) : BinaryDynamics(graphPrior, numSteps, 0, 0), m_C(C) {}

        const double getActivationProb(const VertexNeighborhoodState &vertexNeighborState) const override
        {
            return (vertexNeighborState[0] + vertexNeighborState[1]) / m_C;
        }
        const double getDeactivationProb(const VertexNeighborhoodState &vertexNeighborState) const override
        {
            return 1 - getActivationProb(vertexNeighborState);
        }
        const double getC() const { return m_C; }
        void setC(double C) { m_C = C; }
    };

} // namespace GraphInf

#endif
