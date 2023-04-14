#include "GraphInf/data/dynamics/binary_dynamics.h"

namespace GraphInf
{

    const State BinaryDynamics::getRandomState(int initialActive) const
    {
        size_t N = m_graphPriorPtr->getSize();
        State randomState(N);
        if (initialActive < 0 or initialActive > N)
            return Dynamics::getRandomState();

        auto indices = sampleUniformlySequenceWithoutReplacement(N, initialActive);
        for (auto i : indices)
            randomState[i] = 1;
        return randomState;
    };

    const double BinaryDynamics::getTransitionProb(
        const VertexState &prevVertexState, const VertexState &nextVertexState, const VertexNeighborhoodState &neighborhoodState) const
    {
        double p;
        double transProb;
        if (prevVertexState == 0)
        {
            p = (1 - m_autoActivationProb) * getActivationProb(neighborhoodState) + m_autoActivationProb;
            if (nextVertexState == 0)
                transProb = 1 - p;
            else
                transProb = p;
        }
        else
        {
            p = (1 - m_autoDeactivationProb) * getDeactivationProb(neighborhoodState) + m_autoDeactivationProb;
            if (nextVertexState == 1)
                transProb = 1 - p;
            else
                transProb = p;
        }

        return clipProb(transProb);
    };
} // namespace GraphInf
