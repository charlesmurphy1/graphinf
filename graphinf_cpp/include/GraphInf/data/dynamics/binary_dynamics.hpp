#ifndef GRAPH_INF_BINARY_DYNAMICS_H
#define GRAPH_INF_BINARY_DYNAMICS_H


#include <vector>
#include <map>

#include "GraphInf/random_graph/random_graph.hpp"
#include "GraphInf/data/dynamics/dynamics.hpp"
#include "GraphInf/types.h"


namespace GraphInf{

template <typename GraphPriorType=RandomGraph>
class BinaryDynamics: public Dynamics<GraphPriorType>{
private:
    double m_autoActivationProb;
    double m_autoDeactivationProb;
public:
    using BaseClass = Dynamics<GraphPriorType>;
    explicit BinaryDynamics(
            size_t numSteps,
            double autoActivationProb=0.0,
            double autoDeactivationProb=0.0):
        BaseClass(2, numSteps),
        m_autoActivationProb(autoActivationProb),
        m_autoDeactivationProb(autoDeactivationProb){ }
    explicit BinaryDynamics(
            GraphPriorType& randomGraph,
            size_t numSteps,
            double autoActivationProb=0.0,
            double autoDeactivationProb=0.0):
        BaseClass(randomGraph, 2, numSteps),
        m_autoActivationProb(autoActivationProb),
        m_autoDeactivationProb(autoDeactivationProb){ }
    const double getTransitionProb(
        const VertexState& prevVertexState, const VertexState& nextVertexState, const VertexNeighborhoodState& neighborhoodState
    ) const override;

    const State getRandomState(int initialActive) const;
    const State getRandomState() const override { return getRandomState(-1); }
    virtual const double getActivationProb(const VertexNeighborhoodState& neighborState) const = 0;
    virtual const double getDeactivationProb(const VertexNeighborhoodState& neighborState) const = 0;

    void setAutoActivationProb(double autoActivationProb){ m_autoActivationProb = autoActivationProb; }
    void setAutoDeactivationProb(double autoDeactivationProb){ m_autoDeactivationProb = autoDeactivationProb; }
    const double getAutoActivationProb() const { return m_autoActivationProb; }
    const double getAutoDeactivationProb() const { return m_autoDeactivationProb; }

};

template <typename GraphPriorType>
const State BinaryDynamics<GraphPriorType>::getRandomState(int initialActive) const {
    size_t N = BaseClass::m_graphPriorPtr->getSize();
    State randomState(N);
    if (initialActive < 0 or initialActive > N)
        return Dynamics<GraphPriorType>::getRandomState();

    auto indices = sampleUniformlySequenceWithoutReplacement(N, initialActive);
    for (auto i: indices)
        randomState[i] = 1;
    return randomState;
};

template <typename GraphPriorType>
const double BinaryDynamics<GraphPriorType>::getTransitionProb(
    const VertexState& prevVertexState, const VertexState& nextVertexState, const VertexNeighborhoodState& neighborhoodState) const {
    double p;
    double transProb;
    if ( prevVertexState == 0 ) {
        p = (1 - m_autoActivationProb) * getActivationProb(neighborhoodState) + m_autoActivationProb;
        if (nextVertexState == 0) transProb = 1 - p;
        else transProb = p;
    }
    else {
        p = (1 - m_autoDeactivationProb) * getDeactivationProb(neighborhoodState) + m_autoDeactivationProb;
        if (nextVertexState == 1) transProb = 1 - p;
        else transProb = p;
    }

    return clipProb(transProb);
};

} // namespace GraphInf

#endif
