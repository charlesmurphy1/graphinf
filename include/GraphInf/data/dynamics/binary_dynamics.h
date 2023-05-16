#ifndef GRAPH_INF_BINARY_DYNAMICS_H
#define GRAPH_INF_BINARY_DYNAMICS_H

#include <vector>
#include <map>

#include "GraphInf/graph/random_graph.hpp"
#include "GraphInf/data/dynamics/dynamics.h"
#include "GraphInf/types.h"

namespace GraphInf
{

    class BinaryDynamics : public Dynamics
    {
    private:
        double m_autoActivationProb;
        double m_autoDeactivationProb;

    public:
        explicit BinaryDynamics(
            RandomGraph &randomGraph,
            size_t numSteps,
            double autoActivationProb = 0.0,
            double autoDeactivationProb = 0.0,
            double activationStddev = 0.1,
            double deactivationStddev = 0.1) : Dynamics(randomGraph, 2, numSteps),
                                               m_autoActivationProb(autoActivationProb),
                                               m_autoDeactivationProb(autoDeactivationProb)
        {
            m_paramProposer.insertGaussianProposer("activation", 1, 0., activationStddev);
            m_paramProposer.insertGaussianProposer("deactivation", 1, 0., deactivationStddev);
        }
        const double getTransitionProb(
            const VertexState &prevVertexState, const VertexState &nextVertexState, const VertexNeighborhoodState &neighborhoodState) const override;

        const State getRandomState(int initialActive) const;
        const State getRandomState() const override { return getRandomState(-1); }
        virtual const double getActivationProb(const VertexNeighborhoodState &neighborState) const = 0;
        virtual const double getDeactivationProb(const VertexNeighborhoodState &neighborState) const = 0;

        void setAutoActivationProb(double autoActivationProb) { m_autoActivationProb = autoActivationProb; }
        void setAutoDeactivationProb(double autoDeactivationProb) { m_autoDeactivationProb = autoDeactivationProb; }
        const double getAutoActivationProb() const { return m_autoActivationProb; }
        const double getAutoDeactivationProb() const { return m_autoDeactivationProb; }
        void applyParamMove(const ParamMove &move) override
        {
            if (move.key == "activation")
                m_autoActivationProb += move.value;

            if (move.key == "deactivation")
                m_autoDeactivationProb += move.value;
            Dynamics::applyParamMove(move);
        }
        virtual bool isValidParamMove(const ParamMove &move) const override
        {
            if (move.key == "activation")
                return 0 <= m_autoActivationProb + move.value && m_autoActivationProb + move.value <= 1;
            if (move.key == "deactivation")
                return 0 <= m_autoDeactivationProb + move.value && m_autoDeactivationProb + move.value <= 1;
            return Dynamics::isValidParamMove(move);
        }
    };

} // namespace GraphInf

#endif
