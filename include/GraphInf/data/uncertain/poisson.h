#ifndef GRAPH_INF_POISSON_UNCERTAIN_H
#define GRAPH_INF_POISSON_UNCERTAIN_H

#include <random>
#include <stdexcept>
#include "GraphInf/rng.h"
#include "GraphInf/types.h"
#include "uncertain.h"

namespace GraphInf
{

    class PoissonUncertainGraph : public UncertainGraph
    {
        double m_averageNoEdge, m_averageEdge;

    protected:
        void applyGraphMoveToSelf(const GraphMove &move) override {}

    public:
        PoissonUncertainGraph(RandomGraph &prior, double averageEdge, double averageNoEdge = 0) : UncertainGraph(prior), m_averageNoEdge(averageNoEdge), m_averageEdge(averageEdge) {}

        virtual void sampleState() override
        {
            const auto &graph = m_graphPriorPtr->getState();
            auto n = graph.getSize();
            m_state = MultiGraph(n);
            for (size_t i = 0; i < n; i++)
            {
                for (size_t j = i + 1; j < n; j++)
                {
                    size_t multiplicity = graph.getEdgeMultiplicity(i, j);
                    double average = getAverage(graph.getEdgeMultiplicity(i, j));

                    m_state.setEdgeMultiplicity(i, j,
                                                std::poisson_distribution<size_t>(average)(rng));
                }
            }
        }
        const double getLogLikelihood() const override
        {
            double logLikelihood = 0;

            const auto &graph = m_graphPriorPtr->getState();
            auto n = graph.getSize();
            for (size_t i = 0; i < n; i++)
            {
                for (size_t j = i + 1; j < n; j++)
                {
                    const auto &observation = m_state.getEdgeMultiplicity(i, j);
                    double average = getAverage(graph.getEdgeMultiplicity(i, j));

                    logLikelihood += observation * log(average) - average - lgamma(observation + 1);
                }
            }
            return logLikelihood;
        }
        const double getLogLikelihoodRatioFromGraphMove(const GraphMove &move) const override
        {
            double logLikelihoodRatio = 0;

            for (auto removedEdge : move.removedEdges)
                logLikelihoodRatio += computeLogLikelihoodRatioOfPair(removedEdge.first, removedEdge.second, 0);
            for (auto addedEdge : move.addedEdges)
                logLikelihoodRatio += computeLogLikelihoodRatioOfPair(addedEdge.first, addedEdge.second, 1);
            return logLikelihoodRatio;
        }

        double computeLogLikelihoodRatioOfPair(size_t i, size_t j, bool addingEdge) const
        {
            const auto &observation = m_state.getEdgeMultiplicity(i, j);
            const auto &graph = m_graphPriorPtr->getState();
            auto multiplicity = graph.getEdgeMultiplicity(i, j);

            double currentAverage = getAverage(multiplicity);
            double newAverage = getAverage(multiplicity + 2 * addingEdge - 1);

            return observation * (log(newAverage) - log(currentAverage)) - newAverage + currentAverage;
        }

        double getAverage(size_t multiplicity) const
        {
            if (multiplicity == 0)
                return m_averageNoEdge;
            return multiplicity * m_averageEdge;
        }
    };

};

#endif
