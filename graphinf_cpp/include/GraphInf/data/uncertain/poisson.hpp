#ifndef GRAPH_INF_POISSON_UNCERTAIN_HPP
#define GRAPH_INF_POISSON_UNCERTAIN_HPP


#include <random>
#include <stdexcept>
#include "GraphInf/rng.h"
#include "GraphInf/types.h"
#include "uncertain.hpp"

namespace GraphInf{

template<typename GraphPriorType=RandomGraph>
class UncertainPoissonModel: public UncertainGraphModel<GraphPriorType>{
    MultiGraph m_state;
    const MultiGraph& m_graph;
    double m_averageNoEdge, m_averageEdge;


protected:
    void applyGraphMoveToSelf(const GraphMove& move) {}

public:
    UncertainPoissonModel(const MultiGraph& graph, double averageNoEdge, double averageEdge):
        m_averageNoEdge(averageNoEdge), m_averageEdge(averageEdge), m_graph(graph) {}

    const MultiGraph& getState() const { return m_state; }
    void setState(const MultiGraph& observations) {
        if (observations.getSize() != m_graph.getSize())
            throw std::logic_error("Cannot assign observations with size "
                    +std::to_string(observations.getSize())+" to PoissonModel with graph of size "
                    +std::to_string(m_graph.getSize())+".\n");
        m_state = observations;
    }

    virtual void sampleState() {
        auto n = m_graph.getSize();
        for (size_t i=0; i<n; i++) {
            for (size_t j=i+1; j<n; j++) {
                size_t multiplicity = m_graph.getEdgeMultiplicityIdx(i, j);
                double average = getAverage(m_graph.getEdgeMultiplicityIdx(i, j));

                m_state.setEdgeMultiplicityIdx(i, j,
                    std::poisson_distribution<size_t>(average)(rng));
            }
        }
    }
    const double getLogLikelihood() const {
        double logLikelihood = 0;

        auto n = m_graph.getSize();
        for (size_t i=0; i<n; i++) {
            for (size_t j=i+1; j<n; j++) {
                const auto& observation = m_state.getEdgeMultiplicityIdx(i, j);
                double average = getAverage(m_graph.getEdgeMultiplicityIdx(i, j));

                logLikelihood += observation*log(average)-average-lgamma(observation+1);
            }
        }
        return logLikelihood;
    }
    const double getLogLikelihoodRatioFromGraphMove(const GraphMove& move) const {
        double logLikelihoodRatio = 0;

        for (auto removedEdge: move.removedEdges)
            logLikelihoodRatio += computeLogLikelihoodRatioOfPair(removedEdge.first, removedEdge.second, 0);
        for (auto addedEdge: move.addedEdges)
            logLikelihoodRatio += computeLogLikelihoodRatioOfPair(addedEdge.first, addedEdge.second, 1);
        return logLikelihoodRatio;
    }

    double computeLogLikelihoodRatioOfPair(size_t i, size_t j, bool addingEdge) const {
        const auto& observation = m_state.getEdgeMultiplicityIdx(i, j);
        size_t multiplicity = m_graph.getEdgeMultiplicityIdx(i, j);

        double currentAverage = getAverage(multiplicity);
        double newAverage = getAverage(multiplicity+2*addingEdge-1);

        return observation*(log(newAverage)-log(currentAverage))
                - newAverage + currentAverage;
    }

    double getAverage(size_t multiplicity) const {
        if (multiplicity == 0)
            return m_averageNoEdge;
        return multiplicity*m_averageEdge;
    }
};

};

#endif
