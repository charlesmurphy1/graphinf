#ifndef GRAPHINF_COLLECTOR_HPP
#define GRAPHINF_COLLECTOR_HPP

#include <vector>
#include <fstream>

#include "callback.hpp"
#include "GraphInf/types.h"
#include "BaseGraph/fileio.hpp"
#include "GraphInf/mcmc/mcmc.h"
#include "GraphInf/utility/functions.h"

namespace GraphInf
{

    template <typename MCMCType>
    class Collector : public CallBack<MCMCType>
    {
    public:
        virtual void collect() = 0;
    };

    template <typename MCMCType>
    class SweepCollector : public Collector<MCMCType>
    {
    public:
        void onSweepEnd() override { this->collect(); }
    };

    template <typename MCMCType>
    class StepCollector : public Collector<MCMCType>
    {
    public:
        void onStepEnd() override { this->collect(); }
    };

    template <typename GraphMCMC>
    class CollectGraphOnSweep : public SweepCollector<GraphMCMC>
    {
    private:
        std::vector<MultiGraph> m_collectedGraphs;

    public:
        using BaseClass = SweepCollector<GraphMCMC>;
        void collect() override { m_collectedGraphs.push_back(BaseClass::m_mcmcPtr->getGraph()); }
        void clear() override { m_collectedGraphs.clear(); }
        const std::vector<MultiGraph> &getData() const { return m_collectedGraphs; }
    };

    template <typename GraphMCMC>
    class CollectEdgeMultiplicityOnSweep : public SweepCollector<GraphMCMC>
    {
    private:
        CounterMap<BaseGraph::Edge> m_observedEdges;
        CounterMap<std::pair<BaseGraph::Edge, size_t>> m_observedEdgesCount;
        CounterMap<BaseGraph::Edge> m_observedEdgesMaxCount;
        size_t m_totalCount;

    public:
        using BaseClass = SweepCollector<GraphMCMC>;
        void collect() override;
        void clear() override
        {
            m_observedEdges.clear();
            m_observedEdgesCount.clear();
            m_observedEdgesMaxCount.clear();
        }
        const double getMarginalEntropy();
        const MultiGraph &getCurrentGraph() { return BaseClass::m_mcmcPtr->getGraph(); }
        const double getLogPosteriorEstimate(const MultiGraph &);
        const double getLogPosteriorEstimate() { return getLogPosteriorEstimate(BaseClass::m_mcmcPtr->getGraph()); }
        size_t getTotalCount() const { return m_totalCount; }
        size_t getEdgeObservationCount(BaseGraph::Edge edge) const { return m_observedEdges[edge]; }
        const double getEdgeCountProb(BaseGraph::Edge edge, size_t count) const;
        const std::map<BaseGraph::Edge, std::vector<double>> getEdgeProbs();
    };

    template <typename GraphMCMC>
    void CollectEdgeMultiplicityOnSweep<GraphMCMC>::collect()
    {
        ++m_totalCount;
        const MultiGraph &graph = getCurrentGraph();

        for (const auto &edge : graph.edges())
        {
            const auto oedge = getOrderedPair<BaseGraph::VertexIndex>(edge);
            const auto mult = graph.getEdgeMultiplicity(edge.first, edge.second);
            m_observedEdges.increment(oedge);
            m_observedEdgesCount.increment({oedge, mult});
            if (mult > m_observedEdgesMaxCount[oedge])
                m_observedEdgesMaxCount.set(oedge, mult);
        }
    }

    template <typename GraphMCMC>
    const double CollectEdgeMultiplicityOnSweep<GraphMCMC>::getEdgeCountProb(BaseGraph::Edge edge, size_t count) const
    {
        if (count == 0)
            return 1.0 - ((double)m_observedEdges.get(edge)) / ((double)m_totalCount);
        else
            return ((double)m_observedEdgesCount.get({edge, count})) / ((double)m_totalCount);
    }

    template <typename GraphMCMC>
    const double CollectEdgeMultiplicityOnSweep<GraphMCMC>::getMarginalEntropy()
    {
        double marginalEntropy = 0;
        for (auto edge : m_observedEdges)
        {
            for (size_t count = 0; count <= m_observedEdgesMaxCount[edge.first]; ++count)
            {
                double p = getEdgeCountProb(edge.first, count);
                if (p > 0)
                    marginalEntropy -= p * log(p);
            }
        }
        return marginalEntropy;
    }

    template <typename GraphMCMC>
    const double CollectEdgeMultiplicityOnSweep<GraphMCMC>::getLogPosteriorEstimate(const MultiGraph &graph)
    {
        double logPosterior = 0;
        for (auto edge : m_observedEdges)
            logPosterior += log(getEdgeCountProb(edge.first, graph.getEdgeMultiplicity(edge.first.first, edge.first.second)));
        return logPosterior;
    }

    template <typename GraphMCMC>
    const std::map<BaseGraph::Edge, std::vector<double>> CollectEdgeMultiplicityOnSweep<GraphMCMC>::getEdgeProbs()
    {
        std::map<BaseGraph::Edge, std::vector<double>> edgeProbs;

        for (auto edge : m_observedEdges)
        {
            edgeProbs.insert({edge.first, {}});
            for (size_t count = 0; count <= m_observedEdgesMaxCount[edge.first]; ++count)
            {
                double p = getEdgeCountProb(edge.first, count);
                edgeProbs[edge.first].push_back(p);
            }
        }
        return edgeProbs;
    }

    template <typename MCMCType>
    class CollectPartitionOnSweep : public SweepCollector<MCMCType>
    {
    private:
        std::vector<std::vector<BlockIndex>> m_partitions;

    public:
        using BaseClass = SweepCollector<MCMCType>;
        void collect() override
        {
            m_partitions.push_back(BaseClass::m_mcmcPtr->getGraphPrior().getLabels());
        }
        void clear() override { m_partitions.clear(); }
        const std::vector<BlockSequence> &getData() const { return m_partitions; }
    };

    template <typename MCMCType>
    class CollectNestedPartitionOnSweep : public SweepCollector<MCMCType>
    {
    private:
        std::vector<std::vector<BlockSequence>> m_nestedPartitions;

    public:
        using BaseClass = SweepCollector<MCMCType>;
        void collect() override
        {
            m_nestedPartitions.push_back(BaseClass::m_mcmcPtr->getGraphPrior().getNestedLabels());
        }
        void clear() override { m_nestedPartitions.clear(); }
        const std::vector<std::vector<BlockSequence>> &getData() const { return m_nestedPartitions; }
    };

    class CollectLikelihoodOnSweep : public SweepCollector<MCMC>
    {
    private:
        std::vector<double> m_collectedLikelihoods;

    public:
        void collect() override { m_collectedLikelihoods.push_back(m_mcmcPtr->getLogLikelihood()); }
        void clear() override { m_collectedLikelihoods.clear(); }
        const std::vector<double> &getData() const { return m_collectedLikelihoods; }
    };

    class CollectPriorOnSweep : public SweepCollector<MCMC>
    {
    private:
        std::vector<double> m_collectedPriors;

    public:
        void collect() override { m_collectedPriors.push_back(m_mcmcPtr->getLogPrior()); }
        void clear() override { m_collectedPriors.clear(); }
        const std::vector<double> &getData() const { return m_collectedPriors; }
    };

    class CollectJointOnSweep : public SweepCollector<MCMC>
    {
    private:
        std::vector<double> m_collectedJoints;

    public:
        void collect() override { m_collectedJoints.push_back(m_mcmcPtr->getLogJoint()); }
        void clear() override { m_collectedJoints.clear(); }
        const std::vector<double> &getData() const { return m_collectedJoints; }
    };

}

#endif