#ifndef GRAPH_INF_EDGE_COUNT_H
#define GRAPH_INF_EDGE_COUNT_H

#include "GraphInf/types.h"
#include "GraphInf/rng.h"
#include "GraphInf/utility/functions.h"
#include "prior.hpp"

namespace GraphInf
{

    class EdgeCountPrior : public Prior<size_t>
    {
    protected:
        void _samplePriors() override {}
        const double _getLogPrior() const override { return 0; }
        void _applyGraphMove(const GraphMove &move) override { setState(getStateAfterGraphMove(move)); }

        const double _getLogPriorRatioFromGraphMove(const GraphMove &move) const override
        {
            return 0;
        }

    public:
        using Prior<size_t>::Prior;
        virtual const double getLogLikelihoodFromState(const size_t &) const = 0;
        const double getLogLikelihood() const override { return getLogLikelihoodFromState(m_state); }
        virtual const double getLogLikelihoodRatioFromGraphMove(const GraphMove &move) const override
        {
            return getLogLikelihoodFromState(getStateAfterGraphMove(move)) - getLogLikelihood();
        }

        const double getLogPriorRatioFromGraphMove(const GraphMove &move) const { return 0; }

        size_t getStateAfterGraphMove(const GraphMove &move) const;
    };

    class EdgeCountDeltaPrior : public EdgeCountPrior
    {
    private:
        size_t m_edgeCount = 0;

    public:
        EdgeCountDeltaPrior() {}
        EdgeCountDeltaPrior(const size_t &edgeCount) : m_edgeCount(edgeCount) { setState(m_edgeCount); }
        EdgeCountDeltaPrior(const EdgeCountDeltaPrior &other) { setState(m_edgeCount); }
        virtual ~EdgeCountDeltaPrior() {}
        const EdgeCountDeltaPrior &operator=(const EdgeCountDeltaPrior &other)
        {
            setState(other.m_edgeCount);
            return *this;
        }
        void setState(const size_t &state) override
        {
            Prior<size_t>::setState(state);
            m_edgeCount = state;
        }

        void sampleState() override {};
        const double getLogLikelihoodFromState(const size_t &state) const override
        {
            if (state == m_state)
                return 0.;
            else
                return -INFINITY;
        };
        const double getLogLikelihoodRatioFromGraphMove(const GraphMove &move) const override
        {
            if (move.addedEdges.size() == move.removedEdges.size())
                return 0;
            else
                return -INFINITY;
        }
    };

    class EdgeCountPoissonPrior : public EdgeCountPrior
    {
    private:
        double m_mean;
        std::poisson_distribution<size_t> m_poissonDistribution;

    public:
        EdgeCountPoissonPrior() {}
        EdgeCountPoissonPrior(double mean) { setMean(mean); }
        EdgeCountPoissonPrior(const EdgeCountPoissonPrior &other)
        {
            setMean(other.m_mean);
            setState(other.m_state);
        }
        virtual ~EdgeCountPoissonPrior() {};
        const EdgeCountPoissonPrior &operator=(const EdgeCountPoissonPrior &other)
        {
            setMean(other.m_mean);
            setState(other.m_state);
            return *this;
        }

        double getMean() const { return m_mean; }
        void setMean(double mean)
        {
            m_mean = mean;
            m_poissonDistribution = std::poisson_distribution<size_t>(mean);
        }
        void sampleState() override { setState(m_poissonDistribution(rng)); }
        const double getLogLikelihoodFromState(const size_t &state) const override { return logPoissonPMF(state, m_mean); }
        void checkSelfSafety() const override
        {
            if (m_mean < 0)
                throw SafetyError("EdgeCountPoissonPrior", "m_mean", "<0");
            if (m_state < 0)
                throw SafetyError("EdgeCountPoissonPrior", "m_state", "<0");
        }
    };

    class EdgeCountGeometricPrior : public EdgeCountPrior
    {
    private:
        double m_mean;
        std::geometric_distribution<size_t> m_geometricDistribution;

    public:
        EdgeCountGeometricPrior() {}
        EdgeCountGeometricPrior(double mean) { setMean(mean); }
        EdgeCountGeometricPrior(const EdgeCountGeometricPrior &other)
        {
            setMean(other.m_mean);
            setState(other.m_state);
        }
        virtual ~EdgeCountGeometricPrior() {};
        const EdgeCountGeometricPrior &operator=(const EdgeCountGeometricPrior &other)
        {
            setMean(other.m_mean);
            setState(other.m_state);
            return *this;
        }

        double getMean() const { return m_mean; }
        void setMean(double mean)
        {
            m_mean = mean;
            double p = 1. / (m_mean + 1);
            m_geometricDistribution = std::geometric_distribution<size_t>(p);
        }
        void sampleState() override { setState(m_geometricDistribution(rng)); }
        const double getLogLikelihoodFromState(const size_t &state) const override
        {
            double p = 1. / (m_mean + 1);
            return state * log(1 - p) + log(p);
        }
        void checkSelfSafety() const override
        {
            if (m_mean < 0)
                throw SafetyError("EdgeCountExponentialPrior", "m_mean", "<0");
            if (m_state < 0)
                throw SafetyError("EdgeCountExponentialPrior", "m_state", "<0");
        }
    };

    class EdgeCountUniformPrior : public EdgeCountPrior
    {
    private:
        size_t m_min, m_max;
        std::uniform_int_distribution<size_t> m_uniformDistribution;

    public:
        EdgeCountUniformPrior() {}
        EdgeCountUniformPrior(double min, double max) : m_min(min), m_max(max), m_uniformDistribution(min, max) { setState(m_uniformDistribution(rng)); }
        EdgeCountUniformPrior(const EdgeCountUniformPrior &other)
        {
            setMinMax(other.getMin(), other.getMax());
            setState(other.getState());
        }
        virtual ~EdgeCountUniformPrior() {};
        const EdgeCountUniformPrior &operator=(const EdgeCountUniformPrior &other)
        {
            setMinMax(other.m_min, other.m_max);
            setState(other.m_state);
            return *this;
        }

        double getMin() const { return m_min; }
        double getMax() const { return m_max; }
        void setMinMax(double min, double max)
        {
            m_min = min;
            m_max = max;
            m_uniformDistribution = std::uniform_int_distribution<size_t>(min, max);
        }
        void sampleState() override { setState(m_uniformDistribution(rng)); }
        const double getLogLikelihoodFromState(const size_t &state) const override
        {
            if (state >= m_min && state <= m_max)
                return 0;
            return -INFINITY;
        }
        void checkSelfSafety() const override {}
    };
}

#endif
