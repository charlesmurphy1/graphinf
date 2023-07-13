#ifndef GRAPHINF_PARAM_PROPOSER_H
#define GRAPHINF_PARAM_PROPOSER_H

#include <random>
#include <cmath>
#include <map>
#include <algorithm>
#include <sstream>
#include <memory>
#include "GraphInf/utility/polylog2_integral.h"
#include "GraphInf/rng.h"
#include "SamplableSet.hpp"

namespace GraphInf
{

    class ParamProposer
    {

    public:
        ParamProposer() {}
        virtual ~ParamProposer() {}

        virtual double proposeMove() const = 0;
        virtual double logProposal(const double move) const = 0;
        double logProposalRatio(const double move) const { return logProposal(-move) - logProposal(move); }
    };

    class StepParamProposer : public ParamProposer
    {
    private:
        double m_stepSize, m_prob;
        mutable std::bernoulli_distribution m_bernoulli;

    public:
        StepParamProposer(const double stepSize = 0.01, const double prob = 0.5) : ParamProposer(), m_stepSize(stepSize), m_prob(prob), m_bernoulli(prob) {}

        double proposeMove() const override
        {
            if (m_bernoulli(rng))
                return m_stepSize;
            return -m_stepSize;
        }

        double logProposal(const double move) const override
        {
            if (move == m_stepSize)
                return log(m_prob);
            return log(1 - m_prob);
        }
    };

    class GaussianParamProposer : public ParamProposer
    {
    private:
        double m_mean, m_stddev;
        mutable std::normal_distribution<double> m_normal;

    public:
        GaussianParamProposer(const double mean = 0, const double stddev = 0.1) : ParamProposer(), m_mean(mean), m_stddev(stddev), m_normal(mean, stddev) {}

        double proposeMove() const override { return m_normal(rng); }

        double logProposal(const double move) const override
        {
            double z = (move - m_mean) / m_stddev;
            return -pow(z, 2) - 0.5 * log(2 * PI * m_stddev * m_stddev);
        }
    };

    struct ParamMove
    {
        ParamMove(std::string key, double value) : key(key), value(value) {}

        std::string key;
        double value;

        friend std::ostream &operator<<(std::ostream &os, const ParamMove &move)
        {
            os << move.display();
            return os;
        }

        std::string display() const
        {
            return "ParamMove(" + key + "=" + std::to_string(value) + ")";
        }
    };

    class MultiParamProposer
    {
    private:
        sset::SamplableSet<std::string> m_moveSampler;
        std::map<std::string, std::shared_ptr<ParamProposer>> m_proposersPtrMap;

    public:
        MultiParamProposer(double min = 1, double max = 10) : m_moveSampler(min, max) {}
        ~MultiParamProposer() {}
        void insertStepProposer(std::string key, double rate = 1, double stepSize = 0.01, double p = 0.5)
        {
            m_proposersPtrMap.insert({key, std::shared_ptr<ParamProposer>(new StepParamProposer(stepSize, p))});
            m_moveSampler.insert(key, rate);
        }
        void insertGaussianProposer(std::string key, double rate = 1, double mean = 0, double scale = 0.1)
        {
            m_proposersPtrMap.insert({key, std::shared_ptr<ParamProposer>(new GaussianParamProposer(mean, scale))});
            m_moveSampler.insert(key, rate);
        }
        void erase(std::string key)
        {
            m_proposersPtrMap.erase(key);
            m_moveSampler.erase(key);
        }
        size_t size() { return m_moveSampler.size(); }
        void freeze(std::string key)
        {
            m_moveSampler.erase(key);
        }
        void unfreeze(std::string key, double weight = 1)
        {
            if (m_proposersPtrMap.count(key) == 0)
                return;
            m_moveSampler.insert(key, weight);
        }

        const ParamMove proposeMove(std::string key) const
        {
            auto value = m_proposersPtrMap.at(key)->proposeMove();
            return ParamMove(key, value);
        }
        const ParamMove proposeMove() const
        {
            const auto key = m_moveSampler.sample_ext_RNG(rng).first;
            return proposeMove(key);
        }
        double logProposalRatio(const ParamMove move) const
        {
            return m_proposersPtrMap.at(move.key)->logProposalRatio(move.value);
        }
    };

}
#endif