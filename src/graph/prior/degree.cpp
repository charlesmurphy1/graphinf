#include <map>
#include <random>
#include <string>
#include <tuple>
#include <vector>

#include "GraphInf/graph/prior/degree.h"
#include "GraphInf/generators.h"
#include "GraphInf/utility/functions.h"
#include "GraphInf/utility/integer_partition.h"
#include "GraphInf/utility/maps.hpp"

using namespace std;

namespace GraphInf
{

    const DegreeCountsMap DegreePrior::computeDegreeCounts(const std::vector<size_t> &degrees)
    {
        DegreeCountsMap degreeCounts;
        for (size_t vertex = 0; vertex < degrees.size(); ++vertex)
        {
            degreeCounts.increment(degrees[vertex]);
        }
        return degreeCounts;
    }

    const size_t DegreePrior::computeEdgeCountFromDegrees(const std::vector<size_t> &degrees)
    {
        size_t edgeCount = 0;
        for (auto k : degrees)
            edgeCount += k;
        return edgeCount / 2;
    }

    void DegreePrior::recomputeConsistentState()
    {
        m_degreeCounts = computeDegreeCounts(m_state);
        m_edgeCountPriorPtr->setState(computeEdgeCountFromDegrees(m_state));
    }

    void DegreePrior::setState(const DegreeSequence &state)
    {
        m_state = state;
        recomputeConsistentState();
    }

    void DegreePrior::setGraph(const MultiGraph &graph)
    {
        m_size = graph.getSize();
        m_state = graph.getDegrees();
        recomputeConsistentState();
    }

    void DegreePrior::applyGraphMoveToState(const GraphMove &move)
    {
        for (auto edge : move.addedEdges)
        {
            ++m_state[edge.first];
            ++m_state[edge.second];
        }
        for (auto edge : move.removedEdges)
        {
            --m_state[edge.first];
            --m_state[edge.second];
        }
    }
    void DegreePrior::applyGraphMoveToDegreeCounts(const GraphMove &move)
    {
        const DegreeSequence &degreeSeq = getState();

        IntMap<BaseGraph::VertexIndex> diffDegreeMap;
        for (auto edge : move.addedEdges)
        {
            diffDegreeMap.increment(edge.first);
            diffDegreeMap.increment(edge.second);
        }
        for (auto edge : move.removedEdges)
        {
            diffDegreeMap.decrement(edge.first);
            diffDegreeMap.decrement(edge.second);
        }

        for (auto diff : diffDegreeMap)
        {
            m_degreeCounts.decrement(degreeSeq[diff.first]);
            m_degreeCounts.increment(degreeSeq[diff.first] + diff.second);
        }
    }

    void DegreePrior::_applyGraphMove(const GraphMove &move)
    {
        m_edgeCountPriorPtr->applyGraphMove(move);
        applyGraphMoveToDegreeCounts(move);
        applyGraphMoveToState(move);
    }

    void DegreePrior::checkDegreeSequenceConsistencyWithEdgeCount(const DegreeSequence &degreeSeq, size_t expectedEdgeCount)
    {
        size_t actualEdgeCount = 0;
        for (auto k : degreeSeq)
            actualEdgeCount += k;
        if (actualEdgeCount != 2 * expectedEdgeCount)
            throw ConsistencyError(
                "DegreePrior",
                "m_state", "E=" + std::to_string(expectedEdgeCount),
                "edgeCount", "value=" + std::to_string(actualEdgeCount));
    }

    void DegreePrior::checkDegreeSequenceConsistencyWithDegreeCounts(
        const DegreeSequence &degreeSeq,
        const DegreeCountsMap &expected)
    {
        DegreeCountsMap actual = DegreePrior::computeDegreeCounts(degreeSeq);
        if (expected.size() != actual.size())
            throw ConsistencyError(
                "DegreePrior",
                "m_state", "countSize=" + std::to_string(actual.size()),
                "vertexCounts", "size=" + std::to_string(expected.size()));

        for (auto nk : actual)
        {
            if (expected.get(nk.first) != nk.second)
                throw ConsistencyError(
                    "DegreePrior",
                    "m_state", "count=" + std::to_string(nk.second),
                    "vertexCounts", "value=" + std::to_string(expected.get(nk.first)),
                    "k=" + std::to_string(nk.first));
        }
    }

    void DegreePrior::checkSelfConsistency() const
    {
        m_edgeCountPriorPtr->checkConsistency();
        checkDegreeSequenceConsistencyWithEdgeCount(getState(), getEdgeCount());
        checkDegreeSequenceConsistencyWithDegreeCounts(getState(), getDegreeCounts());
    };

    const double DegreeDeltaPrior::getLogLikelihoodRatioFromGraphMove(const GraphMove &move) const
    {
        CounterMap<BaseGraph::VertexIndex> map;

        for (auto edge : move.addedEdges)
        {
            map.increment(edge.first);
            map.increment(edge.second);
        }
        for (auto edge : move.addedEdges)
        {
            map.decrement(edge.first);
            map.decrement(edge.second);
        }

        for (auto k : map)
        {
            if (k.second != 0)
                return -INFINITY;
        }
        return 0.;
    }

    void DegreeUniformPrior::sampleState()
    {
        auto degreeList = sampleRandomWeakComposition(2 * getEdgeCount(), getSize());
        DegreeSequence degreeSeq;
        for (auto k : degreeList)
            degreeSeq.push_back(k);
        setState(degreeSeq);
    }

    void DegreeUniformHyperPrior::sampleState()
    {
        auto orderedDegreeList = sampleRandomRestrictedPartition(2 * getEdgeCount(), getSize());
        std::vector<size_t> degreeSeq(orderedDegreeList.begin(), orderedDegreeList.end());
        std::shuffle(std::begin(degreeSeq), std::end(degreeSeq), rng);
        setState(degreeSeq);
    }

    const double DegreeUniformHyperPrior::getLogLikelihood() const
    {
        double logLikelihood = -logMultinomialCoefficient(m_degreeCounts.getValues()) - log_q(2 * getEdgeCount(), getSize(), m_exact);
        return logLikelihood;
    }

    const double DegreeUniformHyperPrior::getLogLikelihoodRatioFromGraphMove(const GraphMove &move) const
    {
        IntMap<size_t> diffDegreeMap, diffDegreeCountMap;
        int dE = move.addedEdges.size() - move.removedEdges.size();

        for (auto edge : move.addedEdges)
        {
            diffDegreeMap.increment(edge.first);
            diffDegreeMap.increment(edge.second);
        }

        for (auto edge : move.removedEdges)
        {
            diffDegreeMap.decrement(edge.first);
            diffDegreeMap.decrement(edge.second);
        }

        for (auto diff : diffDegreeMap)
        {
            diffDegreeCountMap.increment(m_state[diff.first] + diff.second);
            diffDegreeCountMap.decrement(m_state[diff.first]);
        }

        double logLikelihoodRatio = log_q(2 * getEdgeCount(), getSize(), m_exact) - log_q(2 * (getEdgeCount() + dE), getSize(), m_exact);
        for (auto diff : diffDegreeCountMap)
        {
            logLikelihoodRatio += logFactorial(m_degreeCounts.get(diff.first) + diff.second);
            logLikelihoodRatio -= logFactorial(m_degreeCounts.get(diff.first));
        }
        return logLikelihoodRatio;
    }

} // GraphInf
