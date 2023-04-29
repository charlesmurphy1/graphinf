#include <algorithm>
#include <random>
#include <string>
#include <vector>

#include "GraphInf/graph/prior/block.h"
#include "GraphInf/utility/functions.h"
#include "GraphInf/utility/maps.hpp"
#include "GraphInf/graph/proposer/movetypes.h"
#include "GraphInf/rng.h"
#include "GraphInf/generators.h"
#include "GraphInf/exceptions.h"

namespace GraphInf
{

    CounterMap<BlockIndex> BlockPrior::computeVertexCounts(const BlockSequence &state)
    {
        CounterMap<BlockIndex> vertexCount;
        for (auto block : state)
        {
            if (block < 0)
                continue;
            vertexCount.increment(block);
        }
        return vertexCount;
    }

    bool BlockPrior::isValidBlockMove(const BlockMove &move) const
    {
        return m_vertexCounts.size() + getAddedBlocks(move) > m_blockCountPriorPtr->getState() + move.addedLabels;
    }

    void BlockPrior::checkBlockSequenceConsistencyWithVertexCounts(
        std::string prefix, const BlockSequence &blockSeq, CounterMap<BlockIndex> expectedVertexCounts)
    {
        CounterMap<BlockIndex> actualVertexCounts = computeVertexCounts(blockSeq);
        if (actualVertexCounts.size() != expectedVertexCounts.size())
            throw ConsistencyError(
                prefix,
                "blockSeq", "B=" + std::to_string(actualVertexCounts.size()),
                "vertexCounts", "B=" + std::to_string(expectedVertexCounts.size()));

        for (size_t i = 0; i < actualVertexCounts.size(); ++i)
        {
            auto x = actualVertexCounts[i];
            auto y = expectedVertexCounts[i];
            if (x != y)
            {
                throw ConsistencyError(
                    prefix,
                    "blockSeq", "counts=" + std::to_string(actualVertexCounts.size()),
                    "vertexCounts", "value=" + std::to_string(expectedVertexCounts.size()),
                    "r=" + std::to_string(i));
            }
        }
    }

    void BlockUniformPrior::sampleState()
    {
        BlockSequence blockSeq(getSize());
        std::uniform_int_distribution<size_t> dist(0, getBlockCount() - 1);
        for (size_t vertex = 0; vertex < getSize(); vertex++)
        {
            blockSeq[vertex] = dist(rng);
        }

        m_state = reducePartition(blockSeq);

        m_vertexCounts = computeVertexCounts(m_state);
    }

    const double BlockUniformPrior::getLogLikelihood() const
    {
        return -(getSize() * log(getBlockCount()));
    }

    const double BlockUniformPrior::getLogLikelihoodRatioFromLabelMove(const BlockMove &move) const
    {
        if (isValidBlockMove(move))
            return -INFINITY;
        size_t prevNumBlocks = m_blockCountPriorPtr->getState();
        size_t newNumBlocks = prevNumBlocks + move.addedLabels;
        double logLikelihoodRatio = 0;
        logLikelihoodRatio += -(double)getSize() * log(newNumBlocks);
        logLikelihoodRatio -= -(double)getSize() * log(prevNumBlocks);
        return logLikelihoodRatio;
    }

    void BlockUniformHyperPrior::sampleState()
    {

        std::list<size_t> vertexCountList = sampleRandomComposition(getSize(), getBlockCount());
        std::vector<size_t> vertexCounts;
        for (auto nr : vertexCountList)
        {
            vertexCounts.push_back(nr);
        }

        std::vector<size_t> blocks = sampleRandomPermutation(vertexCounts);
        m_state.clear();
        for (auto b : blocks)
            m_state.push_back(b);
        m_vertexCounts = computeVertexCounts(m_state);
    }

    const double BlockUniformHyperPrior::getLogLikelihood() const
    {
        return -logMultinomialCoefficient(m_vertexCounts.getValues()) - logBinomialCoefficient(getSize() - 1, getBlockCount() - 1);
    }

    const double BlockUniformHyperPrior::getLogLikelihoodRatioFromLabelMove(const BlockMove &move) const
    {
        if (m_vertexCounts.size() + getAddedBlocks(move) != getBlockCount() + move.addedLabels)
            return -INFINITY;
        double logLikelihoodRatio = 0;
        logLikelihoodRatio += logFactorial(m_vertexCounts[move.prevLabel] - 1) - logFactorial(m_vertexCounts[move.prevLabel]);
        logLikelihoodRatio += logFactorial(m_vertexCounts[move.nextLabel] + 1) - logFactorial(m_vertexCounts[move.nextLabel]);
        logLikelihoodRatio -= logBinomialCoefficient(getSize() - 1, getBlockCount() + move.addedLabels - 1) - logBinomialCoefficient(getSize() - 1, getBlockCount() - 1);
        return logLikelihoodRatio;
    }

}
