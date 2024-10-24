#ifndef GRAPH_INF_SBM_H
#define GRAPH_INF_SBM_H

#include <map>
#include <memory>
#include <utility>
#include <vector>

#include "BaseGraph/types.h"
#include "prior/label_graph.h"
#include "prior/block.h"
#include "GraphInf/graph/random_graph.hpp"
#include "GraphInf/graph/util.h"
#include "GraphInf/graph/likelihood/sbm.h"
#include "GraphInf/graph/proposer/edge/edge_proposer.h"
#include "GraphInf/utility/maps.hpp"
#include "GraphInf/generators.h"
#include "GraphInf/types.h"

namespace GraphInf
{

    class StochasticBlockModelBase : public BlockLabeledRandomGraph
    {
    protected:
        std::unique_ptr<StochasticBlockModelLikelihood> m_sbmLikelihoodModelUPtr = nullptr;
        LabelGraphPrior *m_labelGraphPriorPtr = nullptr;
        bool m_stubLabeled;

        void _applyGraphMove(const GraphMove &move) override
        {
            m_labelGraphPriorPtr->applyGraphMove(move);
            RandomGraph::_applyGraphMove(move);
        }
        void _applyLabelMove(const BlockMove &move) override
        {
            m_labelGraphPriorPtr->applyLabelMove(move);
        }
        const double _getLogPrior() const override { return m_labelGraphPriorPtr->getLogJoint(); }
        const double _getLogPriorRatioFromGraphMove(const GraphMove &move) const override { return m_labelGraphPriorPtr->getLogJointRatioFromGraphMove(move); }
        const double _getLogPriorRatioFromLabelMove(const BlockMove &move) const override
        {
            return m_labelGraphPriorPtr->getLogJointRatioFromLabelMove(move);
        }
        void sampleOnlyPrior(bool canonical = true) override
        {
            if (canonical)
                m_labelGraphPriorPtr->sample();
            else
                m_labelGraphPriorPtr->sampleMicrocanonical();
        }
        void setUpLikelihood() override
        {

            m_sbmLikelihoodModelUPtr->m_statePtr = &m_state;
            m_sbmLikelihoodModelUPtr->m_withSelfLoopsPtr = &m_withSelfLoops;
            m_sbmLikelihoodModelUPtr->m_withParallelEdgesPtr = &m_withParallelEdges;
            m_sbmLikelihoodModelUPtr->m_labelGraphPriorPtrPtr = &m_labelGraphPriorPtr;
        }

        StochasticBlockModelBase(size_t size, double edgeCount, bool stubLabeled = true, bool canonical = false, bool withSelfLoops = true, bool withParallelEdges = true) : VertexLabeledRandomGraph<BlockIndex>(size, edgeCount, canonical, withSelfLoops, withParallelEdges),
                                                                                                                                                                             m_stubLabeled(stubLabeled)
        {
            m_sbmLikelihoodModelUPtr = std::unique_ptr<StochasticBlockModelLikelihood>(makeSBMLikelihood(stubLabeled));
            m_likelihoodModelPtr = m_vertexLabeledlikelihoodModelPtr = m_sbmLikelihoodModelUPtr.get();
        }
        void computeConsistentState() override
        {
            m_labelGraphPriorPtr->setGraph(m_state);
        }

    public:
        // void fromGraph(const MultiGraph &graph) override
        // {
        //     RandomGraph::fromGraph(graph);
        //     m_labelGraphPriorPtr->setGraph(m_state);
        //     computationFinished();
        // }
        void sampleOnlyLabels() override
        {
            m_labelGraphPriorPtr->samplePartition();
            m_labelProposerPtr->setUpWithPrior(*this);
            computationFinished();
        }
        void sampleWithLabels() override
        {
            m_labelGraphPriorPtr->sampleState();
            sampleState();
            computationFinished();
        }
        void setLabels(const std::vector<BlockIndex> &labels, bool reduce = false) override
        {
            m_labelGraphPriorPtr->setPartition(labels);
            if (reduce)
                reduceLabels();
        }

        LabelGraphPrior &getLabelGraphPriorRef() const { return *m_labelGraphPriorPtr; }
        const LabelGraphPrior &getLabelGraphPrior() const { return *m_labelGraphPriorPtr; }
        void setLabelGraphPrior(LabelGraphPrior &labelGraphPrior)
        {
            m_labelGraphPriorPtr = &labelGraphPrior;
            m_labelGraphPriorPtr->isRoot(false);
            setUpLikelihood();
        }

        const BlockSequence &getLabels() const override { return m_labelGraphPriorPtr->getBlockPrior().getState(); }
        const size_t getLabelCount() const override { return m_labelGraphPriorPtr->getBlockPrior().getBlockCount(); }
        const CounterMap<BlockIndex> &getVertexCounts() const override { return m_labelGraphPriorPtr->getBlockPrior().getVertexCounts(); }
        const CounterMap<BlockIndex> &getEdgeLabelCounts() const override { return m_labelGraphPriorPtr->getEdgeCounts(); }
        const LabelGraph &getLabelGraph() const override { return m_labelGraphPriorPtr->getState(); }
        const bool isStubLabeled() const { return m_stubLabeled; }
        const double getLabelLogJoint() const override
        {
            double logP = m_labelGraphPriorPtr->getBlockPrior().getLogJoint();
            computationFinished();
            return logP;
        }
        void reduceLabels() override
        {
            m_labelGraphPriorPtr->reducePartition();
            setUp();
        }

        void checkSelfConsistency() const override
        {
            VertexLabeledRandomGraph<BlockIndex>::checkSelfConsistency();
            m_labelGraphPriorPtr->checkSelfConsistency();
            checkGraphConsistencyWithLabelGraph("StochasticBlockModelBase", m_state, getLabels(), getLabelGraph());
        }
        void computationFinished() const override
        {
            m_isProcessed = false;
            m_labelGraphPriorPtr->computationFinished();
        }
        void checkSelfSafety() const override
        {
            RandomGraph::checkSelfSafety();
            if (not m_labelGraphPriorPtr)
                throw SafetyError("StochasticBlockModelBase", "m_labelGraphPriorPtr");
        }
    };

    class StochasticBlockModelFamily : public StochasticBlockModelBase
    {
        std::unique_ptr<BlockCountPrior> m_blockCountPriorUPtr = nullptr;
        std::unique_ptr<BlockPrior> m_blockPriorUPtr = nullptr;
        std::unique_ptr<LabelGraphPrior> m_labelGraphPriorUPtr = nullptr;
        std::unique_ptr<EdgeProposer> m_edgeProposerUPtr = nullptr;
        std::unique_ptr<LabelProposer<BlockIndex>> m_labelProposerUPtr = nullptr;

    public:
        StochasticBlockModelFamily(
            size_t size,
            double edgeCount,
            size_t blockCount = 0,
            bool useBlockHyperPrior = true,
            bool usePlantedPrior = false,
            bool canonical = false,
            bool stubLabeled = true,
            bool withSelfLoops = true,
            bool withParallelEdges = true,
            std::string blockProposerType = "mixed",
            double sampleLabelCountProb = 0.1,
            double labelCreationProb = 0.5,
            double shift = 1) : StochasticBlockModelBase(size, edgeCount, stubLabeled, canonical, withSelfLoops, withParallelEdges)
        {

            if (blockCount < 1 or blockCount > size - 1)
                m_blockCountPriorUPtr = std::unique_ptr<BlockCountPrior>(new BlockCountUniformPrior(1, size));
            else
            {
                m_blockCountPriorUPtr = std::unique_ptr<BlockCountPrior>(new BlockCountDeltaPrior(blockCount));
                sampleLabelCountProb = 0;
            }
            if (stubLabeled)
                withSelfLoops = withParallelEdges = true;

            m_blockPriorUPtr = std::unique_ptr<BlockPrior>(makeBlockPrior(size, *m_blockCountPriorUPtr, useBlockHyperPrior));
            m_labelGraphPriorUPtr = std::unique_ptr<LabelGraphPrior>(makeLabelGraphPrior(*m_edgeCountPriorPtr, *m_blockPriorUPtr, usePlantedPrior));
            setLabelGraphPrior(*m_labelGraphPriorUPtr);

            m_labelProposerUPtr = std::unique_ptr<LabelProposer<BlockIndex>>(
                makeBlockProposer(blockProposerType, useBlockHyperPrior, sampleLabelCountProb, labelCreationProb, shift));
            setLabelProposer(*m_labelProposerUPtr);
            checkSafety();
            sample();
        }
    };

    std::vector<BlockIndex> getPlantedBlocks(std::vector<size_t> sizes);
    std::vector<BlockIndex> getPlantedBlocks(size_t, size_t);
    LabelGraph getPlantedLabelGraph(size_t blockCount, size_t edgeCount, double assortativity = 0);

    // class PlantedPartitionModel : public StochasticBlockModel
    // {
    //     double m_assortativity;

    // public:
    //     PlantedPartitionModel(
    //         std::vector<size_t> sizes,
    //         size_t edgeCount,
    //         double assortativity = 0,
    //         bool stubLabeled = true,
    //         bool withSelfLoops = true,
    //         bool withParallelEdges = true) : StochasticBlockModel(getPlantedBlocks(sizes),
    //                                                               getPlantedLabelGraph(sizes.size(), edgeCount, assortativity),
    //                                                               stubLabeled,
    //                                                               withSelfLoops,
    //                                                               withParallelEdges),
    //                                          m_assortativity(assortativity) {}

    //     PlantedPartitionModel(
    //         size_t size,
    //         size_t edgeCount,
    //         size_t blockCount,
    //         double assortativity = 0,
    //         bool stubLabeled = true,
    //         bool withSelfLoops = true,
    //         bool withParallelEdges = true) : StochasticBlockModel(getPlantedBlocks(size, blockCount),
    //                                                               getPlantedLabelGraph(blockCount, edgeCount, assortativity),
    //                                                               stubLabeled,
    //                                                               withSelfLoops,
    //                                                               withParallelEdges),
    //                                          m_assortativity(assortativity) {}
    //     const double getAssortativity() const
    //     {
    //         return m_assortativity;
    //     }

    //     const bool isCompatible(const MultiGraph &graph) const override
    //     {
    //         if (not VertexLabeledRandomGraph<BlockIndex>::isCompatible(graph))
    //             return false;
    //         auto labelGraph = getLabelGraphFromGraph(graph, getLabels());
    //         return labelGraph.getAdjacencyMatrix() == getLabelGraph().getAdjacencyMatrix();
    //     }
    // };

} // end GraphInf
#endif
