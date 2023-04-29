#ifndef GRAPH_INF_DEGREE_H
#define GRAPH_INF_DEGREE_H

#include <map>
#include "GraphInf/types.h"
#include "prior.hpp"
#include "label_graph.h"
#include "GraphInf/graph/proposer/movetypes.h"
#include "GraphInf/utility/maps.hpp"

namespace GraphInf
{

    class VertexLabeledDegreePrior : public BlockLabeledPrior<DegreeSequence>
    {
    protected:
        LabelGraphPrior *m_labelGraphPriorPtr = nullptr;
        VertexLabeledDegreeCountsMap m_degreeCounts;

        void _samplePriors() override { m_labelGraphPriorPtr->sample(); }

        void _applyGraphMove(const GraphMove &move) override;
        void _applyLabelMove(const BlockMove &move) override;

        const double _getLogPrior() const override
        {
            return m_labelGraphPriorPtr->getLogJoint();
        }
        const double _getLogPriorRatioFromGraphMove(const GraphMove &move) const override { return m_labelGraphPriorPtr->getLogJointRatioFromGraphMove(move); }
        const double _getLogPriorRatioFromLabelMove(const BlockMove &move) const override { return m_labelGraphPriorPtr->getLogJointRatioFromLabelMove(move); }

        // void onBlockCreation(const BlockMove&) override;

        void applyGraphMoveToState(const GraphMove &);
        void applyGraphMoveToDegreeCounts(const GraphMove &);
        void applyLabelMoveToDegreeCounts(const BlockMove &);

    public:
        using BlockLabeledPrior<DegreeSequence>::BlockLabeledPrior;
        /* Constructors */
        VertexLabeledDegreePrior() {}
        VertexLabeledDegreePrior(LabelGraphPrior &labelGraphPrior)
        {
            setLabelGraphPrior(labelGraphPrior);
        }
        VertexLabeledDegreePrior(const VertexLabeledDegreePrior &other)
        {
            setLabelGraphPrior(*other.m_labelGraphPriorPtr);
        }
        virtual ~VertexLabeledDegreePrior() {}
        const VertexLabeledDegreePrior &operator=(const VertexLabeledDegreePrior &other)
        {
            setLabelGraphPrior(*other.m_labelGraphPriorPtr);
            return *this;
        }

        void samplePartition()
        {
            m_labelGraphPriorPtr->samplePartition();
            recomputeConsistentState();
        }
        void setGraph(const MultiGraph &);
        // const MultiGraph& getGraph() const { return *m_graphPtr; }
        virtual void setState(const DegreeSequence &) override;
        void setPartition(const std::vector<BlockIndex> &);
        static const VertexLabeledDegreeCountsMap computeDegreeCounts(const std::vector<size_t> &, const std::vector<BlockIndex>);

        const BlockPrior &getBlockPrior() const { return m_labelGraphPriorPtr->getBlockPrior(); }
        BlockPrior &getBlockPriorRef() const { return m_labelGraphPriorPtr->getBlockPriorRef(); }
        void setBlockPrior(BlockPrior &prior) const { m_labelGraphPriorPtr->setBlockPrior(prior); }

        const LabelGraphPrior &getLabelGraphPrior() const { return *m_labelGraphPriorPtr; }
        LabelGraphPrior &getLabelGraphPriorRef() const { return *m_labelGraphPriorPtr; }
        void setLabelGraphPrior(LabelGraphPrior &labelGraphPrior)
        {
            m_labelGraphPriorPtr = &labelGraphPrior;
            m_labelGraphPriorPtr->isRoot(false);
        }

        const size_t &getDegree(BaseGraph::VertexIndex idx) const { return m_state[idx]; }
        virtual const VertexLabeledDegreeCountsMap &getDegreeCounts() const { return m_degreeCounts; }
        void reducePartition()
        {
            m_labelGraphPriorPtr->reducePartition();
            recomputeConsistentState();
        }

        virtual const double getLogLikelihoodRatioFromGraphMove(const GraphMove &) const override = 0;
        virtual const double getLogLikelihoodRatioFromLabelMove(const BlockMove &) const override = 0;

        virtual void computationFinished() const override
        {
            m_isProcessed = false;
            m_labelGraphPriorPtr->computationFinished();
        }
        static void checkDegreeSequenceConsistencyWithEdgeCount(const DegreeSequence &, size_t);
        static void checkDegreeSequenceConsistencyWithDegreeCounts(const DegreeSequence &, const BlockSequence &, const VertexLabeledDegreeCountsMap &);

        bool isSafe() const override
        {
            return (m_labelGraphPriorPtr != nullptr) and (m_labelGraphPriorPtr->isSafe());
        }
        void checkSelfConsistency() const override;
        virtual void checkSelfSafety() const override
        {
            if (m_labelGraphPriorPtr == nullptr)
                throw SafetyError("VertexLabeledDegreePrior", "m_labelGraphPriorPtr");
            m_labelGraphPriorPtr->checkSafety();
        }
        void recomputeConsistentState();
    };

    class VertexLabeledDegreeDeltaPrior : public VertexLabeledDegreePrior
    {
        DegreeSequence m_degreeSeq;

    public:
        VertexLabeledDegreeDeltaPrior() {}
        VertexLabeledDegreeDeltaPrior(const DegreeSequence &degreeSeq) : VertexLabeledDegreePrior() { setState(degreeSeq); }

        VertexLabeledDegreeDeltaPrior(const VertexLabeledDegreeDeltaPrior &degreeDeltaPrior) : VertexLabeledDegreePrior()
        {
            setState(degreeDeltaPrior.getState());
        }
        virtual ~VertexLabeledDegreeDeltaPrior() {}
        const VertexLabeledDegreeDeltaPrior &operator=(const VertexLabeledDegreeDeltaPrior &other)
        {
            this->setState(other.getState());
            return *this;
        }

        void setState(const DegreeSequence &degrees) override
        {
            m_degreeSeq = degrees;
            m_state = degrees;
        }
        void sampleState() override{};

        const double getLogLikelihood() const override { return 0.; }

        const double getLogLikelihoodRatioFromGraphMove(const GraphMove &move) const override;
        const double getLogLikelihoodRatioFromLabelMove(const BlockMove &move) const override
        {
            return (move.prevLabel != move.nextLabel) ? -INFINITY : 0.;
        }
        void checkSelfConsistency() const override{};
        void checkSelfSafety() const override
        {
            if (m_degreeSeq.size() == 0)
                throw SafetyError("DegreeDeltaPrior", "m_degreeSeq", "empty");
        }

        void computationFinished() const override { m_isProcessed = false; }
    };

    class VertexLabeledDegreeUniformPrior : public VertexLabeledDegreePrior
    {
    public:
        using VertexLabeledDegreePrior::VertexLabeledDegreePrior;
        void sampleState() override;

        const double getLogLikelihood() const override;
        const double getLogLikelihoodRatioFromGraphMove(const GraphMove &) const override;
        const double getLogLikelihoodRatioFromLabelMove(const BlockMove &) const override;
    };

    class VertexLabeledDegreeUniformHyperPrior : public VertexLabeledDegreePrior
    {
        bool m_exact;

    public:
        VertexLabeledDegreeUniformHyperPrior(bool exact = false) : VertexLabeledDegreePrior(), m_exact(exact) {}
        VertexLabeledDegreeUniformHyperPrior(LabelGraphPrior &labelGraphPrior, bool exact = false) : VertexLabeledDegreePrior(labelGraphPrior), m_exact(exact) {}
        void sampleState() override;

        const double getLogLikelihood() const override;
        const double getLogLikelihoodRatioFromGraphMove(const GraphMove &) const override;
        const double getLogLikelihoodRatioFromLabelMove(const BlockMove &) const override;
    };

} // GraphInf

#endif
