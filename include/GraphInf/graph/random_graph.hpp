#ifndef GRAPH_INF_GRAPH_H
#define GRAPH_INF_GRAPH_H

#include <vector>

#include "GraphInf/types.h"
#include "GraphInf/rv.hpp"
#include "GraphInf/exceptions.h"
#include "GraphInf/mcmc.h"
#include "GraphInf/utility/maps.hpp"
#include "GraphInf/mcmc.h"
#include "GraphInf/graph/likelihood/likelihood.hpp"
#include "GraphInf/graph/prior/edge_count.h"

#include "GraphInf/graph/proposer/edge/hinge_flip.h"
#include "GraphInf/graph/proposer/edge/double_edge_swap.h"
#include "GraphInf/graph/proposer/edge/single_edge.h"
#include "GraphInf/graph/proposer/label/base.hpp"
#include "GraphInf/graph/proposer/nested_label/base.hpp"
#include "GraphInf/graph/util.h"

// #include "GraphInf/graph/util.h"

namespace GraphInf
{

    class RandomGraph : public NestedRandomVariable
    {
    private:
        int m_samplingIteration = 0;
        int m_maxIteration = 100;

    protected:
        GraphLikelihoodModel *m_likelihoodModelPtr = nullptr;
        SingleEdgeProposer m_singleEdgeProposer;
        HingeFlipUniformProposer m_hingeFlipProposer;
        DoubleEdgeSwapProposer m_doubleEdgeSwapProposer;
        EdgeCountPrior *m_edgeCountPriorPtr = nullptr;
        std::string m_graphMoveType = "canonical";
        mutable std::discrete_distribution<int> m_canonicalProposer = std::discrete_distribution<int>({1, 1, 1});
        mutable std::discrete_distribution<int> m_microcanonicalProposer = std::discrete_distribution<int>({1, 1});
        bool m_withSelfLoops, m_withParallelEdges;
        size_t m_size;
        MultiGraph m_state;
        virtual void _applyGraphMove(const GraphMove &);
        void _applyGraphMoveToProposers(const GraphMove &move)
        {
            m_singleEdgeProposer.applyGraphMove(move);
            m_hingeFlipProposer.applyGraphMove(move);
            m_doubleEdgeSwapProposer.applyGraphMove(move);
        }
        virtual const double _getLogPrior() const { return 0; }
        virtual const double _getLogPriorRatioFromGraphMove(const GraphMove &move) const { return 0; }
        virtual void sampleOnlyPrior() {};
        virtual void setUpLikelihood()
        {
            m_likelihoodModelPtr->m_statePtr = &m_state;
        }
        virtual void setUp();
        virtual void computeConsistentState()
        {
            m_edgeCountPriorPtr->setState(m_state.getTotalEdgeNumber());
        }

        GraphMove proposeCanonicalMove() const
        {
            auto s = m_canonicalProposer(rng);
            if (s == 2 && getEdgeCount() > 2)
                return m_doubleEdgeSwapProposer.proposeMove();
            if (s == 1 && getEdgeCount() > 1)
                return m_hingeFlipProposer.proposeMove();
            return m_singleEdgeProposer.proposeMove();
        }

        GraphMove proposeMicrocanonicalMove() const
        {
            auto s = m_microcanonicalProposer(rng);
            if (s == 1 && getEdgeCount() > 2)
                return m_doubleEdgeSwapProposer.proposeMove();
            if (getEdgeCount() > 1)
                return m_hingeFlipProposer.proposeMove();
            throw std::runtime_error("RandomGraph: cannot propose microcanonical move with less than 1 edges.");
        }

    public:
        RandomGraph(size_t size, double edgeCount, bool canonical = false, bool withSelfLoops = true, bool withParallelEdges = true) : m_size(size), m_state(size),
                                                                                                                                       m_withSelfLoops(withSelfLoops),
                                                                                                                                       m_withParallelEdges(withParallelEdges),
                                                                                                                                       m_singleEdgeProposer(size, 0.5, withSelfLoops, withParallelEdges),
                                                                                                                                       m_hingeFlipProposer(withSelfLoops, withParallelEdges),
                                                                                                                                       m_doubleEdgeSwapProposer(withSelfLoops, withParallelEdges)
        {

            setUpEdgeCountPrior(edgeCount, canonical);
        }

        RandomGraph(
            size_t size,
            double edgeCount,
            GraphLikelihoodModel &likelihoodModel,
            bool canonical = false,
            bool withSelfLoops = true,
            bool withParallelEdges = true) : m_size(size), m_state(size),
                                             m_likelihoodModelPtr(&likelihoodModel),
                                             m_withSelfLoops(withSelfLoops),
                                             m_withParallelEdges(withParallelEdges),
                                             m_singleEdgeProposer(size, 0.5, withSelfLoops, withParallelEdges),
                                             m_hingeFlipProposer(withSelfLoops, withParallelEdges),
                                             m_doubleEdgeSwapProposer(withSelfLoops, withParallelEdges)
        {

            setUpEdgeCountPrior(edgeCount, canonical);
        }

        virtual ~RandomGraph()
        {
        }

        const MultiGraph &getState() const { return m_state; }

        virtual void setState(const MultiGraph &state)
        {
            if (state.getSize() != m_size)
                throw std::invalid_argument("Cannot set state with graph of size " + std::to_string(state.getSize()) + " != " + std::to_string(m_size));
            m_state = MultiGraph(state);

            computeConsistentState();
            setUp();
        }
        const size_t getSize() const { return m_size; }
        void setSize(const size_t size) { m_size = size; }
        const size_t getEdgeCount() const { return m_edgeCountPriorPtr->getState(); };
        const double getAverageDegree() const
        {
            double avgDegree = 2 * (double)getEdgeCount();
            avgDegree /= (double)getSize();
            return avgDegree;
        }
        const bool withSelfLoops() const { return m_withSelfLoops; }
        const bool withSelfLoops(bool condition) { return m_withSelfLoops = condition; }
        const bool withParallelEdges() const { return m_withParallelEdges; }
        const bool withParallelEdges(bool condition) { return m_withParallelEdges = condition; }
        void setGraphMoveType(std::string moveType)
        {
            std::vector<std::string> validMoveTypes = {"canonical", "microcanonical", "single_edge", "hinge_flip", "double_edge_swap"};
            if (std::find(validMoveTypes.begin(), validMoveTypes.end(), moveType) == validMoveTypes.end())
                throw std::invalid_argument("RandomGraph: invalid move type " + moveType + ".");
            m_graphMoveType = moveType;
        }
        std::string getGraphMoveType() const { return m_graphMoveType; }

        SingleEdgeProposer &getSingleEdgeProposer()
        {
            return m_singleEdgeProposer;
        }

        HingeFlipUniformProposer &getHingeFlipProposer()
        {
            return m_hingeFlipProposer;
        }

        DoubleEdgeSwapProposer &getDoubleEdgeSwapProposer()
        {
            return m_doubleEdgeSwapProposer;
        }
        const EdgeCountPrior &getEdgeCountPrior() const
        {
            return *m_edgeCountPriorPtr;
        }
        void setUpEdgeCountPrior(double edgeCount, bool canonical)
        {
            if (canonical)
            {
                m_edgeCountPriorPtr = new EdgeCountPoissonPrior(edgeCount);
                setGraphMoveType("canonical");
            }
            else
            {
                m_edgeCountPriorPtr = new EdgeCountDeltaPrior(edgeCount);
                setGraphMoveType("microcanonical");
            }
            m_edgeCountPriorPtr->isRoot(false);
        }

        void sample()
        {

            try
            {
                processRecursiveFunction([&]()
                                         { sampleOnlyPrior(); });
                sampleState();
                setUp();
            }
            catch (std::invalid_argument)
            {
                if (m_samplingIteration < m_maxIteration)
                {
                    ++m_samplingIteration;
                    sample();
                }
                else
                    throw std::runtime_error("RandomGraph: could not sample the model after " + std::to_string(m_maxIteration) + " iterations.");
            }
            m_samplingIteration = 0;
        }
        void sampleState()
        {

            auto g = m_likelihoodModelPtr->sample();
            setState(m_likelihoodModelPtr->sample());
            computationFinished();
        }
        void samplePrior()
        {
            processRecursiveFunction([&]()
                                     {
            sampleOnlyPrior();
            computeConsistentState(); });
        }

        virtual const MCMCSummary metropolisSweep(size_t numSteps, const double betaPrior = 1, const double betaLikelihood = 1)
        {
            MCMCSummary summary;
            return summary;
        }

        const double getLogLikelihood() const
        {
            return m_likelihoodModelPtr->getLogLikelihood();
        }
        const double getLogLikelihoodRatioFromGraphMove(const GraphMove &move) const { return m_likelihoodModelPtr->getLogLikelihoodRatioFromGraphMove(move); }
        const double getLogProposalRatioFromGraphMove(const GraphMove &move) const;

        const double getLogPrior() const
        {
            return processRecursiveFunction<double>([&]()
                                                    { return _getLogPrior(); },
                                                    0);
        }
        const double getLogPriorRatioFromGraphMove(const GraphMove &move) const
        {
            return processRecursiveConstFunction<double>([&]()
                                                         { return _getLogPriorRatioFromGraphMove(move); },
                                                         0);
        }

        const double getLogJoint() const
        {
            return getLogLikelihood() + getLogPrior();
        }
        const double getLogJointRatioFromGraphMove(const GraphMove &move) const
        {
            return getLogPriorRatioFromGraphMove(move) + getLogLikelihoodRatioFromGraphMove(move);
        }

        void applyGraphMove(const GraphMove &move);
        const GraphMove proposeGraphMove() const;
        bool isTrivialGraphMove(const GraphMove &move) const { return m_singleEdgeProposer.isTrivialMove(move); }

        virtual const bool isCompatible(const MultiGraph &graph) const { return graph.getSize() == m_size; }
        virtual bool isSafe() const override { return m_likelihoodModelPtr and m_likelihoodModelPtr->isSafe(); }
        virtual void checkSelfSafety() const override
        {
            if (m_likelihoodModelPtr == nullptr)
                throw SafetyError("RandomGraph", "m_likelihoodModelPtr");
        }
        virtual void checkSelfConsistency() const override
        {
            m_likelihoodModelPtr->checkConsistency();
            m_singleEdgeProposer.checkConsistency();
            m_hingeFlipProposer.checkConsistency();
            m_doubleEdgeSwapProposer.checkConsistency();
        }

        virtual bool isValidGraphMove(const GraphMove &move) const { return true; }
    };

    template <typename Label>
    class VertexLabeledRandomGraph : public RandomGraph
    {
    protected:
        LabelProposer<Label> *m_labelProposerPtr = nullptr;
        virtual void _applyLabelMove(const LabelMove<Label> &) {};
        virtual const double _getLogPriorRatioFromLabelMove(const LabelMove<Label> &move) const { return 0; }
        VertexLabeledGraphLikelihoodModel<Label> *m_vertexLabeledlikelihoodModelPtr = nullptr;
        std::uniform_real_distribution<double> m_uniform;

        virtual void setUp() override;

    public:
        VertexLabeledRandomGraph(size_t size, double edgeCount, bool canonical = false, bool withSelfLoops = true, bool withParallelEdges = true) : RandomGraph(size, edgeCount, canonical, withSelfLoops, withParallelEdges), m_uniform(0, 1)
        {
        }
        VertexLabeledRandomGraph(
            size_t size, double edgeCount, VertexLabeledGraphLikelihoodModel<Label> &likelihoodModel, bool canonical = false,
            bool withSelfLoops = true, bool withParallelEdges = true) : RandomGraph(size, edgeCount, likelihoodModel, canonical, withSelfLoops, withParallelEdges), m_uniform(0, 1),
                                                                        m_vertexLabeledlikelihoodModelPtr(&likelihoodModel)
        {
        }
        virtual ~VertexLabeledRandomGraph() {}
        virtual const std::vector<Label> &getLabels() const = 0;
        virtual const size_t getLabelCount() const = 0;
        virtual const CounterMap<Label> &getVertexCounts() const = 0;
        virtual const CounterMap<Label> &getEdgeLabelCounts() const = 0;
        virtual const LabelGraph &getLabelGraph() const = 0;
        const Label &getLabel(BaseGraph::VertexIndex vertex) const { return getLabels()[vertex]; }

        virtual void setLabels(const std::vector<Label> &, bool reduce = false) = 0;
        virtual void sampleOnlyLabels() = 0;
        virtual void sampleWithLabels() = 0;

        void setLabelProposer(LabelProposer<Label> &proposer)
        {
            proposer.isRoot(false);
            m_labelProposerPtr = &proposer;
        }
        const LabelProposer<Label> &getLabelProposer() const
        {
            return *m_labelProposerPtr;
        }
        LabelProposer<Label> &getLabelProposerRef()
        {
            return *m_labelProposerPtr;
        }

        virtual const double getLabelLogJoint() const = 0;

        const double getLogLikelihoodRatioFromLabelMove(const LabelMove<Label> &move) const
        {
            return m_vertexLabeledlikelihoodModelPtr->getLogLikelihoodRatioFromLabelMove(move);
        }
        const double getLogPriorRatioFromLabelMove(const LabelMove<Label> &move) const
        {
            return processRecursiveConstFunction<double>([&]()
                                                         { return _getLogPriorRatioFromLabelMove(move); },
                                                         0);
        }
        const double getLogJointRatioFromLabelMove(const LabelMove<Label> &move) const
        {
            return getLogPriorRatioFromLabelMove(move) + getLogLikelihoodRatioFromLabelMove(move);
        }
        const double getLogProposalRatioFromLabelMove(const LabelMove<Label> &move) const;

        const StepResult<LabelMove<Label>> metropolisStep(double m_betaPrior = 1, double m_betaLikelihood = 1)
        {
            const auto move = proposeLabelMove();
            if (m_labelProposerPtr->isTrivialMove(move))
                return {move, 0, true};

            // Log likelihood ratio
            double logLikelihoodRatio = 0;
            if (m_betaLikelihood > 0)
                logLikelihoodRatio = m_betaLikelihood * getLogLikelihoodRatioFromLabelMove(move);

            // Log prior ratio
            double logPriorRatio = 0;
            if (m_betaPrior > 0)
                logPriorRatio = m_betaPrior * getLogPriorRatioFromLabelMove(move);

            // Log proposal ratio
            double logProposalRatio = getLogProposalRatioFromLabelMove(move);

            // Acceptance probability
            double acceptProb = exp(logLikelihoodRatio + logPriorRatio + logProposalRatio);

            // Metropolis-Hastings step
            bool accepted = false;
            if (m_uniform(rng) < acceptProb)
            {
                applyLabelMove(move);
                accepted = true;
            }
            return {move, logLikelihoodRatio + logPriorRatio + logProposalRatio, accepted};
        }

        const MCMCSummary metropolisSweep(size_t numSteps, const double betaPrior = 1, const double betaLikelihood = 1) override
        {
            MCMCSummary summary;
            for (size_t i = 0; i < numSteps; i++)
                summary.update(metropolisStep(betaPrior, betaLikelihood));
            return summary;
        }

        void applyLabelMove(const LabelMove<Label> &move);
        const LabelMove<Label> proposeLabelMove() const;
        virtual bool isValidLabelMove(const LabelMove<Label> &move) const { return true; }
        virtual void checkSelfSafety() const override
        {
            RandomGraph::checkSelfSafety();
            if (m_labelProposerPtr == nullptr)
                throw SafetyError("RandomGraph", "m_labelProposerPtr");
        }
        virtual void checkSelfConsistency() const override
        {
            RandomGraph::checkSelfConsistency();
            m_labelProposerPtr->checkConsistency();
        }
        virtual void reduceLabels() {}
    };

    using BlockLabeledRandomGraph = VertexLabeledRandomGraph<BlockIndex>;

    template <typename Label>
    class NestedVertexLabeledRandomGraph : public VertexLabeledRandomGraph<Label>
    {
    protected:
        NestedLabelProposer<Label> *m_nestedLabelProposerPtr = nullptr;
        using VertexLabeledRandomGraph<Label>::m_state;
        using VertexLabeledRandomGraph<Label>::m_labelProposerPtr;
        virtual void setUp() override;

    public:
        using VertexLabeledRandomGraph<Label>::VertexLabeledRandomGraph;

        void setLabels(const std::vector<Label> &, bool reduce = false) override
        {
            throw DepletedMethodError("NestedVertexLabeledRandomGraph", "setLabels");
        }
        virtual void setNestedLabels(const std::vector<std::vector<Label>> &, bool reduce = false) = 0;

        virtual const size_t getDepth() const = 0;

        virtual const Label getLabel(BaseGraph::VertexIndex vertex, Level level) const = 0;
        virtual const Label getNestedLabel(BaseGraph::VertexIndex vertex, Level level) const = 0;
        virtual const std::vector<std::vector<Label>> &getNestedLabels() const = 0;
        virtual const std::vector<Label> &getNestedLabels(Level) const = 0;
        virtual const std::vector<size_t> &getNestedLabelCount() const = 0;
        virtual const size_t getNestedLabelCount(Level) const = 0;
        virtual const std::vector<CounterMap<Label>> &getNestedVertexCounts() const = 0;
        virtual const CounterMap<Label> &getNestedVertexCounts(Level) const = 0;
        virtual const std::vector<CounterMap<Label>> &getNestedEdgeLabelCounts() const = 0;
        virtual const CounterMap<Label> &getNestedEdgeLabelCounts(Level) const = 0;
        virtual const std::vector<MultiGraph> &getNestedLabelGraph() const = 0;
        virtual const MultiGraph &getNestedLabelGraph(Level) const = 0;

        void setNestedLabelProposer(NestedLabelProposer<Label> &proposer)
        {
            proposer.isRoot(false);
            m_labelProposerPtr = &proposer;
            m_nestedLabelProposerPtr = &proposer;
        }
        const NestedLabelProposer<Label> &getNestedLabelProposer()
        {
            return *m_nestedLabelProposerPtr;
        }
        NestedLabelProposer<Label> &getNestedLabelProposerRef()
        {
            return *m_nestedLabelProposerPtr;
        }

        using VertexLabeledRandomGraph<Label>::getLabel;
        const std::vector<Label> &getLabels() const override { return getNestedLabels()[0]; }
        const size_t getLabelCount() const override { return getNestedLabelCount()[0]; }
        const CounterMap<Label> &getVertexCounts() const override { return getNestedVertexCounts()[0]; }
        const CounterMap<Label> &getEdgeLabelCounts() const override { return getNestedEdgeLabelCounts()[0]; }
        const MultiGraph &getLabelGraph() const override { return getNestedLabelGraph()[0]; }
        virtual void checkSelfConsistency() const override
        {
            RandomGraph::checkSelfConsistency();
            m_labelProposerPtr->checkConsistency();
        }
    };
    using NestedBlockLabeledRandomGraph = NestedVertexLabeledRandomGraph<BlockIndex>;

    template <typename Label>
    void VertexLabeledRandomGraph<Label>::setUp()
    {
        RandomGraph::setUp();
        m_labelProposerPtr->setUpWithPrior(*this);
    }

    template <typename Label>
    const double VertexLabeledRandomGraph<Label>::getLogProposalRatioFromLabelMove(const LabelMove<Label> &move) const
    {
        return m_labelProposerPtr->getLogProposalProbRatio(move);
    }

    template <typename Label>
    void VertexLabeledRandomGraph<Label>::applyLabelMove(const LabelMove<Label> &move)
    {
        processRecursiveFunction([&]()
                                 { _applyLabelMove(move); });
        m_labelProposerPtr->applyLabelMove(move);
#if DEBUG
        checkConsistency();
#endif
    }

    template <typename Label>
    const LabelMove<Label> VertexLabeledRandomGraph<Label>::proposeLabelMove() const
    {
        return m_labelProposerPtr->proposeMove();
    }

    template <typename Label>
    void NestedVertexLabeledRandomGraph<Label>::setUp()
    {
        VertexLabeledRandomGraph<Label>::setUp();
        m_nestedLabelProposerPtr->setUpWithNestedPrior(*this);
    }

} // namespace GraphInf

#endif
