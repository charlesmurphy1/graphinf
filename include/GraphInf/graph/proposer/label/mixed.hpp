#ifndef GRAPH_INF_MIXED_PROPOSER_H
#define GRAPH_INF_MIXED_PROPOSER_H

#include "SamplableSet.hpp"

#include "GraphInf/rng.h"
#include "GraphInf/exceptions.h"
#include "GraphInf/graph/proposer/movetypes.h"
#include "GraphInf/graph/proposer/label/base.hpp"
#include "GraphInf/graph/random_graph.hpp"

namespace GraphInf
{

    template <typename Label>
    class MixedSampler
    {
    protected:
        double m_shift;
        const VertexLabeledRandomGraph<Label> **m_graphPriorPtrPtr = nullptr;
        mutable std::uniform_real_distribution<double> m_uniform01 = std::uniform_real_distribution<double>(0, 1);

        const Label sampleNeighborLabel(BaseGraph::VertexIndex vertex) const;

        virtual const Label sampleLabelUniformly() const = 0;

        const Label sampleLabelFromNeighborLabel(const Label neighborLabel) const;
        const double _getLogProposalProbForMove(const LabelMove<Label> &move) const;
        const double _getLogProposalProbForReverseMove(const LabelMove<Label> &move) const;
        const LabelMove<Label> _proposeLabelMove(const BaseGraph::VertexIndex &) const;
        IntMap<std::pair<Label, Label>> getEdgeMatrixDiff(const LabelMove<Label> &move) const;
        IntMap<Label> getEdgeCountsDiff(const LabelMove<Label> &move) const;

        bool creatingNewLabel(const LabelMove<Label> &move) const
        {
            return (*m_graphPriorPtrPtr)->getVertexCounts().get(move.nextLabel) == 0;
        };
        bool destroyingLabel(const LabelMove<Label> &move) const
        {
            return move.prevLabel != move.nextLabel and (*m_graphPriorPtrPtr)->getVertexCounts().get(move.prevLabel) == 1;
        }

    public:
        MixedSampler(double shift = 1) : m_shift(shift) {}
        virtual ~MixedSampler() {}
        virtual const size_t getAvailableLabelCount() const = 0;

        const double getShift() const { return m_shift; }
    };

    template <typename Label>
    const Label MixedSampler<Label>::sampleNeighborLabel(BaseGraph::VertexIndex vertex) const
    {
        BaseGraph::VertexIndex neighbor = sampleRandomNeighbor((*m_graphPriorPtrPtr)->getState(), vertex);
        Label label = (*m_graphPriorPtrPtr)->getLabel(neighbor);
        return label;
    }

    template <typename Label>
    const Label MixedSampler<Label>::sampleLabelFromNeighborLabel(const Label neighborLabel) const
    {
        return sampleRandomNeighbor((*m_graphPriorPtrPtr)->getLabelGraph(), neighborLabel);
    }

    template <typename Label>
    const LabelMove<Label> MixedSampler<Label>::_proposeLabelMove(const BaseGraph::VertexIndex &vertex) const
    {
        const auto &B = getAvailableLabelCount();

        if ((*m_graphPriorPtrPtr)->getState().getDegree(vertex) == 0)
            return {vertex, (*m_graphPriorPtrPtr)->getLabel(vertex), sampleLabelUniformly()};
        Label neighborLabel = MixedSampler<Label>::sampleNeighborLabel(vertex);
        const auto &Et = (*m_graphPriorPtrPtr)->getLabelGraph().getDegree(neighborLabel);
        double probUniformSampling = m_shift * B / (Et + m_shift * B);
        Label nextLabel;
        if (m_uniform01(rng) < probUniformSampling)
            nextLabel = sampleLabelUniformly();
        else
            nextLabel = sampleLabelFromNeighborLabel(neighborLabel);
        LabelMove<Label> move = {vertex, (*m_graphPriorPtrPtr)->getLabel(vertex), nextLabel};
        return move;
    }

    template <typename Label>
    const double MixedSampler<Label>::_getLogProposalProbForMove(const LabelMove<Label> &move) const
    {
        const auto &labels = (*m_graphPriorPtrPtr)->getLabels();
        const auto &graph = (*m_graphPriorPtrPtr)->getState();
        const auto &labelGraph = (*m_graphPriorPtrPtr)->getLabelGraph();

        double weight = 0, degree = 0;
        for (auto neighbor : graph.getOutNeighbours(move.vertexIndex))
        {
            auto t = labels[neighbor];
            size_t edgeMult = graph.getEdgeMultiplicity(move.vertexIndex, neighbor);
            if (move.vertexIndex == neighbor)
                edgeMult *= 2;
            size_t Est = 0;
            if (move.nextLabel < labelGraph.getSize())
                Est = labelGraph.getEdgeMultiplicity(t, move.nextLabel);

            if (t == move.nextLabel)
                Est *= 2;
            size_t Et = (*m_graphPriorPtrPtr)->getLabelGraph().getDegree(t);
            degree += edgeMult;
            weight += edgeMult * (Est + m_shift) / (Et + m_shift * getAvailableLabelCount());
        }

        if (degree == 0)
            return -log(getAvailableLabelCount());
        double logProposal = log(weight) - log(degree);
        return logProposal;
    }

    template <typename Label>
    const double MixedSampler<Label>::_getLogProposalProbForReverseMove(const LabelMove<Label> &move) const
    {
        const auto &labels = (*m_graphPriorPtrPtr)->getLabels();
        const auto &edgeCounts = (*m_graphPriorPtrPtr)->getEdgeLabelCounts();
        const auto &graph = (*m_graphPriorPtrPtr)->getState();
        const auto &labelGraph = (*m_graphPriorPtrPtr)->getLabelGraph();

        auto edgeMatDiff = getEdgeMatrixDiff(move);
        auto edgeCountsDiff = getEdgeCountsDiff(move);

        double weight = 0, degree = 0;
        for (auto neighbor : graph.getOutNeighbours(move.vertexIndex))
        {
            Label t = labels[neighbor];
            if (move.vertexIndex == neighbor)
                t = move.nextLabel;
            size_t edgeMult = graph.getEdgeMultiplicity(move.vertexIndex, neighbor);
            if (move.vertexIndex == neighbor)
                edgeMult *= 2;
            auto rt = getOrderedEdge({t, move.prevLabel});

            size_t Ert = 0;
            if (t < labelGraph.getSize() and move.prevLabel < labelGraph.getSize())
                Ert = labelGraph.getEdgeMultiplicity(rt.first, rt.second);
            Ert += edgeMatDiff.get(rt);
            if (t == move.prevLabel)
                Ert *= 2;
            size_t Et = 0;
            if (t < labelGraph.getSize())
                Et = labelGraph.getDegree(t);
            Et += edgeCountsDiff.get(t);
            degree += edgeMult;
            weight += edgeMult * (Ert + m_shift) / (Et + m_shift * (getAvailableLabelCount() + move.addedLabels));
        }

        if (degree == 0)
            return -log(getAvailableLabelCount() + move.addedLabels);
        double logProposal = log(weight) - log(degree);
        return logProposal;
    }

    template <typename Label>
    IntMap<std::pair<Label, Label>> MixedSampler<Label>::getEdgeMatrixDiff(const LabelMove<Label> &move) const
    {

        const auto &labels = (*m_graphPriorPtrPtr)->getLabels();
        const auto &graph = (*m_graphPriorPtrPtr)->getState();

        IntMap<std::pair<Label, Label>> edgeMatDiff;
        Label r = move.prevLabel, s = move.nextLabel;
        for (auto neighbor : graph.getOutNeighbours(move.vertexIndex))
        {
            Label t = labels[neighbor];
            const auto mult = graph.getEdgeMultiplicity(move.vertexIndex, neighbor);
            if (move.vertexIndex == neighbor)
                t = move.prevLabel;
            edgeMatDiff.decrement(getOrderedEdge({r, t}), mult);
            if (move.vertexIndex == neighbor)
                t = move.nextLabel;
            edgeMatDiff.increment(getOrderedEdge({s, t}), mult);
        }
        return edgeMatDiff;
    }

    template <typename Label>
    IntMap<Label> MixedSampler<Label>::getEdgeCountsDiff(const LabelMove<Label> &move) const
    {
        IntMap<Label> edgeCountsDiff;
        size_t degree = (*m_graphPriorPtrPtr)->getState().getDegree(move.vertexIndex);
        edgeCountsDiff.decrement(move.prevLabel, degree);
        edgeCountsDiff.increment(move.nextLabel, degree);
        return edgeCountsDiff;
    }

    template <typename Label>
    class GibbsMixedLabelProposer : public GibbsLabelProposer<Label>, public MixedSampler<Label>
    {
    protected:
        const Label sampleLabelUniformly() const override
        {
            return std::uniform_int_distribution<size_t>(0, getAvailableLabelCount() - 1)(rng);
        }
        const double getLogProposalProbForReverseMove(const LabelMove<Label> &move) const override
        {
            return MixedSampler<Label>::_getLogProposalProbForReverseMove(move);
        }
        const double getLogProposalProbForMove(const LabelMove<Label> &move) const override
        {
            return MixedSampler<Label>::_getLogProposalProbForMove(move);
        }

    public:
        GibbsMixedLabelProposer(double sampleLabelCountProb = 0.1, double labelCreationProb = 0.5, double shift = 1) : GibbsLabelProposer<Label>(sampleLabelCountProb, labelCreationProb),
                                                                                                                       MixedSampler<Label>(shift) { this->m_graphPriorPtrPtr = &this->m_graphPriorPtr; }

        const LabelMove<Label> proposeLabelMove(const BaseGraph::VertexIndex &vertex) const override
        {
            auto move = MixedSampler<Label>::_proposeLabelMove(vertex);

            return move;
        }
        const size_t getAvailableLabelCount() const override { return GibbsLabelProposer<Label>::m_graphPriorPtr->getLabelCount(); }
    };

    using GibbsMixedBlockProposer = GibbsMixedLabelProposer<BlockIndex>;

    template <typename Label>
    class RestrictedMixedLabelProposer : public RestrictedLabelProposer<Label>, public MixedSampler<Label>
    {
    protected:
        const Label sampleLabelUniformly() const override { return *sampleUniformlyFrom(m_availableLabels.begin(), m_availableLabels.end()); }
        const double getLogProposalProbForReverseMove(const LabelMove<Label> &move) const override
        {
            return MixedSampler<Label>::_getLogProposalProbForReverseMove(move);
        }
        const double getLogProposalProbForMove(const LabelMove<Label> &move) const override
        {
            double dS = MixedSampler<Label>::_getLogProposalProbForMove(move);
            return dS;
        }

        using RestrictedLabelProposer<Label>::m_availableLabels;
        using RestrictedLabelProposer<Label>::m_emptyLabels;

    public:
        RestrictedMixedLabelProposer(double sampleLabelCountProb = 0.1, double shift = 1) : RestrictedLabelProposer<Label>(sampleLabelCountProb),
                                                                                            MixedSampler<Label>(shift) { this->m_graphPriorPtrPtr = &this->m_graphPriorPtr; }

        const LabelMove<Label> proposeLabelMove(const BaseGraph::VertexIndex &vertex) const override
        {
            auto move = MixedSampler<Label>::_proposeLabelMove(vertex);
            move.addedLabels = -(int)RestrictedLabelProposer<Label>::destroyingLabel(move);
            return move;
        }
        const size_t getAvailableLabelCount() const override { return m_availableLabels.size(); }
    };
    using RestrictedMixedBlockProposer = RestrictedMixedLabelProposer<BlockIndex>;

} // namespace GraphInf

#endif
