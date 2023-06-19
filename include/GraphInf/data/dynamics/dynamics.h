#ifndef GRAPH_INF_DYNAMICS_H
#define GRAPH_INF_DYNAMICS_H

#include <vector>
#include <map>
#include <memory>
#include <iostream>

#include "BaseGraph/types.h"

#include "GraphInf/types.h"
#include "GraphInf/exceptions.h"
#include "GraphInf/graph/random_graph.hpp"
#include "GraphInf/utility/functions.h"
#include "GraphInf/rng.h"
#include "GraphInf/generators.h"

#include "GraphInf/data/data_model.h"
#include "GraphInf/data/types.h"

namespace GraphInf
{

    class Dynamics : public DataModel
    {
    protected:
        size_t m_numStates;
        size_t m_length;
        std::vector<VertexState> m_state;
        Matrix<VertexState> m_neighborsState;
        bool m_acceptSelfLoops = false;
        Matrix<VertexState> m_pastStateSequence;
        Matrix<VertexState> m_futureStateSequence;
        Matrix<std::vector<VertexState>> m_neighborsPastStateSequence;

        void updateNeighborsStateInPlace(
            BaseGraph::VertexIndex vertexIdx,
            VertexState prevVertexState,
            VertexState newVertexState,
            NeighborsState &neighborsState) const;
        void updateNeighborsStateFromEdgeMove(
            BaseGraph::Edge,
            int direction,
            std::map<BaseGraph::VertexIndex, VertexNeighborhoodStateSequence> &,
            std::map<BaseGraph::VertexIndex, VertexNeighborhoodStateSequence> &) const;

        void checkConsistencyOfNeighborsState() const;
        void checkConsistencyOfNeighborsPastStateSequence() const;
        void computeConsistentState() override;

    public:
        explicit Dynamics(RandomGraph &graphPrior, size_t numStates, size_t length) : DataModel(graphPrior),
                                                                                      m_numStates(numStates),
                                                                                      m_length(length) {}

        const std::vector<VertexState> &getState() const { return m_state; }
        void setCurrentState(std::vector<VertexState> &state)
        {
            m_state = state;
            computeConsistentState();
#if DEBUG
            checkSelfConsistency();
#endif
        }
        void setState(Matrix<VertexState> states)
        {
            m_pastStateSequence.clear();
            m_futureStateSequence.clear();

            for (size_t t = 0; t < states.size() - 1; t++)
            {
                m_pastStateSequence.push_back(states[t]);
                m_futureStateSequence.push_back(states[t + 1]);
            }
            computeConsistentState();
        }
        void setState(Matrix<VertexState> past, Matrix<VertexState> future)
        {
            m_pastStateSequence = past;
            m_futureStateSequence = future;
            computeConsistentState();
        }
        bool acceptSelfLoops() { return m_acceptSelfLoops; }
        void acceptSelfLoops(bool condition) { m_acceptSelfLoops = condition; }
        const Matrix<VertexState> &getNeighborsState() const { return m_neighborsState; }
        const Matrix<VertexState> &getPastStates() const { return m_pastStateSequence; }
        const Matrix<VertexState> &getFutureStates() const { return m_futureStateSequence; }
        const Matrix<std::vector<VertexState>> &getNeighborsPastStates() const { return m_neighborsPastStateSequence; }
        const size_t getNumStates() const { return m_numStates; }
        const size_t getLength() const { return m_length; }
        void setLength(size_t length) { m_length = length; }

        void sampleState(const std::vector<VertexState> &initialState = {}, bool asyncMode = false, size_t initialBurn = 0);
        void sample(const std::vector<VertexState> &initialState = {}, bool asyncMode = false, size_t initialBurn = 0)
        {
            DataModel::m_graphPriorPtr->sample();
            sampleState(initialState, asyncMode, initialBurn);
            DataModel::computationFinished();
#if DEBUG
            checkSelfConsistency();
#endif
        }
        virtual const State getRandomState() const;
        const NeighborsState computeNeighborsState(const State &state) const;
        const NeighborsStateSequence computeNeighborsStateSequence(const StateSequence &stateSequence) const;

        void syncUpdateState();
        void asyncUpdateState(size_t num_updates);

        const double getLogLikelihood() const override;
        virtual const double getTransitionProb(
            const VertexState &prevVertexState, const VertexState &nextVertexState, const VertexNeighborhoodState &neighborhoodState) const = 0;

        const std::vector<double> getTransitionProbs(
            const VertexState &prevVertexState,
            const VertexNeighborhoodState &neighborhoodState) const;
        const std::vector<double> getTransitionProbs(const BaseGraph::VertexIndex vertex) const
        {
            return getTransitionProbs(m_state[vertex], m_neighborsState[vertex]);
        }
        const std::vector<std::vector<double>> getTransitionMatrix(VertexState outState = -1) const;

        const double getLogLikelihoodRatioFromGraphMove(const GraphMove &move) const override;
        void applyGraphMoveToSelf(const GraphMove &move) override;
        virtual bool isValidParamMove(const ParamMove &move) const override
        {
            return DataModel::isValidParamMove(move);
        }
        void checkSelfSafety() const override;
        void checkSelfConsistency() const override;

        bool isSafe() const override
        {
            return DataModel::isSafe() and (m_state.size() != 0) and (m_pastStateSequence.size() != 0) and (m_futureStateSequence.size() != 0) and (m_neighborsPastStateSequence.size() != 0);
        }
    };

} // namespace GraphInf

#endif
