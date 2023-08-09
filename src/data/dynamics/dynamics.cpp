#include "GraphInf/data/dynamics/dynamics.h"

namespace GraphInf
{

    void Dynamics::computeConsistentState()
    {
        if (m_state.size() != m_graphPriorPtr->getSize() or m_state.size() == 0)
        {
            sampleState();
            return;
        }
        if (m_state.size() != 0)
            m_neighborsState = computeNeighborsState(m_state);
        if (m_pastStateSequence.size() != 0)
        {
            m_neighborsPastStateSequence = computeNeighborsStateSequence(m_pastStateSequence);
            m_length = m_pastStateSequence[0].size();
        }
    }

    void Dynamics::sampleState(const State &x0, bool asyncMode, size_t initialBurn)
    {
        if (x0.size() == 0)
            m_state = getRandomState();
        else
            m_state = x0;

        m_neighborsState = computeNeighborsState(m_state);

        StateSequence reversedPastState;
        StateSequence reversedFutureState;
        NeighborsStateSequence reversedNeighborsPastState;

        for (size_t t = 0; t < initialBurn; t++)
        {
            if (asyncMode)
            {
                asyncUpdateState(DataModel::getSize());
            }
            else
            {
                syncUpdateState();
            }
        }

        for (size_t t = 0; t < m_length; t++)
        {
            reversedPastState.push_back(m_state);
            reversedNeighborsPastState.push_back(m_neighborsState);
            if (asyncMode)
            {
                asyncUpdateState(DataModel::getSize());
            }
            else
            {
                syncUpdateState();
            }
            reversedFutureState.push_back(m_state);
        }

        const auto N = DataModel::getSize();
        const auto &graph = DataModel::getGraph();
        m_pastStateSequence.clear();
        m_pastStateSequence.resize(N);
        m_futureStateSequence.clear();
        m_futureStateSequence.resize(N);
        m_neighborsPastStateSequence.clear();
        m_neighborsPastStateSequence.resize(N);
        for (const auto &idx : graph)
        {
            m_pastStateSequence[idx].resize(m_length);
            m_futureStateSequence[idx].resize(m_length);
            m_neighborsPastStateSequence[idx].resize(m_length);
            for (size_t t = 0; t < m_length; t++)
            {
                m_pastStateSequence[idx][t] = reversedPastState[t][idx];
                m_futureStateSequence[idx][t] = reversedFutureState[t][idx];
                m_neighborsPastStateSequence[idx][t] = reversedNeighborsPastState[t][idx];
            }
        }

#if DEBUG
        checkSelfConsistency();
#endif
    }

    const State Dynamics::getRandomState() const
    {
        size_t N = DataModel::getSize();
        State rnd_state(N);
        std::uniform_int_distribution<size_t> dist(0, m_numStates - 1);

        for (size_t i = 0; i < N; i++)
            rnd_state[i] = dist(rng);

        return rnd_state;
    };

    const NeighborsState Dynamics::computeNeighborsState(const State &state) const
    {
        const auto N = DataModel::getSize();
        const auto &graph = DataModel::getGraph();
        NeighborsState neighborsState(N);
        for (auto vertex : graph)
        {
            neighborsState[vertex].resize(m_numStates);
            for (auto neighbor : graph.getOutNeighbours(vertex))
            {
                size_t mult = graph.getEdgeMultiplicity(vertex, neighbor);
                if (vertex == neighbor)
                {
                    if (m_acceptSelfLoops)
                        mult *= 2;
                    else
                        continue;
                }

                neighborsState[vertex][state[neighbor]] += mult;
            }
        }
        return neighborsState;
    };

    const NeighborsStateSequence Dynamics::computeNeighborsStateSequence(const StateSequence &stateSequence) const
    {

        const auto N = DataModel::getSize();
        const auto &graph = DataModel::getGraph();
        NeighborsStateSequence neighborsStateSequence(N);
        for (const auto &vertex : graph)
        {
            neighborsStateSequence[vertex].resize(m_length);
            for (size_t t = 0; t < m_length; t++)
            {
                neighborsStateSequence[vertex][t].resize(m_numStates);
                for (const auto &neighbor : graph.getOutNeighbours(vertex))
                {
                    size_t edgeMult = graph.getEdgeMultiplicity(vertex, neighbor);
                    if (vertex == neighbor)
                    {
                        if (m_acceptSelfLoops)
                            edgeMult *= 2;
                        else
                            continue;
                    }
                    neighborsStateSequence[vertex][t][stateSequence[neighbor][t]] += edgeMult;
                }
            }
        }
        return neighborsStateSequence;
    };

    void Dynamics::updateNeighborsStateInPlace(
        BaseGraph::VertexIndex vertex,
        VertexState prevVertexState,
        VertexState newVertexState,
        NeighborsState &neighborsState) const
    {
        const auto &graph = DataModel::getGraph();
        if (prevVertexState == newVertexState)
            return;
        for (auto neighbor : graph.getOutNeighbours(vertex))
        {
            size_t mult = graph.getEdgeMultiplicity(vertex, neighbor);
            if (vertex == neighbor)
            {
                if (m_acceptSelfLoops)
                    mult *= 2;
                else
                    continue;
            }
            neighborsState[neighbor][prevVertexState] -= mult;
            neighborsState[neighbor][newVertexState] += mult;
        }
    };

    void Dynamics::syncUpdateState()
    {
        State futureState(m_state);
        std::vector<double> transProbs(m_numStates);
        const auto &graph = DataModel::getGraph();

        for (const auto idx : graph)
        {
            transProbs = getTransitionProbs(idx);
            futureState[idx] = generateCategorical<double, size_t>(transProbs);
        }
        for (const auto idx : graph)
            updateNeighborsStateInPlace(idx, m_state[idx], futureState[idx], m_neighborsState);
        m_state = futureState;
    };

    void Dynamics::asyncUpdateState(size_t numUpdates)
    {
        size_t N = DataModel::getSize();
        VertexState newVertexState;
        State currentState(m_state);
        std::vector<double> transProbs(m_numStates);
        std::uniform_int_distribution<BaseGraph::VertexIndex> idxGenerator(0, N - 1);

        for (auto i = 0; i < numUpdates; i++)
        {
            BaseGraph::VertexIndex idx = idxGenerator(rng);
            transProbs = getTransitionProbs(currentState[idx], m_neighborsState[idx]);
            newVertexState = generateCategorical<double, size_t>(transProbs);
            updateNeighborsStateInPlace(idx, currentState[idx], newVertexState, m_neighborsState);
            currentState[idx] = newVertexState;
        }
        m_state = currentState;
    };

    const double Dynamics::getLogLikelihood() const
    {
        double logLikelihood = 0;
        std::vector<int> neighborsState(getNumStates(), 0);
        const auto &graph = DataModel::getGraph();
        for (size_t t = 0; t < m_length; t++)
        {
            for (auto idx : graph)
            {
                logLikelihood += log(getTransitionProb(
                    m_pastStateSequence[idx][t],
                    m_futureStateSequence[idx][t],
                    m_neighborsPastStateSequence[idx][t]));
            }
        }
        return logLikelihood;
    };

    const std::vector<double> Dynamics::getTransitionProbs(const VertexState &prevVertexState, const VertexNeighborhoodState &neighborhoodState) const
    {
        std::vector<double> transProbs(getNumStates());
        for (VertexState nextVertexState = 0; nextVertexState < getNumStates(); nextVertexState++)
        {
            transProbs[nextVertexState] = getTransitionProb(prevVertexState, nextVertexState, neighborhoodState);
        }
        return transProbs;
    };
    const std::vector<std::vector<double>> Dynamics::getTransitionMatrix(VertexState outState) const
    {
        std::vector<std::vector<double>> probs;
        for (auto idx : getGraph())
        {
            probs.push_back({});
            for (size_t t = 0; t < m_length; t++)
            {
                VertexState futureState = outState;
                if (outState == -1)
                    futureState = m_futureStateSequence[idx][t];
                probs[idx].push_back(getTransitionProb(
                    m_pastStateSequence[idx][t],
                    futureState,
                    m_neighborsPastStateSequence[idx][t]));
            }
        }
        return probs;
    };

    void Dynamics::updateNeighborsStateFromEdgeMove(
        BaseGraph::Edge edge,
        int counter,
        std::map<BaseGraph::VertexIndex, VertexNeighborhoodStateSequence> &prevNeighborMap,
        std::map<BaseGraph::VertexIndex, VertexNeighborhoodStateSequence> &nextNeighborMap) const
    {
        edge = getOrderedEdge(edge);
        BaseGraph::VertexIndex v = edge.first, u = edge.second;
        if (u == v and not m_acceptSelfLoops)
            return;
        const auto &graph = DataModel::getGraph();

        if (graph.getEdgeMultiplicity(edge.first, edge.second) == 0 and counter < 0)
            throw std::logic_error("Dynamics: Edge (" + std::to_string(edge.first) + ", " + std::to_string(edge.second) + ") " + "with multiplicity 0 cannot be removed.");

        if (prevNeighborMap.count(v) == 0)
        {
            prevNeighborMap.insert({v, m_neighborsPastStateSequence[v]});
            nextNeighborMap.insert({v, m_neighborsPastStateSequence[v]});
        }
        if (prevNeighborMap.count(u) == 0)
        {
            prevNeighborMap.insert({u, m_neighborsPastStateSequence[u]});
            nextNeighborMap.insert({u, m_neighborsPastStateSequence[u]});
        }

        VertexState vState, uState;
        for (size_t t = 0; t < m_length; t++)
        {
            uState = m_pastStateSequence[u][t];
            vState = m_pastStateSequence[v][t];
            nextNeighborMap[u][t][vState] += counter;
            if (u != v)
                nextNeighborMap[v][t][uState] += counter;
        }
    };

    const double Dynamics::getLogLikelihoodRatioFromGraphMove(const GraphMove &move) const
    {
        double logLikelihoodRatio = 0;
        std::set<size_t> verticesAffected;
        std::map<BaseGraph::VertexIndex, VertexNeighborhoodStateSequence> prevNeighborMap, nextNeighborMap;

        for (const auto &edge : move.addedEdges)
        {
            size_t v = edge.first, u = edge.second;
            verticesAffected.insert(v);
            verticesAffected.insert(u);
            updateNeighborsStateFromEdgeMove(edge, 1, prevNeighborMap, nextNeighborMap);
        }
        for (const auto &edge : move.removedEdges)
        {
            size_t v = edge.first, u = edge.second;
            verticesAffected.insert(v);
            verticesAffected.insert(u);
            updateNeighborsStateFromEdgeMove(edge, -1, prevNeighborMap, nextNeighborMap);
        }

        for (const auto &idx : verticesAffected)
        {
            for (size_t t = 0; t < m_length; t++)
            {
                logLikelihoodRatio += log(
                    getTransitionProb(m_pastStateSequence[idx][t], m_futureStateSequence[idx][t], nextNeighborMap[idx][t]));
                logLikelihoodRatio -= log(
                    getTransitionProb(m_pastStateSequence[idx][t], m_futureStateSequence[idx][t], prevNeighborMap[idx][t]));
            }
        }

        return logLikelihoodRatio;
    }

    void Dynamics::applyGraphMoveToSelf(const GraphMove &move)
    {
        std::set<BaseGraph::VertexIndex> verticesAffected;
        std::map<BaseGraph::VertexIndex, VertexNeighborhoodStateSequence> prevNeighborMap, nextNeighborMap;
        VertexNeighborhoodStateSequence neighborsState(m_length);
        size_t v, u;

        for (const auto &edge : move.addedEdges)
        {
            v = edge.first;
            u = edge.second;
            if (u == v and not m_acceptSelfLoops)
                continue;
            verticesAffected.insert(v);
            verticesAffected.insert(u);
            updateNeighborsStateFromEdgeMove(edge, 1, prevNeighborMap, nextNeighborMap);
            m_neighborsState[u][m_state[v]] += 1;
            m_neighborsState[v][m_state[u]] += 1;
        }
        for (const auto &edge : move.removedEdges)
        {
            v = edge.first;
            u = edge.second;
            if (u == v and not m_acceptSelfLoops)
                continue;
            verticesAffected.insert(v);
            verticesAffected.insert(u);
            updateNeighborsStateFromEdgeMove(edge, -1, prevNeighborMap, nextNeighborMap);
            m_neighborsState[u][m_state[v]] -= 1;
            m_neighborsState[v][m_state[u]] -= 1;
        }

        for (const auto &idx : verticesAffected)
        {
            for (size_t t = 0; t < m_length; t++)
            {
                m_neighborsPastStateSequence[idx][t] = nextNeighborMap[idx][t];
            }
        }
    }

    void Dynamics::checkConsistencyOfNeighborsPastStateSequence() const
    {
        const auto N = DataModel::getSize();
        if (m_neighborsPastStateSequence.size() == 0)
            return;
        else if (m_neighborsPastStateSequence.size() != N)
            throw ConsistencyError(
                "Dynamics",
                "graph prior", "size=" + std::to_string(N),
                "m_neighborsPastStateSequence", "size=" + std::to_string(m_neighborsPastStateSequence.size()));
        const auto &actual = m_neighborsPastStateSequence;
        const auto expected = computeNeighborsStateSequence(m_pastStateSequence);
        for (size_t v = 0; v < N; ++v)
        {
            if (actual[v].size() != getLength())
                throw ConsistencyError(
                    "Dynamics",
                    "m_length", "value=" + std::to_string(getLength()),
                    "m_neighborsPastStateSequence", "size=" + std::to_string(actual[v].size()),
                    "vertex=" + std::to_string(v));
            for (size_t t = 0; t < m_length; ++t)
            {
                if (actual[v][t].size() != getNumStates())
                    throw ConsistencyError(
                        "Dynamics",
                        "m_numStates", "value=" + std::to_string(getNumStates()),
                        "m_neighborsPastStateSequence", "size=" + std::to_string(actual[v][t].size()),
                        "vertex=" + std::to_string(v) + ", time=" + std::to_string(t));
                for (size_t s = 0; s < m_numStates; ++s)
                {
                    if (actual[v][t][s] != expected[v][t][s])
                        throw ConsistencyError(
                            "Dynamics",
                            "neighbor counts", "value=" + std::to_string(expected[v][t][s]),
                            "m_neighborsPastStateSequence", "value=" + std::to_string(actual[v][t][s]),
                            "vertex=" + std::to_string(v) + ", time=" + std::to_string(t) + ", state=" + std::to_string(s));
                }
            }
        }
    }

    void Dynamics::checkConsistencyOfNeighborsState() const
    {
        const auto &actual = m_neighborsState;
        const auto expected = computeNeighborsState(m_state);
        const auto N = DataModel::getSize();
        const auto &graph = DataModel::getGraph();
        if (m_neighborsState.size() == 0)
            return;
        else if (actual.size() != N)
            throw ConsistencyError(
                "Dynamics",
                "graph prior", "size=" + std::to_string(N),
                "m_neighborsState", "value=" + std::to_string(actual.size()));
        for (size_t v = 0; v < N; ++v)
        {
            if (actual[v].size() != getNumStates())
                throw ConsistencyError(
                    "Dynamics",
                    "m_numStates", "value=" + std::to_string(getNumStates()),
                    "m_neighborsState", "size=" + std::to_string(actual[v].size()),
                    "vertex=" + std::to_string(v)

                );
            for (size_t s = 0; s < m_numStates; ++s)
            {
                if (actual[v][s] != expected[v][s])
                    throw ConsistencyError(
                        "Dynamics",
                        "neighbor counts", "value=" + std::to_string(expected[v][s]),
                        "m_neighborsState", "value=" + std::to_string(actual[v][s]),
                        "vertex=" + std::to_string(v) + ", state=" + std::to_string(s));
            }
        }
    }

    void Dynamics::checkSelfConsistency() const
    {
        checkConsistencyOfNeighborsPastStateSequence();
        checkConsistencyOfNeighborsState();
    }

    void Dynamics::checkSelfSafety() const
    {
        DataModel::checkSelfSafety();

        if (m_state.size() == 0)
            throw SafetyError("Dynamics", "m_state.size()", "0");
        if (m_pastStateSequence.size() == 0)
            throw SafetyError("Dynamics", "m_pastStateSequence.size()", "0");
        if (m_futureStateSequence.size() == 0)
            throw SafetyError("Dynamics", "m_futureStateSequence.size()", "0");
        if (m_neighborsPastStateSequence.size() == 0)
            throw SafetyError("Dynamics", "m_neighborsPastStateSequence.size()", "0");
    }
}