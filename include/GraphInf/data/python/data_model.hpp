#ifndef GRAPH_INF_PYTHON_DATAMODEL_HPP
#define GRAPH_INF_PYTHON_DATAMODEL_HPP

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "GraphInf/python/rv.hpp"
#include "GraphInf/data/data_model.h"
#include "GraphInf/data/dynamics/dynamics.h"
#include "GraphInf/data/dynamics/binary_dynamics.h"

namespace GraphInf
{

    template <typename BaseClass = DataModel>
    class PyDataModel : public PyNestedRandomVariable<BaseClass>
    {
    protected:
        void applyGraphMoveToSelf(const GraphMove &move) override
        {
            PYBIND11_OVERRIDE_PURE(void, BaseClass, applyGraphMoveToSelf, move);
        }

    public:
        using PyNestedRandomVariable<BaseClass>::PyNestedRandomVariable;
        /* Pure abstract methods */
        const double getLogLikelihood() const override
        {
            PYBIND11_OVERRIDE_PURE(const double, BaseClass, getLogLikelihood, );
        }
        const double getLogLikelihoodRatioFromGraphMove(const GraphMove &move) const override
        {
            PYBIND11_OVERRIDE_PURE(const double, BaseClass, getLogLikelihoodRatioFromGraphMove, move);
        }
        /* Abstract methods */
        void computeConsistentState() override { PYBIND11_OVERRIDE(void, BaseClass, computeConsistentState, ); }

        bool isSafe() const override { PYBIND11_OVERRIDE(bool, BaseClass, isSafe, ); }
    };

    template <typename BaseClass = Dynamics>
    class PyDynamics : public PyDataModel<BaseClass>
    {
    public:
        using PyDataModel<BaseClass>::PyDataModel;

        /* Pure abstract methods */
        const double getTransitionProb(
            const VertexState &prevVertexState, const VertexState &nextVertexState, const VertexNeighborhoodState &neighborhoodState) const
        {
            PYBIND11_OVERRIDE_PURE(const double, BaseClass, getTransitionProb, prevVertexState, nextVertexState, neighborhoodState);
        }
        /* Abstract methods */
        const State getRandomState() const { PYBIND11_OVERRIDE(const State, BaseClass, getRandomState, ); }
    };

    template <typename BaseClass = BinaryDynamics>
    class PyBinaryDynamics : public PyDynamics<BaseClass>
    {
    public:
        using PyDynamics<BaseClass>::PyDynamics;
        /* Pure abstract methods */
        const double getActivationProb(const VertexNeighborhoodState &neighborState) const override
        {
            PYBIND11_OVERRIDE_PURE(const double, BaseClass, getActivationProb, neighborState);
        }
        const double getDeactivationProb(const VertexNeighborhoodState &neighborState) const override
        {
            PYBIND11_OVERRIDE_PURE(const double, BaseClass, getDeactivationProb, neighborState);
        }
        /* Abstract methods */
    };

}

#endif
