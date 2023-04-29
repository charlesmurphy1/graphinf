#ifndef GRAPH_INF_PYTHON_PRIOR_HPP
#define GRAPH_INF_PYTHON_PRIOR_HPP

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "GraphInf/types.h"
#include "GraphInf/python/rv.hpp"
#include "GraphInf/graph/prior/prior.hpp"
#include "GraphInf/graph/proposer/movetypes.h"

namespace GraphInf
{

    template <typename StateType, typename BaseClass = Prior<StateType>>
    class PyPrior : public PyNestedRandomVariable<BaseClass>
    {
    public:
        using PyNestedRandomVariable<BaseClass>::PyNestedRandomVariable;
        using cdouble = const double;
        ~PyPrior() override = default;
        /* Pure abstract methods */
        void sampleState() override { PYBIND11_OVERRIDE_PURE(void, BaseClass, sampleState, ); }
        cdouble getLogLikelihood() const override { PYBIND11_OVERRIDE_PURE(cdouble, BaseClass, getLogLikelihood, ); }
        cdouble getLogLikelihoodRatioFromGraphMove(const GraphMove &move) const override { PYBIND11_OVERRIDE_PURE(cdouble, BaseClass, getLogLikelihoodRatioFromGraphMove, move); }

        /* Abstract methods */
        void setState(const StateType &state) override { PYBIND11_OVERRIDE(void, BaseClass, setState, state); }

    protected:
        void _samplePriors() override { PYBIND11_OVERRIDE_PURE(void, BaseClass, _samplePriors, ); }
        void _applyGraphMove(const GraphMove &move) override { PYBIND11_OVERRIDE_PURE(void, BaseClass, _applyGraphMove, move); }
        cdouble _getLogPrior() const override { PYBIND11_OVERRIDE_PURE(cdouble, BaseClass, _getLogPrior, ); }
        cdouble _getLogPriorRatioFromGraphMove(const GraphMove &move) const override { PYBIND11_OVERRIDE_PURE(cdouble, BaseClass, _getLogPriorRatioFromGraphMove, move); }
    };

    template <typename StateType, typename Label, typename BaseClass = VertexLabeledPrior<StateType, Label>>
    class PyVertexLabeledPrior : public PyPrior<StateType, BaseClass>
    {
    public:
        using PyPrior<StateType, BaseClass>::PyPrior;
        ~PyVertexLabeledPrior() override = default;
        const double getLogLikelihoodRatioFromLabelMove(const LabelMove<Label> &move) const override { PYBIND11_OVERRIDE_PURE(const double, BaseClass, getLogLikelihoodRatioFromLabelMove, move); }

        /* Pure abstract methods */
    protected:
        void _applyLabelMove(const LabelMove<Label> &move) override { PYBIND11_OVERRIDE_PURE(void, BaseClass, _applyLabelMove, move); }
        const double _getLogPriorRatioFromLabelMove(const LabelMove<Label> &move) const override { PYBIND11_OVERRIDE_PURE(const double, BaseClass, _getLogPriorRatioFromLabelMove, move); }

        /* Abstract methods */
    };

}

#endif
