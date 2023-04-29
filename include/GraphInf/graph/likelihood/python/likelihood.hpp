#ifndef GRAPH_INF_PYTHON_LIKELIHOOD_HPP
#define GRAPH_INF_PYTHON_LIKELIHOOD_HPP

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "GraphInf/types.h"
#include "GraphInf/python/rv.hpp"
#include "GraphInf/graph/likelihood/likelihood.hpp"
#include "GraphInf/graph/proposer/movetypes.h"

namespace GraphInf
{

    template <typename BaseClass = GraphLikelihoodModel>
    class PyGraphLikelihoodModel : public PyNestedRandomVariable<BaseClass>
    {
    public:
        using PyNestedRandomVariable<BaseClass>::PyNestedRandomVariable;
        /* Pure abstract methods */
        const double getLogLikelihood() const override { PYBIND11_OVERRIDE_PURE(const double, BaseClass, getLogLikelihood, ); }
        const double getLogLikelihoodRatioFromGraphMove(const GraphMove &move) const override { PYBIND11_OVERRIDE_PURE(const double, BaseClass, getLogLikelihoodRatioFromGraphMove, move); }
        const MultiGraph sample() const override { PYBIND11_OVERRIDE(const MultiGraph, BaseClass, sample, ); }

        /* Abstract methods */
    };

    template <typename Label, typename BaseClass = VertexLabeledGraphLikelihoodModel<Label>>
    class PyVertexLabeledGraphLikelihoodModel : public PyGraphLikelihoodModel<BaseClass>
    {
    public:
        using PyGraphLikelihoodModel<BaseClass>::PyGraphLikelihoodModel;

        /* Pure abstract methods */
        const double getLogLikelihoodRatioFromLabelMove(const LabelMove<Label> &move) const override { PYBIND11_OVERRIDE_PURE(const double, BaseClass, getLogLikelihoodRatioFromLabelMove, move); }

        /* Abstract methods */
    };

}

#endif
