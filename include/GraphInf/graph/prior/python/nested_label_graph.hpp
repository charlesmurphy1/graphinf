#ifndef GRAPH_INF_PYTHON_NESTEDLABELGRAPH_H
#define GRAPH_INF_PYTHON_NESTEDLABELGRAPH_H

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>

#include "GraphInf/types.h"
#include "GraphInf/graph/prior/python/label_graph.hpp"
#include "GraphInf/graph/prior/nested_label_graph.h"

namespace GraphInf
{

    template <typename BaseClass = NestedLabelGraphPrior>
    class PyNestedLabelGraphPrior : public PyLabelGraphPrior<BaseClass>
    {
    public:
        using PyLabelGraphPrior<BaseClass>::PyLabelGraphPrior;
        /* Pure abstract methods */
        const double getLogLikelihoodRatioFromGraphMove(const GraphMove &move) const override { PYBIND11_OVERRIDE_PURE(const double, BaseClass, getLogLikelihoodRatioFromGraphMove, move); }
        const LabelGraph sampleState(Level level) const override { PYBIND11_OVERRIDE_PURE(const LabelGraph, BaseClass, sampleState, level); }
        const double getLogLikelihoodAtLevel(Level level) const override { PYBIND11_OVERRIDE_PURE(const double, BaseClass, getLogLikelihoodAtLevel, level); }

        /* Overloaded abstract methods */
    };

}

#endif
