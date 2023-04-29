#ifndef GRAPH_INF_PYTHON_EDGECOUNT_H
#define GRAPH_INF_PYTHON_EDGECOUNT_H

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "GraphInf/types.h"
#include "GraphInf/graph/prior/prior.hpp"
#include "GraphInf/graph/prior/python/prior.hpp"
#include "GraphInf/graph/prior/edge_count.h"

namespace GraphInf
{

    template <typename EdgeCountPriorBaseClass = EdgeCountPrior>
    class PyEdgeCountPrior : public PyPrior<size_t, EdgeCountPriorBaseClass>
    {
    public:
        using PyPrior<size_t, EdgeCountPriorBaseClass>::PyPrior;
        ~PyEdgeCountPrior() override = default;

        /* Pure abstract methods */
        const double getLogLikelihoodFromState(const size_t &state) const override { PYBIND11_OVERRIDE_PURE(const double, EdgeCountPriorBaseClass, getLogLikelihoodFromState, state); }
        /* Overloaded abstract methods */
        const double getLogLikelihoodRatioFromGraphMove(const GraphMove &move) const override { PYBIND11_OVERRIDE(const double, EdgeCountPriorBaseClass, getLogLikelihoodRatioFromGraphMove, move); }
    };

}

#endif
