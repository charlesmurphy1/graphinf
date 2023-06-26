#ifndef GRAPH_INF_PYTHON_BLOCKCOUNT_H
#define GRAPH_INF_PYTHON_BLOCKCOUNT_H

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "GraphInf/types.h"
#include "GraphInf/graph/prior/prior.hpp"
#include "GraphInf/graph/prior/python/prior.hpp"
#include "GraphInf/graph/prior/block_count.h"

namespace GraphInf
{

    template <typename BaseClass = BlockCountPrior>
    class PyBlockCountPrior : public PyVertexLabeledPrior<size_t, BlockIndex, BaseClass>
    {
    public:
        using PyVertexLabeledPrior<size_t, BlockIndex, BaseClass>::PyVertexLabeledPrior;
        /* Pure abstract methods */
        const double getLogLikelihoodFromState(const size_t &state) const override { PYBIND11_OVERRIDE_PURE(const double, BaseClass, getLogLikelihoodFromState, state); }
        void setMaxBlockCount(size_t maxBlockCount) override { PYBIND11_OVERRIDE(void, BaseClass, setMaxBlockCount, maxBlockCount); }
    };

    template <typename BaseClass = NestedBlockCountPrior>
    class PyNestedBlockCountPrior : public PyBlockCountPrior<BaseClass>
    {
    public:
        using PyBlockCountPrior<BaseClass>::PyBlockCountPrior;
        /* Pure abstract methods */
        const double getLogLikelihoodFromNestedState(const std::vector<size_t> &state) const override { PYBIND11_OVERRIDE_PURE(const double, BaseClass, getLogLikelihoodFromState, state); }
    };

}

#endif
