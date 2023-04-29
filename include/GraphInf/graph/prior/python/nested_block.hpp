#ifndef GRAPH_INF_PYTHON_NESTEDBLOCK_H
#define GRAPH_INF_PYTHON_NESTEDBLOCK_H

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>

#include "GraphInf/types.h"
#include "GraphInf/graph/prior/python/block.hpp"
#include "GraphInf/graph/prior/nested_block.h"

namespace GraphInf
{

    template <typename BaseClass = NestedBlockPrior>
    class PyNestedBlockPrior : public PyBlockPrior<BaseClass>
    {
    public:
        using PyBlockPrior<BaseClass>::PyBlockPrior;
        ~PyNestedBlockPrior() override = default;
        /* Pure abstract methods */
        const std::vector<BlockIndex> sampleState(Level level) const override { PYBIND11_OVERRIDE_PURE(const std::vector<BlockIndex>, BaseClass, sampleState, level); }
        const double getLogLikelihoodAtLevel(Level level) const override { PYBIND11_OVERRIDE_PURE(const double, BaseClass, getLogLikelihoodAtLevel, level); }

        /* Overloaded abstract methods */
        void setSize(size_t size) override { PYBIND11_OVERRIDE(void, BaseClass, setSize, size); }
        void createNewBlock(const BlockMove &move) override { PYBIND11_OVERRIDE(void, BaseClass, createNewBlock, move); }
        void destroyBlock(const BlockMove &move) override { PYBIND11_OVERRIDE(void, BaseClass, destroyBlock, move); }
    };

}

#endif
