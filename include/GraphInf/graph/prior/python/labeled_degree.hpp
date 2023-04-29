#ifndef GRAPH_INF_PYTHON_LABELED_DEGREE_H
#define GRAPH_INF_PYTHON_LABELED_DEGREE_H

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>

#include "GraphInf/types.h"
#include "GraphInf/utility/maps.hpp"
#include "GraphInf/graph/prior/python/prior.hpp"
#include "GraphInf/graph/prior/labeled_degree.h"

namespace GraphInf
{

    template <typename BaseClass = VertexLabeledDegreePrior>
    class PyVertexLabeledDegreePrior : public PyVertexLabeledPrior<std::vector<size_t>, BlockIndex, BaseClass>
    {
    public:
        using PyVertexLabeledPrior<std::vector<size_t>, BlockIndex, BaseClass>::PyVertexLabeledPrior;
        /* Pure abstract methods */
        const double getLogLikelihoodRatioFromGraphMove(const GraphMove &move) const override
        {
            PYBIND11_OVERRIDE_PURE(const double, BaseClass, getLogLikelihoodRatioFromGraphMove, move);
        }
        const double getLogLikelihoodRatioFromLabelMove(const BlockMove &move) const override
        {
            PYBIND11_OVERRIDE_PURE(const double, BaseClass, getLogLikelihoodRatioFromLabelMove, move);
        }

        /* Overloaded abstract methods */
        void setState(const DegreeSequence &state) override
        {
            PYBIND11_OVERRIDE(void, BaseClass, setState, state);
        }
        const VertexLabeledDegreeCountsMap &getDegreeCounts() const override
        {
            PYBIND11_OVERRIDE(const VertexLabeledDegreeCountsMap &, BaseClass, getDegreeCounts, );
        }
        void checkSelfSafety() const override
        {
            PYBIND11_OVERRIDE_PURE(void, BaseClass, checkSelfSafety, );
        }
        void computationFinished() const override
        {
            PYBIND11_OVERRIDE_PURE(void, BaseClass, computationFinished, );
        }
    };

}

#endif
