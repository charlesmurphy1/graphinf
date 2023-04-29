#ifndef GRAPH_INF_PYTHON_DEGREE_H
#define GRAPH_INF_PYTHON_DEGREE_H

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>

#include "GraphInf/types.h"
#include "GraphInf/utility/maps.hpp"
#include "GraphInf/graph/prior/python/prior.hpp"
#include "GraphInf/graph/prior/degree.h"

namespace GraphInf
{

    template <typename DegreePriorBaseClass = DegreePrior>
    class PyDegreePrior : public PyPrior<std::vector<size_t>, DegreePriorBaseClass>
    {
    public:
        using PyPrior<std::vector<size_t>, DegreePriorBaseClass>::PyPrior;
        /* Pure abstract methods */
        const double getLogLikelihoodRatioFromGraphMove(const GraphMove &move) const override
        {
            PYBIND11_OVERRIDE_PURE(const double, DegreePriorBaseClass, getLogLikelihoodRatioFromGraphMove, move);
        }

        /* Overloaded abstract methods */
        void setState(const DegreeSequence &state) override
        {
            PYBIND11_OVERRIDE(void, DegreePriorBaseClass, setState, state);
        }
        const DegreeCountsMap &getDegreeCounts() const override
        {
            PYBIND11_OVERRIDE(const DegreeCountsMap &, DegreePriorBaseClass, getDegreeCounts, );
        }
        void checkSelfSafety() const override
        {
            PYBIND11_OVERRIDE_PURE(void, DegreePriorBaseClass, checkSelfSafety, );
        }
        void computationFinished() const override
        {
            PYBIND11_OVERRIDE_PURE(void, DegreePriorBaseClass, computationFinished, );
        }
    };

}

#endif
