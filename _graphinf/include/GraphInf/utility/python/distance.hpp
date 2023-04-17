#ifndef GRAPH_INF_PYTHON_DISTANCE_HPP
#define GRAPH_INF_PYTHON_DISTANCE_HPP

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "GraphInf/types.h"
#include "GraphInf/utility/distance.h"

namespace GraphInf{

/* CallBack  base class */
template<typename BaseClass = GraphDistance>
class PyGraphDistance: public BaseClass{
public:
    using BaseClass::BaseClass;
    /* Pure abstract methods */
    double compute(const MultiGraph& g1, const MultiGraph& g2) const override { PYBIND11_OVERRIDE(double, BaseClass, compute, g1, g2); }

    /* Abstract methods */
};

}

#endif
