#ifndef GRAPH_INF_PYTHON_MCMC_HPP
#define GRAPH_INF_PYTHON_MCMC_HPP

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "GraphInf/types.h"
#include "GraphInf/python/rv.hpp"
#include "GraphInf/mcmc/mcmc.h"

namespace GraphInf{

template<typename BaseClass = MCMC>
class PyMCMC: public PyNestedRandomVariable<BaseClass>{
public:
    using PyNestedRandomVariable<BaseClass>::PyNestedRandomVariable;
    /* Pure abstract methods */
    const double getLogLikelihood() const override {
        PYBIND11_OVERRIDE_PURE(const double, BaseClass, getLogLikelihood, );
    }
    const double getLogPrior() const override {
        PYBIND11_OVERRIDE_PURE(const double, BaseClass, getLogPrior, );
    }
    const double getLogJoint() const override {
        PYBIND11_OVERRIDE_PURE(const double, BaseClass, getLogJoint, );
    }
    bool doMetropolisHastingsStep() override {
        PYBIND11_OVERRIDE_PURE(bool, BaseClass, doMetropolisHastingsStep, );
    }

    /* Abstract methods */
    void removeCallBack(std::string key) override {
        PYBIND11_OVERRIDE(void, BaseClass, removeCallBack, key);
    }
    void reset() override { PYBIND11_OVERRIDE(void, BaseClass, reset, ); }
    void onBegin() override { PYBIND11_OVERRIDE(void, BaseClass, onBegin, ); }
    void onEnd() override { PYBIND11_OVERRIDE(void, BaseClass, onEnd, ); }
    void onStepBegin() override { PYBIND11_OVERRIDE(void, BaseClass, onStepBegin, ); }
    void onStepEnd() override { PYBIND11_OVERRIDE(void, BaseClass, onStepEnd, ); }
    void onSweepBegin() override { PYBIND11_OVERRIDE(void, BaseClass, onSweepBegin, ); }
    void onSweepEnd() override { PYBIND11_OVERRIDE(void, BaseClass, onSweepEnd, ); }


};



}

#endif