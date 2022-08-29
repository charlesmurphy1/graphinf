#ifndef GRAPH_INF_ACTION_HPP
#define GRAPH_INF_ACTION_HPP

#include "callback.hpp"
#include "GraphInf/mcmc/mcmc.h"

namespace GraphInf{

class CheckConsistencyOnStep: public CallBack<MCMC>{
    using CallBack<MCMC>::m_mcmcPtr;
public:
    void onStepEnd() override { m_mcmcPtr->checkConsistency(); }
};

class CheckSafetyOnStep: public CallBack<MCMC>{
    using CallBack<MCMC>::m_mcmcPtr;
public:
    void onStepEnd() override { m_mcmcPtr->checkSafety(); }
};

class CheckConsistencyOnSweep: public CallBack<MCMC>{
    using CallBack<MCMC>::m_mcmcPtr;
public:
    void onSweepEnd() override { m_mcmcPtr->checkConsistency(); }
};

class CheckSafetyOnSweep: public CallBack<MCMC>{
    using CallBack<MCMC>::m_mcmcPtr;
public:
    void onSweepEnd() override { m_mcmcPtr->checkSafety(); }
};

}

#endif
