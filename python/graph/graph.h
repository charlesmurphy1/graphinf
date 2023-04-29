#ifndef GRAPH_INF_PYWRAPPER_RANDOM_GRAPH_INIT_H
#define GRAPH_INF_PYWRAPPER_RANDOM_GRAPH_INIT_H

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "likelihood.h"
#include "prior/prior.h"
#include "proposer/proposer.h"
#include "randomgraph.h"
#include "erdosrenyi.h"
#include "configuration.h"
#include "sbm.h"
#include "hsbm.h"
#include "dcsbm.h"
#include "hdcsbm.h"

namespace py = pybind11;
namespace GraphInf
{

    void initRandomGraph(py::module &m)
    {
        py::module likelihood = m.def_submodule("likelihood");
        initGraphLikelihoods(likelihood);

        py::module prior = m.def_submodule("prior");
        initPriors(prior);

        py::module proposer = m.def_submodule("proposer");
        initProposers(proposer);

        initRandomGraphBaseClass(m);
        initErdosRenyi(m);
        initConfiguration(m);
        initStochasticBlockModel(m);
        initNestedStochasticBlockModel(m);
        initDegreeCorrectedStochasticBlockModel(m);
        initNestedDegreeCorrectedStochasticBlockModel(m);
    }

}

#endif
