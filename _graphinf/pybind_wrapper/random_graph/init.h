#ifndef GRAPH_INF_PYWRAPPER_RANDOM_GRAPH_INIT_H
#define GRAPH_INF_PYWRAPPER_RANDOM_GRAPH_INIT_H

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "likelihood/init.h"
#include "prior/init.h"
#include "proposer/init.h"
#include "init_randomgraph.h"
#include "init_erdosrenyi.h"
#include "init_configuration.h"
#include "init_sbm.h"
#include "init_hsbm.h"
#include "init_dcsbm.h"
#include "init_hdcsbm.h"

namespace py = pybind11;
namespace GraphInf{

void initRandomGraph(py::module& m){
    py::module likelihood = m.def_submodule("likelihood");
    initLikelihoods(likelihood);

    py::module prior = m.def_submodule("prior");
    initPriors(prior);

    py::module proposer = m.def_submodule("proposer");
    initPriors(proposer);

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
