#ifndef GRAPH_INF_PYWRAPPER_INIT_MCMC_H
#define GRAPH_INF_PYWRAPPER_INIT_MCMC_H

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "init_mcmc.h"
#include "init_callbacks.h"
#include "GraphInf/types.h"

namespace py = pybind11;
namespace GraphInf{

void initMCMC(py::module& m){
    py::module callbacks = m.def_submodule("callbacks");
    initCallBacks(callbacks);

    declareMCMCBaseClass(m);

    declareVertexLabelReconstructionMCMCClass<BlockIndex>(m, "_PartitionReconstructionMCMC");
    declareNestedVertexLabelReconstructionMCMCClass<BlockIndex>(m, "_NestedPartitionReconstructionMCMC");

    declareGraphReconstructionClass<RandomGraph>(m, "_GraphReconstructionMCMC");

    declareGraphReconstructionClass<VertexLabeledRandomGraph<BlockIndex>>(m, "BaseBlockLabeledGraphReconstructionMCMC");
    declareVertexLabeledGraphReconstructionClass<BlockIndex>(m, "_BlockLabeledGraphReconstructionMCMC");

    declareGraphReconstructionClass<NestedVertexLabeledRandomGraph<BlockIndex>>(m, "BaseNestedBlockLabeledGraphReconstructionMCMC");
    declareNestedVertexLabeledGraphReconstructionClass<BlockIndex>(m, "_NestedBlockLabeledGraphReconstructionMCMC");
}

}

#endif
