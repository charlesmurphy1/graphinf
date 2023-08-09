#ifndef GRAPH_INF_PYWRAPPER_LIKELIHOOD_INIT_H
#define GRAPH_INF_PYWRAPPER_LIKELIHOOD_INIT_H

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "GraphInf/graph/likelihood/python/likelihood.hpp"
#include "GraphInf/graph/likelihood/delta.h"
#include "GraphInf/graph/likelihood/erdosrenyi.h"
#include "GraphInf/graph/likelihood/configuration.h"
#include "GraphInf/graph/likelihood/sbm.h"
#include "GraphInf/graph/likelihood/dcsbm.h"

namespace py = pybind11;
namespace GraphInf
{
    template <typename Label>
    py::class_<VertexLabeledGraphLikelihoodModel<Label>, GraphLikelihoodModel, PyVertexLabeledGraphLikelihoodModel<Label>> declareVertexLabeledGraphLikelihoodBaseClass(py::module &m, std::string pyName)
    {
        return py::class_<VertexLabeledGraphLikelihoodModel<Label>, GraphLikelihoodModel, PyVertexLabeledGraphLikelihoodModel<Label>>(m, pyName.c_str())
            .def("log_likelihood_ratio_from_label_move", &VertexLabeledGraphLikelihoodModel<Label>::getLogLikelihoodRatioFromLabelMove, py::arg("move"));
    }

    void initGraphLikelihoods(py::module &m)
    {
        py::class_<GraphLikelihoodModel, NestedRandomVariable, PyGraphLikelihoodModel<>>(m, "GraphLikelihoodModel")
            .def("log_likelihood", &GraphLikelihoodModel::getLogLikelihood)
            .def("log_likelihood_ratio_from_graph_move", &GraphLikelihoodModel::getLogLikelihoodRatioFromGraphMove, py::arg("move"))
            .def("sample", &GraphLikelihoodModel::sample);

        declareVertexLabeledGraphLikelihoodBaseClass<BlockIndex>(m, "BlockLabeledGraphLikelihoodBaseClass");
        py::class_<ErdosRenyiLikelihood, GraphLikelihoodModel>(m, "ErdosRenyiLikelihood");
        py::class_<DeltaGraphLikelihood, GraphLikelihoodModel>(m, "DeltaGraphLikelihood");
        py::class_<ConfigurationModelLikelihood, GraphLikelihoodModel>(m, "ConfigurationModelLikelihood");
        py::class_<StochasticBlockModelLikelihood, VertexLabeledGraphLikelihoodModel<BlockIndex>>(m, "StochasticBlockModelLikelihood");
        py::class_<StubLabeledStochasticBlockModelLikelihood, StochasticBlockModelLikelihood>(m, "StubLabeledStochasticBlockModelLikelihood");
        py::class_<UniformStochasticBlockModelLikelihood, StochasticBlockModelLikelihood>(m, "UniformStochasticBlockModelLikelihood");
        py::class_<DegreeCorrectedStochasticBlockModelLikelihood, VertexLabeledGraphLikelihoodModel<BlockIndex>>(m, "DegreeCorrectedStochasticBlockModelLikelihood");
    }

}

#endif
