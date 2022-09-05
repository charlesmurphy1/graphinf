#ifndef GRAPH_INF_PYWRAPPER_PRIOR_INIT_LABELED_DEGREE_H
#define GRAPH_INF_PYWRAPPER_PRIOR_INIT_LABELED_DEGREE_H

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>

#include "GraphInf/random_graph/prior/python/prior.hpp"
#include "GraphInf/random_graph/prior/python/labeled_degree.hpp"
#include "GraphInf/random_graph/prior/edge_count.h"
#include "GraphInf/random_graph/prior/labeled_degree.h"


namespace py = pybind11;
namespace GraphInf{

void initLabeledDegreePrior(py::module& m){
    py::class_<VertexLabeledDegreePrior, BlockLabeledPrior< std::vector<size_t> >, PyVertexLabeledDegreePrior<>>(m, "VertexLabeledDegreePrior")
        .def(py::init<>())
        .def(py::init<LabelGraphPrior&>(), py::arg("label_graph_prior"))
        .def("set_graph", &VertexLabeledDegreePrior::setGraph)
        .def("set_partition", &VertexLabeledDegreePrior::setPartition)
        .def("sample_partition", &VertexLabeledDegreePrior::samplePartition)
        .def("get_degree_of_idx", &VertexLabeledDegreePrior::getDegreeOfIdx)
        .def("get_degree_counts", &VertexLabeledDegreePrior::getDegreeCounts)
        .def("get_block_prior", &VertexLabeledDegreePrior::getBlockPrior)
        .def("set_block_prior", &VertexLabeledDegreePrior::setBlockPrior, py::arg("block_prior"))
        .def("get_label_graph_prior", &VertexLabeledDegreePrior::getLabelGraphPrior, py::return_value_policy::reference_internal)
        .def("set_label_graph_prior", &VertexLabeledDegreePrior::setLabelGraphPrior, py::arg("label_graph_prior"))
        ;

    py::class_<VertexLabeledDegreeDeltaPrior, VertexLabeledDegreePrior>(m, "VertexLabeledDegreeDeltaPrior")
        .def(py::init<const DegreeSequence&>(), py::arg("degrees"))
        ;

    py::class_<VertexLabeledDegreeUniformPrior, VertexLabeledDegreePrior>(m, "VertexLabeledDegreeUniformPrior")
        .def(py::init<>())
        .def(py::init<LabelGraphPrior&>(), py::arg("label_graph_prior"))
        ;

    py::class_<VertexLabeledDegreeUniformHyperPrior, VertexLabeledDegreePrior>(m, "VertexLabeledDegreeUniformHyperPrior")
        .def(py::init<>())
        .def(py::init<LabelGraphPrior&>(), py::arg("label_graph_prior"))
        ;


}

}

#endif
