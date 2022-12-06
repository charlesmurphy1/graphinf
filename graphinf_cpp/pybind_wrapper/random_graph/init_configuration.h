#ifndef GRAPH_INF_PYWRAPPER_INIT_CONFIGURATION_H
#define GRAPH_INF_PYWRAPPER_INIT_CONFIGURATION_H

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "GraphInf/random_graph/configuration.h"

namespace py = pybind11;
namespace GraphInf{

void initConfiguration(py::module& m){
    py::class_<ConfigurationModelBase, RandomGraph>(m, "ConfigurationModelBase")
        // .def(py::init<size_t>(), py::arg("size"))
        // .def(py::init<size_t, DegreePrior&>(), py::arg("size"), py::arg("degree_prior"))
        .def("get_degree_prior", &ConfigurationModelBase::getDegreePrior, py::return_value_policy::reference_internal)
        .def("set_degree_prior", &ConfigurationModelBase::setDegreePrior, py::arg("prior"))
        ;

    py::class_<ConfigurationModel, ConfigurationModelBase>(m, "ConfigurationModel")
        .def( py::init<std::vector<size_t>>(), py::arg("degrees") )
        ;

    py::class_<ConfigurationModelFamily, ConfigurationModelBase>(m, "ConfigurationModelFamily")
        .def(
            py::init<size_t, double, bool, bool, std::string>(),
            py::arg("size"),
            py::arg("edge_count"),
            py::arg("hyperprior")=true,
            py::arg("canonical")=false,
            py::arg("edge_proposer_type")="degree"
        )
        ;
}

}

#endif
