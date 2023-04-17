#ifndef GRAPH_INF_PYWRAPPER_INIT_UTILITY_FUNCTIONS_H
#define GRAPH_INF_PYWRAPPER_INIT_UTILITY_FUNCTIONS_H

#include <nanobind/nanobind.h>
#include <nanobind/stl/list.h>

#include "GraphInf/utility/functions.h"

namespace nb = nanobind;
namespace GraphInf
{

    void initFunctions(nb::module_ &m)
    {
        nb::module_::import_("basegraph");
        m.def("log_factorial", &logFactorial, nb::arg("n"));
        m.def("log_double_factorial", &logDoubleFactorial, nb::arg("n"));
        m.def("log_binom", &logBinomialCoefficient, nb::arg("n"), nb::arg("k"), nb::arg("force") = true);
        m.def("log_poisson", &logPoissonPMF, nb::arg("k"), nb::arg("mean"));
        m.def("log_truncpoisson", &logZeroTruncatedPoissonPMF, nb::arg("k"), nb::arg("mean"));
        m.def("log_multinom", nb::overload_cast<std::list<size_t>>(&logMultinomialCoefficient), nb::arg("kList"));
        m.def("log_multinom", nb::overload_cast<std::vector<size_t>>(&logMultinomialCoefficient), nb::arg("kVec"));
        m.def("log_multiset", &logMultisetCoefficient, nb::arg("n"), nb::arg("k"));
        m.def("get_edge_list", &getEdgeList, nb::arg("graph"));
        m.def("get_weighted_edge_list", &getWeightedEdgeList, nb::arg("graph"));
        m.def("enumerate_all_graphs", &enumerateAllGraphs, nb::arg("size"), nb::arg("edge_count"), nb::arg("selfloops") = true, nb::arg("parallel_edges") = true);
    }

}

#endif
