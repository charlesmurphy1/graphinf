#ifndef GRAPH_INF_PYWRAPPER_INIT_PROPOSER_MOVETYPES_H
#define GRAPH_INF_PYWRAPPER_INIT_PROPOSER_MOVETYPES_H

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "BaseGraph/types.h"
#include "GraphInf/mcmc.h"

namespace py = pybind11;
namespace GraphInf
{

    template <typename Label>
    py::class_<LabelMove<Label>> declareLabelMove(py::module &m, std::string pyName)
    {
        return py::class_<LabelMove<Label>>(m, pyName.c_str())
            .def(py::init<BaseGraph::VertexIndex, Label, Label, int, int>(),
                 py::arg("vertex_index"), py::arg("prev_label"), py::arg("next_label"), py::arg("added_labels") = 0, py::arg("level") = 0)
            .def_readonly("vertex_id", &BlockMove::vertexIndex)
            .def_readonly("prev_label", &BlockMove::prevLabel)
            .def_readonly("next_label", &BlockMove::nextLabel)
            .def_readonly("added_labels", &BlockMove::addedLabels)
            .def("__repr__", [](const LabelMove<Label> &self)
                 { return self.display(); });
    }

    void initMoveTypes(py::module &m)
    {
        py::class_<GraphMove>(m, "GraphMove")
            .def(py::init<std::vector<BaseGraph::Edge>, std::vector<BaseGraph::Edge>>(),
                 py::arg("removed_edges"), py::arg("added_edges"))
            .def_readonly("removed_edges", &GraphMove::removedEdges)
            .def_readonly("added_edges", &GraphMove::addedEdges)
            .def("__repr__", [](const GraphMove &self)
                 { return self.display(); });

        py::class_<ParamMove>(m, "ParamMove")
            .def(py::init<std::string, double>(),
                 py::arg("key") = "none", py::arg("value") = 0.0)
            .def_readonly("key", &ParamMove::key)
            .def_readonly("value", &ParamMove::value)
            .def("__repr__", [](const ParamMove &self)
                 { return self.display(); });

        declareLabelMove<BlockIndex>(m, "BlockMove");
    }

    template <typename MoveType>
    py::class_<StepResult<MoveType>> declareStepResult(py::module &m, std::string pyName)
    {
        return py::class_<StepResult<MoveType>>(m, pyName.c_str())
            .def(py::init<MoveType, double, bool>(),
                 py::arg("move"), py::arg("log_joint_ratio"), py::arg("accepted"))
            .def_readonly("move", &StepResult<MoveType>::move)
            .def_readonly("log_joint_ratio", &StepResult<MoveType>::logJointRatio)
            .def_readonly("accepted", &StepResult<MoveType>::accepted);
    }

    void initStepResult(py::module &m)
    {
        declareStepResult<GraphMove>(m, "GraphStepResult");
        declareStepResult<ParamMove>(m, "ParamStepResult");
        declareStepResult<BlockMove>(m, "BlockStepResult");
    }

    void initMCMCSummary(py::module &m)
    {
        py::class_<MCMCSummary>(m, "MCMCSummary")
            .def(py::init<double>(), py::arg("log_joint_ratio") = 0.0)
            .def("update", &MCMCSummary::update<GraphMove>, py::arg("step_summary"))
            .def("update", &MCMCSummary::update<ParamMove>, py::arg("step_summary"))
            .def("update", &MCMCSummary::update<BlockMove>, py::arg("step_summary"))
            .def("join", &MCMCSummary::join, py::arg("other"))
            .def_readonly("log_joint_ratio", &MCMCSummary::logJointRatio)
            .def_readonly("accepted", &MCMCSummary::accepted)
            .def_readonly("total", &MCMCSummary::total)
            .def("__repr__", [](const MCMCSummary &self)
                 { return self.display(); });
    }
}
#endif
