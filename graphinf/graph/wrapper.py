import logging
from typing import Callable, Optional, List
import numpy as np

from basegraph import core as bg
from graphinf._graphinf import graph as _graph
from graphinf.wrapper import Wrapper as _Wrapper
from graphinf.utility import convert_basegraph_to_graphtool, sampleRandomWeakComposition
from functools import partial

from .degree_sequences import nbinom_degreeseq, poisson_degreeseq
from .util import (
    reduce_partition,
    log_evidence_annealed,
    log_evidence_exact,
    log_evidence_iid_meanfield,
    log_evidence_partition_meanfield,
)

__all__ = (
    "LiteralError",
    "RandomGraphWrapper",
    "ErdosRenyiModel",
    "ConfigurationModel",
    "ConfigurationModelFamily",
    "PoissonGraph",
    "NegativeBinomialGraph",
    "StochasticBlockModelFamily",
)


class LiteralError(ValueError):
    def __init__(self, value: str, avail_options: List[str] = None) -> None:
        super().__init__(f"Literal {value} is invalid, possible values are {avail_options}.")


class RandomGraphWrapper(_Wrapper):
    def __init__(self, model, labeled=False, nested=False, **kwargs):
        self.labeled = labeled
        self.nested = nested
        self._state = model.state()
        super().__init__(model, params=kwargs)

    def __repr__(self):
        str_format = f"{self.__class__.__name__}("
        if len(self.params) == 0:
            return str_format + ")"
        for k, v in self.params.items():
            if isinstance(v, str):
                v = f"'{v}'"
            if k in dir(self):
                v = getattr(self, k)
            if isinstance(v, Callable):
                v = v()
            str_format += f"\n\t{k}={v},"
        str_format += "\n)"
        return str_format

    def from_graph(self, graph: bg.UndirectedGraph):
        if self.constructor is None:
            return
        args = self.format_graph_into_args(graph)
        self.__wrapped__ = self.constructor(**args)
        self.set_state(graph)
        self.__others__["hidden_state"] = graph
        self.__buildmethods__()
        for k, v in args.items():
            if k in self.__others__["params"]:
                self.__others__["params"][k] = v

    def format_graph_into_args(self, graph: bg.UndirectedGraph):
        return dict()

    # @property
    # def size(self):
    #     return self.get_size()

    # @property
    # def edge_count(self):
    #     return self.get_edge_count()

    def set_state(self, state):
        self._state = state
        self.wrap.set_state(state)

    def post_init(self):
        self.wrap.sample()

    def joint_entropy(self, n_samples: int = 25, reset_original: bool = False):
        entropy = np.zeros(n_samples)
        state = self.state()
        for i in range(n_samples):
            self.sample()
            entropy[i] = -self.log_joint()
        if reset_original:
            self.set_state(state)
        return entropy.mean()

    def state_entropy(self, n_samples: int = 25, reset_original: bool = False, **kwargs):
        state = self.state()
        entropy = np.zeros(n_samples)
        for i in range(n_samples):
            self.sample_prior()
            entropy[i] = self.log_evidence(**kwargs)
        if reset_original:
            self.set_state(state)
        return entropy.mean()

    def log_evidence(
        self,
        graph: Optional[bg.UndirectedMultigraph] = None,
        method: Optional[str] = None,
        n_sweeps: int = 1000,
        n_steps_per_vertex: int = 10,
        burn_sweeps: int = 5,
        start_from_original: bool = False,
        reset_original: bool = False,
        **kwargs,
    ) -> None:
        all_methods = [
            "exact",
            "iid_meanfield",
            "partition_meanfield",
            "annealed",
        ]
        if method is None:
            method = "exact" if (self.size() <= 5 or not self.labeled) else "iid_meanfield"
        if method not in all_methods:
            raise ValueError(f"Cannot parse method '{method}', available options are {all_methods}.")
        if graph is None:
            graph = self.state()
        kwargs["n_sweeps"] = n_sweeps
        kwargs["n_steps_per_vertex"] = n_steps_per_vertex
        kwargs["burn_sweeps"] = burn_sweeps
        kwargs["start_from_original"] = start_from_original
        kwargs["reset_original"] = reset_original
        if not self.labeled:
            if method == "iid_meanfield":
                evidence = log_evidence_iid_meanfield(self, graph, **kwargs)
            else:
                evidence = self.log_joint()
            self.set_state(graph)
            return evidence

        original = self.labels()
        if method == "exact":
            evidence = log_evidence_exact(self, graph)
        elif method == "iid_meanfield":
            evidence = log_evidence_iid_meanfield(self, graph, **kwargs)
        elif method == "partition_meanfield":
            evidence = log_evidence_partition_meanfield(self, graph, **kwargs)
        elif method == "annealed":
            evidence = log_evidence_annealed(self, graph, **kwargs)
        self.set_state(graph)
        self.set_labels(original)
        return evidence


class DeltaGraph(RandomGraphWrapper):
    def __init__(self, graph: bg.UndirectedMultigraph):
        wrapped = _graph.DeltaGraph(graph)
        super().__init__(wrapped)


class ErdosRenyiModel(RandomGraphWrapper):
    def __init__(
        self,
        size: int = 100,
        edge_count: float = 250,
        canonical: bool = False,
        loopy: bool = True,
        multigraph: bool = True,
    ):
        self.constructor = partial(
            _graph.ErdosRenyiModel,
            canonical=canonical,
        )
        wrapped = self.constructor(
            size,
            edge_count,
            with_self_loops=loopy,
            with_parallel_edges=multigraph,
        )

        super().__init__(
            wrapped,
            size=size,
            edge_count=edge_count,
            canonical=canonical,
            loopy=loopy,
            multigraph=multigraph,
        )

    def format_graph_into_args(self, graph: bg.UndirectedGraph):
        return dict(size=graph.get_size(), edge_count=graph.get_total_edge_number())


class ConfigurationModelFamily(RandomGraphWrapper):
    available_degree_prior_types = ["uniform", "hyper"]

    def __init__(
        self,
        size: int = 100,
        edge_count: float = 250,
        degree_prior_type: str = "uniform",
        canonical: bool = False,
    ):
        if degree_prior_type not in self.available_degree_prior_types:
            raise LiteralError(degree_prior_type, self.available_degree_prior_types)
        self.constructor = _graph.ConfigurationModelFamily
        wrapped = self.constructor(
            size,
            edge_count,
            canonical=canonical,
            # hyperprior=(degree_prior_type == "hyper"),
        )

        super().__init__(
            wrapped,
            size=size,
            edge_count=edge_count,
            degree_prior_type=degree_prior_type,
            canonical=canonical,
        )

    def format_graph_into_args(self, graph: bg.UndirectedGraph):
        return dict(size=graph.get_size(), edge_count=graph.get_total_edge_number())


class ConfigurationModel(RandomGraphWrapper):
    def __init__(
        self,
        degrees: list[int] | None = None,
        size: int = 100,
        edge_count: float = 250,
    ):
        if degrees is None:
            assert (
                size is not None and edge_count is not None
            ), f"`size` ({size}) and `edge_count` ({edge_count}) must be provided if degrees is `None`."
            degrees = sampleRandomWeakComposition(2 * edge_count, size)
        self.constructor = _graph.ConfigurationModel
        wrapped = self.constructor(degrees)
        size = len(degrees)
        edge_count = int(sum(degrees) / 2)

        super().__init__(
            wrapped,
            size=size,
            edge_count=edge_count,
        )

    def format_graph_into_args(self, graph: bg.UndirectedGraph):
        return dict(size=graph.get_size(), edge_count=graph.get_total_edge_number())


class PoissonGraph(ConfigurationModel):
    def __init__(self, size: int, edge_count: int):
        avgk = 2 * edge_count / size
        super().__init__(poisson_degreeseq(size, avgk))


class NegativeBinomialGraph(ConfigurationModel):
    def __init__(self, size: int, edge_count: int, heterogeneity: float = 0):
        avgk = 2 * edge_count / size
        super().__init__(nbinom_degreeseq(size, avgk, heterogeneity))


# class PlantedPartitionGraph(RandomGraphWrapper):
#     def __init__(
#         self,
#         size: int = 100,
#         edge_count: int = 250,
#         block_count: int = 3,
#         assortativity: float = 0.5,
#         stub_labeled: bool = False,
#         loopy: bool = True,
#         multigraph: bool = True,
#     ):
#         wrapped = _graph.PlantedPartitionModel(
#             size,
#             edge_count,
#             block_count=block_count,
#             assortativity=assortativity,
#             stub_labeled=stub_labeled,
#             with_self_loops=loopy,
#             with_parallel_edges=multigraph,
#         )
#         super().__init__(
#             wrapped,
#             size=size,
#             edge_count=edge_count,
#             block_count=block_count,
#             assortativity=assortativity,
#             stub_labeled=stub_labeled,
#             loopy=loopy,
#             multigraph=multigraph,
#         )


class StochasticBlockModelFamily(RandomGraphWrapper):
    available_likelihood_types = [
        "uniform",
        "stub_labeled",
        "degree_corrected",
    ]
    available_block_prior_types = ["uniform", "hyper"]
    available_label_graph_prior_types = ["uniform", "planted", "nested"]
    available_degree_prior_types = ["uniform", "hyper"]

    def __init__(
        self,
        size: int = 100,
        edge_count: float = 250,
        block_count: Optional[int] = None,
        likelihood_type: str = "uniform",
        label_graph_prior_type: str = "uniform",
        degree_prior_type: str = "uniform",
        canonical: bool = False,
        block_proposer_type: str = "uniform",
        shift: float = 1,
        sample_label_count_prob: float = 0.1,
    ):
        labeled = True
        nested = False
        block_count = 0 if block_count is None else block_count
        if likelihood_type not in self.available_likelihood_types:
            raise LiteralError(likelihood_type, self.available_likelihood_types)
        if label_graph_prior_type not in self.available_label_graph_prior_types:
            raise LiteralError(label_graph_prior_type, self.available_label_graph_prior_types)
        if degree_prior_type not in self.available_degree_prior_types:
            raise LiteralError(degree_prior_type, self.available_degree_prior_types)

        if likelihood_type == "degree_corrected":
            if label_graph_prior_type == "nested":
                self.constructor = partial(
                    _graph.NestedDegreeCorrectedStochasticBlockModelFamily,
                    canonical=canonical,
                    degree_hyperprior=(degree_prior_type == "hyper"),
                )
                nested = True
            else:
                self.constructor = partial(
                    _graph.DegreeCorrectedStochasticBlockModelFamily,
                    block_count=block_count,
                    block_hyperprior=True,
                    degree_hyperprior=(degree_prior_type == "hyper"),
                    planted=(label_graph_prior_type == "planted"),
                    canonical=canonical,
                )
        else:
            if label_graph_prior_type == "nested":
                self.constructor = partial(
                    _graph.NestedStochasticBlockModelFamily,
                    stub_labeled=(likelihood_type == "stub_labeled"),
                    canonical=canonical,
                    with_self_loops=True,
                    with_parallel_edges=True,
                )
                nested = True
            else:
                self.constructor = partial(
                    _graph.StochasticBlockModelFamily,
                    block_count=block_count,
                    block_hyperprior=True,
                    planted=(label_graph_prior_type == "planted"),
                    stub_labeled=(likelihood_type == "stub_labeled"),
                    canonical=canonical,
                    with_self_loops=True,
                    with_parallel_edges=True,
                )
        wrapped = self.constructor(size, edge_count)
        super().__init__(
            wrapped,
            size=size,
            edge_count=edge_count,
            block_count=None if block_count == 0 else block_count,
            likelihood_type=likelihood_type,
            label_graph_prior_type=label_graph_prior_type,
            degree_prior_type=degree_prior_type,
            canonical=canonical,
            block_proposer_type=block_proposer_type,
            sample_label_count_prob=sample_label_count_prob,
            shift=shift,
            labeled=labeled,
            nested=nested,
        )
        # self._labels = self.labels()

    def format_graph_into_args(self, graph: bg.UndirectedGraph):
        return dict(size=graph.get_size(), edge_count=graph.get_total_edge_number())

    # def set_labels(self, labels):
    #     self._labels = labels
    #     self.wrap.set_labels(labels)

    def metropolis_sweep(
        self,
        n_label_move: Optional[int] = None,
        n_gibbs: int = 10,
    ):
        if self.params["block_proposer_type"] == "uniform":
            n_steps = n_label_move or self.size()
            begin_log_joint = self.log_joint()
            n_success = self.wrap.metropolis_sweep(n_steps)
            end_log_joint = self.log_joint()
            return end_log_joint - begin_log_joint, n_steps, n_success
        blockstate = self.blockstate()
        if self.params["block_proposer_type"] == "multiflip":
            out = blockstate.multiflip_mcmc_sweep(
                psingle=self.size(),
                psplit=1,
                pmergesplit=1,
                niter=n_label_move // self.size(),
                d=self.params["sample_label_count_prob"],
                c=self.params["shift"],
                entropy_args=self.gt_entropy_args(),
                gibbs_sweeps=n_gibbs,
            )

        elif self.params["block_proposer_type"] == "singleflip":
            out = blockstate.mcmc_sweep(
                c=self.params["shift"],
                d=self.params["sample_label_count_prob"],
                niter=n_label_move // self.size(),
                entropy_args=self.gt_entropy_args(),
                sequential=False,
                deterministic=False,
            )
        else:
            raise ValueError(f"Block proposer type {self.params['block_proposer_type']} is invalid.")

        self.sync_with_blockstate(blockstate)
        return out

    def gt_entropy_args(self):
        return dict(
            adjacency=True,
            dl=True,
            partition_dl=True,
            degree_dl=True,
            degree_dl_kind="distributed" if self.params["degree_prior_type"] == "hyper" else "uniform",
            edges_dl=True,
            dense=self.params["likelihood_type"] == "uniform",
            exact=True,
        )

    def gt_mcmc_args(self):
        if self.params["block_proposer_type"] == "multiflip":
            return dict(
                psplit=1,
                pmergesplit=1,
                niter=5,
                d=self.params["sample_label_count_prob"],
                c=self.params["shift"],
                entropy_args=self.gt_entropy_args(),
                gibbs_sweeps=5,
            )

        elif self.params["block_proposer_type"] == "singleflip":
            return dict(
                c=self.params["shift"],
                d=self.params["sample_label_count_prob"],
                niter=5,
                entropy_args=self.gt_entropy_args(),
                sequential=False,
                deterministic=False,
            )
        return {}

    def blockstate(self):
        bg = self.state()
        gt_graph = convert_basegraph_to_graphtool(bg)
        from importlib.util import find_spec

        if find_spec("graph_tool") is None:
            raise ModuleNotFoundError("Module `graph_tool` has not been installed, cannot use `blockstate` method.")
        from graph_tool.inference import BlockState, NestedBlockState

        if self.params["label_graph_prior_type"] == "uniform":
            b = gt_graph.new_vp("int", vals=self.labels_copy())
            return BlockState(
                g=gt_graph,
                b=b,
                deg_corr=self.params["likelihood_type"] == "degree_corrected",
            )
        bs = [np.array(b) for b in self.nested_labels_copy()]
        return NestedBlockState(
            g=gt_graph,
            bs=bs,
            state_args=dict(deg_corr=self.params["likelihood_type"] == "degree_corrected"),
        )

    def sync_with_blockstate(self, blockstate):
        if self.params["label_graph_prior_type"] == "nested":
            bs = reduce_partition([list(b) for b in blockstate.get_state()])
            self.set_nested_labels(bs)
        else:
            b = reduce_partition(list(blockstate.get_blocks().a))[0]
            self.set_labels(b)
