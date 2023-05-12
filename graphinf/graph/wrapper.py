import logging

from typing import Optional, Callable, Optional

from graphinf.wrapper import Wrapper as _Wrapper
from graphinf._graphinf import graph as _graph
from basegraph import core
from .degree_sequences import poisson_degreeseq, nbinom_degreeseq
from .util import (
    log_evidence_exact,
    log_evidence_iid_meanfield,
    log_evidence_partition_meanfield,
    log_evidence_annealed,
)

__all__ = (
    "OptionError",
    "RandomGraphWrapper",
    "ErdosRenyiModel",
    "ConfigurationModel",
    "ConfigurationModelFamily",
    "PoissonGraph",
    "NegativeBinomialGraph",
    "StochasticBlockModelFamily",
    "PlantedPartitionGraph",
)


class OptionError(ValueError):
    def __init__(self, value: str, avail_options: list[str]) -> None:
        super().__init__(
            f"Option {value} is unavailable, possible options are {avail_options}."
        )


class RandomGraphWrapper(_Wrapper):
    def __init__(self, graph_model, labeled=False, nested=False, **kwargs):
        self.labeled = labeled
        self.nested = nested
        super().__init__(graph_model, params=kwargs)

    def __repr__(self):
        str_format = f"{self.__class__.__name__}("
        if len(self.params) == 0:
            return str_format + ")"
        for k, v in self.params.items():
            if isinstance(v, str):
                v = f"'{v}'"
            str_format += f"\n\t{k}={v},"
        str_format += "\n)"
        return str_format

    def post_init(self):
        self.wrap.sample()

    def get_log_evidence(
        self,
        graph: Optional[core.UndirectedMultigraph] = None,
        method: Optional[str] = None,
        n_sweeps: int = 1000,
        n_steps_per_vertex: int = 10,
        burn: int = 0,
        start_from_original: bool = False,
        reset_original: bool = False,
        verbose: bool = False,
        **kwargs,
    ) -> None:
        all_methods = [
            "exact",
            "iid_meanfield",
            "partition_meanfield",
            "annealed",
        ]
        if method is None:
            method = (
                "exact"
                if (self.get_size() <= 5 or not self.labeled)
                else "iid_meanfield"
            )
        if method not in all_methods:
            raise ValueError(
                f"Cannot parse method '{method}', available options are {all_methods}."
            )
        if graph is None:
            graph = self.get_state()
        kwargs["n_sweeps"] = n_sweeps
        kwargs["n_steps"] = n_steps_per_vertex * self.get_size()
        kwargs["burn"] = burn
        kwargs["start_from_original"] = start_from_original
        kwargs["reset_original"] = reset_original
        kwargs["verbose"] = verbose
        if not self.labeled:
            if method == "exact":
                evidence = self.get_log_joint()
            elif method == "iid_meanfield":
                evidence = log_evidence_iid_meanfield(self, graph, **kwargs)
            self.set_state(graph)
            return evidence

        original = self.get_labels()
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
    def __init__(self, graph: core.UndirectedMultigraph):
        wrapped = _graph.DeltaGraph(graph)
        super().__init__(wrapped)


class ErdosRenyiModel(RandomGraphWrapper):
    def __init__(
        self,
        size: int = 100,
        edge_count: float = 250,
        canonical: bool = False,
        with_self_loops: bool = True,
        with_parallel_edges: bool = True,
        edge_proposer_type: str = "uniform",
    ):
        wrapped = _graph.ErdosRenyiModel(
            size,
            edge_count,
            canonical=canonical,
            with_self_loops=with_self_loops,
            with_parallel_edges=with_parallel_edges,
            edge_proposer_type=edge_proposer_type,
        )
        super().__init__(
            wrapped,
            size=size,
            edge_count=edge_count,
            canonical=canonical,
            with_self_loops=with_self_loops,
            with_parallel_edges=with_parallel_edges,
            edge_proposer_type=edge_proposer_type,
        )


class ConfigurationModelFamily(RandomGraphWrapper):
    available_degree_prior_types = ["uniform", "hyper"]

    def __init__(
        self,
        size: int = 100,
        edge_count: float = 250,
        degree_prior_type: str = "uniform",
        canonical: bool = False,
        edge_proposer_type: str = "degree",
    ):
        if degree_prior_type not in self.available_degree_prior_types:
            raise OptionError(
                degree_prior_type, self.available_degree_prior_types
            )
        wrapped = _graph.ConfigurationModelFamily(
            size,
            edge_count,
            canonical=canonical,
            hyperprior=(degree_prior_type == "hyper"),
            edge_proposer_type=edge_proposer_type,
        )
        super().__init__(
            wrapped,
            size=size,
            edge_count=edge_count,
            degree_prior_type=degree_prior_type,
            canonical=canonical,
            edge_proposer_type=edge_proposer_type,
        )


class ConfigurationModel(RandomGraphWrapper):
    def __init__(self, degree_seq: list[int]):
        print(type(degree_seq))
        wrapped = _graph.ConfigurationModel(degree_seq)
        self.degrees = degree_seq
        super().__init__(
            wrapped, size=len(degree_seq), edge_count=int(sum(degree_seq) / 2)
        )


class PoissonGraph(ConfigurationModel):
    def __init__(self, size: int, edge_count: int):
        avgk = 2 * edge_count / size
        super().__init__(poisson_degreeseq(size, avgk))


class NegativeBinomialGraph(ConfigurationModel):
    def __init__(self, size: int, edge_count: int, heterogeneity: float = 0):
        avgk = 2 * edge_count / size
        super().__init__(nbinom_degreeseq(size, avgk, heterogeneity))


class PlantedPartitionGraph(RandomGraphWrapper):
    def __init__(
        self,
        size: int = 100,
        edge_count: int = 250,
        block_count: int = 3,
        assortativity: float = 0.5,
        stub_labeled: bool = False,
        with_self_loops: bool = True,
        with_parallel_edges: bool = True,
    ):
        wrapped = _graph.PlantedPartitionModel(
            size,
            edge_count,
            block_count=block_count,
            assortativity=assortativity,
            stub_labeled=stub_labeled,
            with_self_loops=with_self_loops,
            with_parallel_edges=with_parallel_edges,
        )
        super().__init__(
            wrapped,
            size=size,
            edge_count=edge_count,
            block_count=block_count,
            assortativity=assortativity,
            stub_labeled=stub_labeled,
            with_self_loops=with_self_loops,
            with_parallel_edges=with_parallel_edges,
        )


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
        block_prior_type: str = "uniform",
        label_graph_prior_type: str = "uniform",
        degree_prior_type: str = "uniform",
        canonical: bool = False,
        with_self_loops: bool = True,
        with_parallel_edges: bool = True,
        edge_proposer_type: str = "uniform",
        block_proposer_type: str = "mixed",
        sample_label_count_prob: float = 0.1,
        label_creation_prob: float = 0.5,
        shift: float = 1,
    ):
        labeled = True
        nested = False
        block_count = 0 if block_count is None else block_count
        if likelihood_type not in self.available_likelihood_types:
            raise OptionError(
                likelihood_type, self.available_likelihood_types
            )
        if block_prior_type not in self.available_block_prior_types:
            raise OptionError(
                block_prior_type, self.available_block_prior_types
            )
        if (
            label_graph_prior_type
            not in self.available_label_graph_prior_types
        ):
            raise OptionError(
                label_graph_prior_type, self.available_label_graph_prior_types
            )
        if degree_prior_type not in self.available_degree_prior_types:
            raise OptionError(
                degree_prior_type, self.available_degree_prior_types
            )

        if likelihood_type == "degree_corrected":
            if label_graph_prior_type == "nested":
                wrapped = (
                    _graph.NestedDegreeCorrectedStochasticBlockModelFamily(
                        size,
                        edge_count,
                        degree_hyperprior=(degree_prior_type == "hyper"),
                        canonical=canonical,
                        edge_proposer_type=edge_proposer_type,
                        block_proposer_type=block_proposer_type,
                        sample_label_count_prob=sample_label_count_prob,
                        label_creation_prob=label_creation_prob,
                        shift=shift,
                    )
                )
                nested = True
            else:
                wrapped = _graph.DegreeCorrectedStochasticBlockModelFamily(
                    size,
                    edge_count,
                    block_count=block_count,
                    block_hyperprior=(block_prior_type == "hyper"),
                    degree_hyperprior=(degree_prior_type == "hyper"),
                    planted=(label_graph_prior_type == "planted"),
                    canonical=canonical,
                    edge_proposer_type=edge_proposer_type,
                    block_proposer_type=block_proposer_type,
                    sample_label_count_prob=sample_label_count_prob,
                    label_creation_prob=label_creation_prob,
                    shift=shift,
                )
        else:
            if label_graph_prior_type == "nested":
                wrapped = _graph.NestedStochasticBlockModelFamily(
                    size,
                    edge_count,
                    stub_labeled=(likelihood_type == "stub_labeled"),
                    canonical=canonical,
                    with_self_loops=with_self_loops,
                    with_parallel_edges=with_parallel_edges,
                    edge_proposer_type=edge_proposer_type,
                    block_proposer_type=block_proposer_type,
                    sample_label_count_prob=sample_label_count_prob,
                    label_creation_prob=label_creation_prob,
                    shift=shift,
                )
                nested = True
            else:
                wrapped = _graph.StochasticBlockModelFamily(
                    size,
                    edge_count,
                    block_count=block_count,
                    block_hyperprior=(block_prior_type == "hyper"),
                    planted=(label_graph_prior_type == "planted"),
                    stub_labeled=(likelihood_type == "stub_labeled"),
                    canonical=canonical,
                    with_self_loops=with_self_loops,
                    with_parallel_edges=with_parallel_edges,
                    edge_proposer_type=edge_proposer_type,
                    block_proposer_type=block_proposer_type,
                    sample_label_count_prob=sample_label_count_prob,
                    label_creation_prob=label_creation_prob,
                    shift=shift,
                )
        super().__init__(
            wrapped,
            size=size,
            edge_count=edge_count,
            block_count=None if block_count == 0 else block_count,
            likelihood_type=likelihood_type,
            block_prior_type=block_prior_type,
            label_graph_prior_type=label_graph_prior_type,
            degree_prior_type=degree_prior_type,
            canonical=canonical,
            with_self_loops=with_self_loops,
            with_parallel_edges=with_parallel_edges,
            edge_proposer_type=edge_proposer_type,
            block_proposer_type=block_proposer_type,
            sample_label_count_prob=sample_label_count_prob,
            label_creation_prob=label_creation_prob,
            shift=shift,
            labeled=labeled,
            nested=nested,
        )
