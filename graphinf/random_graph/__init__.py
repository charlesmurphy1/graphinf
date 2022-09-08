from _graphinf import random_graph as _random_graph
from _graphinf.random_graph import (
    RandomGraph,
    BlockLabeledRandomGraph,
    NestedBlockLabeledRandomGraph,
    ErdosRenyiModel,
    ConfigurationModelFamily,
    StochasticBlockModel,
    PlantedPartitionModel,
)
from ..wrapper import Wrapper as _Wrapper
from .degree_sequences import poisson_degreeseq, nbinom_degreeseq

__all__ = (
    "RandomGraph",
    "RandomGraphWrapper",
    "BlockLabeledRandomGraph",
    "NestedBlockLabeledRandomGraph",
    "ErdosRenyiModel",
    "ConfigurationModel",
    "ConfigurationModelFamily",
    "StochasticBlockModel",
    "StochasticBlockModelFamily",
    "PlantedPartitionModel",
)


class RandomGraphWrapper(_Wrapper):
    def __init__(self, graph_model, labeled=False, nested=False, **kwargs):
        self.labeled = labeled
        self.nested = nested
        super().__init__(graph_model, params=kwargs)

    def post_init(self):
        self.wrap.sample()


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
        wrapped = _random_graph.ErdosRenyiModel(
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
        edge_proposer_type: str = "uniform",
    ):
        if degree_prior_type not in self.available_degree_prior_types:
            raise OptionError(degree_prior_type, self.available_degree_prior_types)
        wrapped = _random_graph.ConfigurationModelFamily(
            size,
            edge_count,
            canonical=canonical,
            hyperprior=(degree_prior_type == "hyperprior"),
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
        wrapped = _random_graph.ConfigurationModel(degree_seq)
        super().__init__(wrapped)


class PoissonModel(ConfigurationModel):
    def __init__(self, size: int, edge_count: int):
        avgk = 2 * edge_count / size
        super().__init__(poisson_degreeseq(size, avgk))


class NegativeBinomialModel(ConfigurationModel):
    def __init__(self, size: int, edge_count: int, heterogeneity: float = 0):
        avgk = 2 * edge_count / size
        super().__init__(nbinom_degreeseq(size, avgk, heterogeneity))


class PlantedPartitionModel(RandomGraphWrapper):
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
        wrapped = _random_graph.PlantedPartitionModel(
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
    available_likelihood_types = ["uniform", "stub_labeled", "degree_corrected"]
    available_block_prior_types = ["uniform", "hyper"]
    available_label_graph_prior_types = ["uniform", "planted", "nested"]
    available_degree_prior_types = ["uniform", "hyper"]

    def __init__(
        self,
        size: int = 100,
        edge_count: float = 250,
        block_count: int = 0,
        likelihood_type: str = "uniform",
        block_prior_type: str = "uniform",
        label_graph_prior_type: str = "uniform",
        degree_prior_type: str = "uniform",
        canonical: bool = False,
        with_self_loops: bool = True,
        with_parallel_edges: bool = True,
        edge_proposer_type: str = "uniform",
        block_proposer_type: str = "uniform",
        sample_label_count_prob: float = 0.1,
        label_creation_prob: float = 0.5,
        shift: float = 1,
    ):
        labeled = True
        nested = False
        if likelihood_type not in self.available_likelihood_types:
            raise OptionError(likelihood_type, self.available_likelihood_types)
        if block_prior_type not in self.available_block_prior_types:
            raise OptionError(block_prior_type, self.available_block_prior_types)
        if label_graph_prior_type not in self.available_label_graph_prior_types:
            raise OptionError(
                label_graph_prior_type, self.available_label_graph_prior_types
            )
        if degree_prior_type not in self.available_degree_prior_types:
            raise OptionError(degree_prior_type, self.available_degree_prior_types)

        if likelihood_type == "degree_corrected":
            if label_graph_prior_type == "nested":
                wrapped = _random_graph.NestedDegreeCorrectedStochasticBlockModelFamily(
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
                nested = True
            else:
                wrapped = _random_graph.DegreeCorrectedStochasticBlockModelFamily(
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
                wrapped = _random_graph.NestedStochasticBlockModelFamily(
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
                wrapped = _random_graph.StochasticBlockModelFamily(
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
            block_count=block_count,
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
