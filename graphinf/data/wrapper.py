from __future__ import annotations
import numpy as np

from typing import Literal, Optional, Callable
from collections import defaultdict

from basegraph import core
from graphinf._graphinf.data import DataModel
from graphinf._graphinf.utility import enumerate_all_graphs
from graphinf.graph import DeltaGraph as _DeltaGraph
from graphinf.graph import ErdosRenyiModel as _ErdosRenyiModel
from graphinf.graph import RandomGraphWrapper as _RandomGraphWrapper
from graphinf.wrapper import Wrapper as _Wrapper
from graphinf.utility import EdgeCollector

from .util import (
    mcmc_on_graph,
    log_evidence_annealed,
    log_evidence_exact,
    log_posterior_exact_meanfield,
    log_posterior_meanfield,
)


class DataModelWrapper(_Wrapper):
    constructor = None

    def __init__(
        self,
        prior: _RandomGraphWrapper or core.UndirectedMultigraph = None,
        **kwargs,
    ):
        if prior is None:
            prior = _ErdosRenyiModel(100, 250) if prior is None else prior
        elif isinstance(prior, core.UndirectedMultigraph):
            prior = _DeltaGraph(prior)
        self.labeled = prior.labeled
        self.nested = prior.nested
        data_model = self.constructor(prior.wrap, **kwargs)
        data_model.sample()
        super().__init__(data_model, prior=prior, params=kwargs)

    def __repr__(self):
        str_format = f"{self.__class__.__name__ }(\n\tprior={self.prior.__class__.__name__},"

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

    @property
    def dtype(self):
        if self.nested:
            return "nested"
        elif self.labeled:
            return "labeled"
        return "normal"

    def from_model(self, other: DataModelWrapper):
        if issubclass(other.__class__, DataModelWrapper):
            self.wrap.set_state_from(other.wrap)
        elif issubclass(other.__class__, DataModel):
            self.wrap.set_state_from(other)
        else:
            raise TypeError(
                f"Model `{other}` has an invalid type `{other.__class__.__name__}`"
            )

    def graph_copy(self):
        return self.graph().get_deep_copy()

    def set_prior(self, prior: _RandomGraphWrapper):
        self.labeled = prior.labeled
        self.nested = prior.nested
        self.prior = prior
        self.wrap.set_graph_prior(prior.wrap)
        self.__wrapped__.sample()

    def gibbs_sweep(
        self,
        n_sweeps: int = 4,
        n_steps_per_vertex: int = 1,
        beta_likelihood: float = 1,
        beta_prior: float = 1,
        sample_prior: bool = False,
        sample_params: bool = False,
        **kwargs,
    ):
        n_success = 0
        for _ in range(n_sweeps):
            n_success += self.wrap.metropolis_graph_sweep(
                num_steps=n_steps_per_vertex * self.size(),
                beta_likelihood=beta_likelihood,
                beta_prior=beta_prior,
                **kwargs,
            )
            if self.prior.labeled and sample_prior:
                _, _, m = self.prior.metropolis_sweep(**kwargs)
                n_success += m
            if sample_params:
                n_success += self.wrap.metropolis_param_sweep(
                    n_steps_per_vertex,
                    beta_prior=beta_prior,
                    beta_likelihood=beta_likelihood,
                )
        return n_success

    def posterior_entropy(
        self,
        method: Literal["exact", "meanfield"] = "exact",
        n_sweeps: int = 1000,
        n_gibbs_sweeps: int = 10,
        n_steps_per_vertex: int = 1,
        burn_sweeps: int = 0,
        sample_prior: bool = True,
        sample_params: bool = False,
        start_from_original: bool = False,
        reset_original: bool = True,
        verbose: bool = False,
        **kwargs,
    ):
        kwargs["n_sweeps"] = n_sweeps
        kwargs["n_gibbs_sweeps"] = n_gibbs_sweeps
        kwargs["n_steps_per_vertex"] = n_steps_per_vertex
        kwargs["burn_sweeps"] = burn_sweeps
        kwargs["sample_prior"] = sample_prior
        kwargs["sample_params"] = sample_params
        kwargs["start_from_original"] = start_from_original
        kwargs["reset_original"] = reset_original
        kwargs["verbose"] = verbose

        graph = self.graph_copy()
        N, M, loopy, multigraph = (
            self.prior.size(),
            self.prior.edge_count(),
            self.prior.with_self_loops(),
            self.prior.with_parallel_edges(),
        )
        if graph.get_size() > 6 or method == "meanfield":
            collector = EdgeCollector()
            callback = lambda model: collector.update(model.graph_copy())

            self.set_graph(graph)
            callback(self)
            mcmc_on_graph(self, callback=callback, **kwargs)
            self.set_graph(graph)
            entropy = collector.entropy()
        elif method == "exact_meanfield":
            logprob = defaultdict(float)
            log_evidence = self.log_evidence(method="exact", **kwargs)
            for g in enumerate_all_graphs(N, M, loopy, multigraph):
                self.set_graph(g)
                for e in g.edges():
                    logprob[e] += np.exp(self.log_joint() - log_evidence)
            entropy = -np.sum([p * np.log(p) for p in logprob.values()])
        else:
            entropy = 0
            log_evidence = self.log_evidence(method="exact", **kwargs)
            for g in enumerate_all_graphs(N, M, loopy, multigraph):
                self.set_graph(g)
                logp = self.log_joint() - log_evidence
                entropy -= np.exp(logp) * logp
        if reset_original:
            self.set_graph(graph)
        return entropy

    def log_posterior(
        self,
        graph: Optional[core.UndirectedMultigraph] = None,
        method: Literal[
            "exact", "meanfield", "exact_meanfield", "annealed"
        ] = "exact",
        n_sweeps: int = 1000,
        n_gibbs_sweeps: int = 10,
        n_steps_per_vertex: int = 1,
        burn_sweeps: int = 0,
        sample_prior: bool = True,
        sample_params: bool = False,
        start_from_original: bool = False,
        reset_original: bool = False,
        verbose: bool = False,
        **kwargs,
    ):
        all_methods = ["exact", "meanfield", "annealed"]
        if method not in all_methods:
            raise ValueError(
                f"Cannot parse method '{method}', available options are {all_methods}."
            )

        kwargs["n_sweeps"] = n_sweeps
        kwargs["n_gibbs_sweeps"] = n_gibbs_sweeps
        kwargs["n_steps_per_vertex"] = n_steps_per_vertex
        kwargs["burn_sweeps"] = burn_sweeps
        kwargs["sample_prior"] = sample_prior
        kwargs["sample_params"] = sample_params
        kwargs["start_from_original"] = start_from_original
        kwargs["reset_original"] = reset_original
        kwargs["verbose"] = verbose

        graph = self.graph_copy() if graph is None else graph
        if method == "meanfield":
            posterior = log_posterior_meanfield(self, graph, **kwargs)
        elif method == "exact_meanfield":
            posterior = log_posterior_exact_meanfield(self, graph, **kwargs)
        else:
            self.set_graph(graph)
            prior = self.prior.log_evidence(**kwargs.get("prior_args", {}))
            likelihood = self.log_likelihood()
            if method == "exact":
                posterior = prior + likelihood - log_evidence_exact(self)
            elif method == "annealed":
                posterior = (
                    prior + likelihood - log_evidence_annealed(**kwargs)
                )
        self.set_graph(graph)
        return posterior

    def log_evidence(
        self,
        method: Optional[str] = None,
        n_sweeps: int = 1000,
        n_gibbs_sweeps: int = 10,
        n_steps_per_vertex: int = 1,
        burn_sweeps: int = 0,
        sample_prior: bool = True,
        sample_params: bool = False,
        start_from_original: bool = False,
        reset_original: bool = False,
        verbose: bool = False,
        **kwargs,
    ):
        if method not in ["exact", "meanfield", "annealed"]:
            raise ValueError(
                f"Cannot parse method '{method}', available options are ['exact', 'meanfield', 'annealed']."
            )

        kwargs["n_sweeps"] = n_sweeps
        kwargs["n_gibbs_sweeps"] = n_gibbs_sweeps
        kwargs["n_steps_per_vertex"] = n_steps_per_vertex
        kwargs["burn_sweeps"] = burn_sweeps
        kwargs["sample_prior"] = sample_prior
        kwargs["sample_params"] = sample_params
        kwargs["start_from_original"] = start_from_original
        kwargs["reset_original"] = reset_original
        kwargs["verbose"] = verbose

        if method == "exact":
            return log_evidence_exact(self)
        if method == "annealed":
            return log_evidence_annealed(self, **kwargs)
        prior = self.prior.log_evidence(**kwargs.get("prior_args", {}))
        likelihood = self.log_likelihood()
        posterior = log_posterior_meanfield(self, self.graph_copy(), **kwargs)
        return prior + likelihood - posterior
