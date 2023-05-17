from __future__ import annotations
from typing import Optional, Literal
from graphinf.wrapper import Wrapper as _Wrapper
from graphinf.graph import (
    RandomGraphWrapper as _RandomGraphWrapper,
    ErdosRenyiModel as _ErdosRenyiModel,
    DeltaGraph as _DeltaGraph,
)
from basegraph import core
from graphinf._graphinf.data import DataModel

from .util import (
    log_posterior_meanfield,
    log_evidence_exact,
    log_evidence_annealed,
)


class DataModelWrapper(_Wrapper):
    constructor = None

    def __init__(
        self,
        graph_prior: _RandomGraphWrapper or core.UndirectedMultigraph = None,
        **kwargs,
    ):
        if graph_prior is None:
            graph_prior = (
                _ErdosRenyiModel(100, 250)
                if graph_prior is None
                else graph_prior
            )
        elif isinstance(graph_prior, core.UndirectedMultigraph):
            graph_prior = _DeltaGraph(graph_prior)
        self.labeled = graph_prior.labeled
        self.nested = graph_prior.nested
        data_model = self.constructor(graph_prior.wrap, **kwargs)
        data_model.sample()
        if not graph_prior.labeled:
            data_model.freeze_graph_prior()
        super().__init__(data_model, graph_prior=graph_prior, params=kwargs)

    def __repr__(self):
        str_format = f"{self.__class__.__name__ }(\n\tprior={self.graph_prior.__class__.__name__},"

        for k, v in self.params.items():
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

    def set_state_from(self, other: DataModelWrapper):
        if issubclass(other.__class__, DataModelWrapper):
            self.wrap.set_state_from(other.wrap)
        elif issubclass(other.__class__, DataModel):
            self.wrap.set_state_from(other)
        else:
            raise TypeError(
                f"Model `{other}` has an invalid type `{other.__class__.__name__}`"
            )

    def set_graph_prior(self, graph_prior: _RandomGraphWrapper):
        self.labeled = graph_prior.labeled
        self.nested = graph_prior.nested
        self.graph_prior = graph_prior
        self.wrap.set_graph_prior(graph_prior.wrap)
        self.__wrapped__.sample()

    def get_log_posterior(
        self,
        graph: Optional[core.UndirectedMultigraph] = None,
        method: Optional[str] = None,
        sweep_type: Literal["metropolis", "gibbs"] = "metropolis",
        n_sweeps: int = 1000,
        n_steps_per_vertex: int = 10,
        burn: int = 0,
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

        kwargs["sweep_type"] = sweep_type
        kwargs["n_sweeps"] = n_sweeps
        kwargs["n_steps"] = n_steps_per_vertex * self.get_size()
        kwargs["burn"] = burn
        kwargs["start_from_original"] = start_from_original
        kwargs["reset_original"] = reset_original
        kwargs["verbose"] = verbose

        if method == "meanfield":
            posterior = log_posterior_meanfield(self, graph, **kwargs)
        else:
            self.set_graph(graph)
            prior = self.graph_prior.get_log_evidence(
                **kwargs.get("prior_args", {})
            )
            likelihood = self.get_log_likelihood()
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
        sweep_type: Literal["metropolis", "gibbs"] = "metropolis",
        n_sweeps: int = 1000,
        n_steps_per_vertex: int = 10,
        burn: int = 0,
        start_from_original: bool = False,
        reset_original: bool = False,
        verbose: bool = False,
        **kwargs,
    ):
        if method not in ["exact", "meanfield", "annealed"]:
            raise ValueError(
                f"Cannot parse method '{method}', available options are ['exact', 'meanfield', 'annealed']."
            )

        kwargs["sweep_type"] = sweep_type
        kwargs["n_sweeps"] = n_sweeps
        kwargs["n_steps"] = n_steps_per_vertex * self.get_size()
        kwargs["burn"] = burn
        kwargs["start_from_original"] = start_from_original
        kwargs["reset_original"] = reset_original
        kwargs["verbose"] = verbose

        if method == "exact":
            return log_evidence_exact(self)
        if method == "annealed":
            return log_evidence_annealed(self, **kwargs)
        prior = self.graph_prior.get_log_evidence(
            **kwargs.get("prior_args", {})
        )
        likelihood = self.get_log_likelihood()
        posterior = log_posterior_meanfield(self, self.get_graph(), **kwargs)
        return prior + likelihood - posterior
