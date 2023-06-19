import logging
import sys
import time
from functools import partial
from typing import Callable, List, Literal, Optional
from warnings import warn

import numpy as np
from basegraph import core
from graphinf.data import DataModel
from graphinf.utility import (
    EdgeCollector,
    enumerate_all_graphs,
    log_mean_exp,
    log_sum_exp,
)


def mcmc_on_graph(
    model: DataModel,
    n_sweeps: int = 1000,
    sweep_type: Literal["metropolis", "gibbs"] = "metropolis",
    n_steps: int = 1000,
    burn: int = 0,
    beta_prior: float = 1,
    beta_likelihood: float = 1,
    start_from_original: bool = False,
    reset_original: bool = False,
    callback: Optional[Callable[[DataModel], None]] = None,
    graph_rate: Optional[float] = None,
    prior_rate: Optional[float] = None,
    param_rate: Optional[float] = None,
    verbose: bool = False,
) -> None:
    if sweep_type == "metropolis":
        sweep = model.metropolis_sweep

    elif sweep_type == "gibbs":
        sweep = model.gibbs_sweep
    else:
        raise ValueError()

    if verbose:
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    else:
        logger = None

    original = model.get_graph()
    if not start_from_original:
        model.sample_prior()

    if graph_rate is not None and graph_rate >= 0:
        model.unfreeze_graph(graph_rate)
    if prior_rate is not None and prior_rate >= 0:
        model.unfreeze_graph_prior(prior_rate)
    if param_rate is not None and param_rate >= 0:
        model.unfreeze_param(param_rate)

    if burn > 0:
        sweep(burn, beta_prior=beta_prior, beta_likelihood=beta_likelihood)
    for i in range(n_sweeps):
        t0 = time.time()
        success = sweep(
            n_steps, beta_prior=beta_prior, beta_likelihood=beta_likelihood
        )
        t1 = time.time()
        if logger is not None:
            logger.info(
                f"Epoch {i}: "
                f"time={t1 - t0: 0.4f}, "
                f"accepted={success}, "
                f"log(likelihood)={model.log_likelihood(): 0.4f}, "
                f"log(prior)={model.log_prior(): 0.4f}"
            )

        if callback is not None:
            callback(model)

    if reset_original:
        model.set_graph(original)


def log_posterior_meanfield(
    model: DataModel, graph: core.UndirectedMultigraph, **kwargs
):
    collector = EdgeCollector()
    callback = lambda model: collector.update(model.get_graph())

    model.set_graph(graph)
    callback(model)
    mcmc_on_graph(model, callback=callback, **kwargs)

    return collector.log_prob_estimate(graph)


def log_posterior_exact_meanfield(
    model: DataModel, graph: core.UndirectedMultigraph, **kwargs
):
    g = model.graph_prior
    N, M = g.get_size(), g.get_edge_count()
    ws, wp = g.with_self_loops(), g.with_parallel_edges()
    if N > 7:
        warn(
            f"A model with size {N} is being used"
            f"for exact evaluation, which might not finish."
        )
    original = model.get_graph()
    evidence = []

    logits = dict()
    for g in enumerate_all_graphs(N, M, selfloops=ws, parallel_edges=wp):
        model.set_graph(g)
        likelihood = model.log_likelihood()
        prior = model.graph_prior.log_evidence(method="exact")
        evidence.append(likelihood + prior)
        for e in g.edges():
            logits[e] = likelihood + prior
    model.set_graph(original)
    evidence = log_sum_exp(evidence)

    logp = 0
    for e in original.edges():
        logp += logits[e] - evidence
    return logp


def log_evidence_exact(model: DataModel, **kwargs):
    g = model.graph_prior
    N, M = g.get_size(), g.get_edge_count()
    ws, wp = g.with_self_loops(), g.with_parallel_edges()
    if N > 7:
        warn(
            f"A model with size {N} is being used"
            f"for exact evaluation, which might not finish."
        )
    original = model.get_graph()
    samples = []
    for g in enumerate_all_graphs(N, M, selfloops=ws, parallel_edges=wp):
        model.set_graph(g)
        likelihood = model.log_likelihood()
        prior = model.graph_prior.log_evidence(method="exact")
        samples.append(likelihood + prior)
    model.set_graph(original)
    return log_sum_exp(samples)


def log_evidence_annealed(
    model: DataModel, betas: List[float] = None, **kwargs
):
    if betas is None:
        betas = np.linspace(0, 1, 11) ** (1.0 / 2)

    kwargs.pop("beta_likelihood")
    samples = []
    for lb, ub in zip(betas[:-1], betas[1:]):
        likelihoods = []
        callback = lambda model: likelihoods.append(model.log_likelihood())
        kwargs["beta_likelihood"] = lb
        if kwargs.get("verbose"):
            print(f"---Temps: {lb:0.4f}---")
        mcmc_on_graph(model, callback=callback, **kwargs)
        logp_k = (ub - lb) * np.array(likelihoods)
        samples.append(log_mean_exp(logp_k))

    return sum(samples)
