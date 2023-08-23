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
    n_gibbs_sweeps: int = 4,
    n_steps_per_vertex: int = 1,
    burn_sweeps: int = 0,
    beta_prior: float = 1,
    beta_likelihood: float = 1,
    sample_prior: bool = True,
    sample_params: bool = False,
    start_from_original: bool = False,
    reset_original: bool = False,
    callback: Optional[Callable[[DataModel], None]] = None,
    verbose: bool = False,
) -> None:
    if verbose:
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    else:
        logger = None

    sweep = partial(
        model.gibbs_sweep,
        n_sweeps=n_gibbs_sweeps,
        n_steps_per_vertex=n_steps_per_vertex,
        beta_prior=beta_prior,
        beta_likelihood=beta_likelihood,
        sample_prior=sample_prior,
        sample_params=sample_params,
    )

    original = model.graph()
    if not start_from_original:
        model.sample_prior()

    for _ in range(burn_sweeps):
        sweep()
    for i in range(n_sweeps):
        t0 = time.time()
        success = sweep()
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
    callback = lambda model: collector.update(model.graph_copy())

    model.set_graph(graph)
    callback(model)
    mcmc_on_graph(model, callback=callback, **kwargs)

    return collector.log_prob_estimate(graph)


def log_posterior_exact_meanfield(
    model: DataModel, graph: core.UndirectedMultigraph, **kwargs
):
    g = model.prior
    N, M = g.size(), g.edge_count()
    ws, wp = g.with_self_loops(), g.with_parallel_edges()
    if N > 7:
        warn(
            f"A model with size {N} is being used"
            f"for exact evaluation, which might not finish."
        )
    original = model.graph_copy()
    evidence = []

    logits = dict()
    for g in enumerate_all_graphs(N, M, selfloops=ws, parallel_edges=wp):
        model.set_graph(g)
        likelihood = model.log_likelihood()
        prior = model.prior.log_evidence(method="exact")
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
    g = model.prior
    N, M = g.size(), g.edge_count()
    ws, wp = g.with_self_loops(), g.with_parallel_edges()
    if N > 7:
        warn(
            f"A model with size {N} is being used"
            f"for exact evaluation, which might not finish."
        )
    original = model.graph_copy()
    samples = []
    for g in enumerate_all_graphs(N, M, selfloops=ws, parallel_edges=wp):
        model.set_graph(g)
        likelihood = model.log_likelihood()
        prior = model.prior.log_evidence(method="exact")
        samples.append(likelihood + prior)
    model.set_graph(original)
    return log_sum_exp(samples)


def log_evidence_annealed(model: DataModel, betas: List[float] = None, **kwargs):
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
