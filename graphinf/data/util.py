import logging
import sys
import time
import numpy as np
import multiprocessing as mp
from collections import deque
from functools import partial
from typing import Callable, List, Literal, Optional
from warnings import warn

from basegraph import core
from graphinf.data import DataModel
from graphinf.utility import (
    EdgeCollector,
    enumerate_all_graphs,
    log_mean_exp,
    log_sum_exp,
)


def adj_matrix_to_graph(adj_matrix: np.ndarray) -> core.UndirectedMultigraph:
    n = adj_matrix.shape[0]
    g = core.UndirectedMultigraph(size=n)
    for i in range(adj_matrix.shape[0]):
        for j in range(i + 1, adj_matrix.shape[1]):
            if adj_matrix[i, j] > 0:
                g.add_multiedge(i, j, adj_matrix[i, j])
    return g


def mcmc_on_graph(
    model: DataModel,
    n_sweeps: int = 1000,
    n_gibbs_sweeps: int = 1,
    burn_sweeps: int = 0,
    start_from_original: bool = False,
    reset_original: bool = False,
    callback: Optional[Callable[[DataModel], None]] = None,
    logger: Optional[logging.Logger] = None,
    **kwargs,
) -> None:

    original = model.graph()
    if not start_from_original:
        model.sample_prior()

    time_queue = deque(maxlen=100)

    def step(i, prefix=""):
        t0 = time.time()
        summary = model.gibbs_sweep(n_sweeps=n_gibbs_sweeps, **kwargs)
        t1 = time.time()
        time_queue.append(t1 - t0)

        if logger is not None:
            msg = f"[{prefix}]"
            msg += f"Epoch {i}: "
            msg += f"time={t1 - t0: 0.4f}s ({(n_sweeps - i + 1) * np.mean(time_queue): 0.4f}s remaining)\n\t"
            msg += f"accepted={ {k : v / float(summary.total[k]) for k, v in summary.accepted.items()} }, \n\t"
            msg += f"total={ {k : v for k, v in summary.total.items()} }, \n\t"
            msg += f"avg[posterior ratio]={summary.log_joint_ratio: 0.4f}, "
            msg += f"log(likelihood)={model.log_likelihood(): 0.4f}, "
            msg += f"log(prior)={model.log_prior(): 0.4f}, "
            msg += f"log(joint)={model.log_joint(): 0.4f}, \n\t"

            for k, v in model.params.items():
                msg += f"{k}={v: 0.4f}, "
            logger.info(msg)

    for i in range(burn_sweeps):
        step(i, "burn-in")

    for i in range(n_sweeps):
        step(i, "sampling")
        if callback is not None:
            callback(model)

    if reset_original:
        model.set_graph(original)


def log_posterior_meanfield(model: DataModel, graph: core.UndirectedMultigraph, **kwargs):
    collector = EdgeCollector()
    callback = lambda model: collector.update(model.graph_copy())

    model.set_graph(graph)
    callback(model)
    mcmc_on_graph(model, callback=callback, **kwargs)

    return collector.log_prob_estimate(graph)


def log_posterior_exact_meanfield(model: DataModel, graph: core.UndirectedMultigraph, **kwargs):
    g = model.prior
    N, M = g.size(), g.edge_count()
    ws, wp = g.with_self_loops(), g.with_parallel_edges()
    if N > 7:
        warn(f"A model with size {N} is being used" f"for exact evaluation, which might not finish.")
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
        warn(f"A model with size {N} is being used" f"for exact evaluation, which might not finish.")
    samples = []
    original = model.graph_copy()
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
