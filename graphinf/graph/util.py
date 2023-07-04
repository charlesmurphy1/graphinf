import importlib
import logging
import sys
import time
from typing import Callable, Optional
from warnings import warn

import numpy as np
import tqdm
from basegraph import core
from graphinf.graph import RandomGraph
from graphinf.utility import (
    EdgeCollector,
    enumerate_all_partitions,
    log_mean_exp,
    log_sum_exp,
)
from scipy.special import loggamma


def mcmc_on_labels(
    model: RandomGraph,
    n_sweeps: int = 1000,
    n_steps: int = 1000,
    burn: int = 0,
    beta_prior: float = 1,
    beta_likelihood: float = 1,
    start_from_original: bool = False,
    reset_original: bool = False,
    callback: Optional[Callable[[RandomGraph], None]] = None,
    verbose: bool = False,
    **kwargs,
) -> None:
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

    original = model.get_labels()
    if not start_from_original:
        model.sample_only_labels()

    if burn > 0:
        model.metropolis_sweep(
            burn, beta_prior=beta_prior, beta_likelihood=beta_likelihood
        )
    for i in range(n_sweeps):
        t0 = time.time()
        success = model.metropolis_sweep(
            n_steps, beta_prior=beta_prior, beta_likelihood=beta_likelihood
        )
        model.reduce_labels()
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
        model.set_labels(original)


def log_evidence_exact(
    model: RandomGraph,
    graph: core.UndirectedMultigraph,
    verbose: bool = False,
) -> float:
    if model.nested:
        raise TypeError("`model` must not be nested for exact evaluation.")

    if model.get_size() > 6:
        warn(
            f"A model with size {model.get_size()} is being used"
            f"for exact evaluation, which might not finish."
        )

    original = model.get_labels()
    model.set_state(graph)
    samples = []
    partitions = enumerate_all_partitions(model.get_size())
    if verbose:
        partitions = tqdm.tqdm(partitions)
    for p in partitions:
        model.set_labels(p, False)
        samples.append(model.log_joint())

    model.set_labels(original)
    return log_sum_exp(samples)


def log_evidence_iid_meanfield(
    model: RandomGraph,
    graph: core.UndirectedMultigraph,
    **kwargs,
):
    collector = EdgeCollector()
    collector.update(graph)
    sweeps = range(kwargs.get("n_sweeps", 100))
    if kwargs.get("verbose", False):
        sweeps = tqdm.tqdm(sweeps)
    for _ in sweeps:
        model.sample()
        collector.update(model.get_state())
    return collector.log_prob_estimate(graph)


def log_evidence_partition_meanfield(
    model: RandomGraph, graph: core.UndirectedMultigraph, **kwargs
):
    if importlib.util.find_spec("graph_tool"):
        from graph_tool.inference import ModeClusterState, mcmc_equilibrate
    else:
        raise ModuleNotFoundError(
            "Module `graph_tool` has not been installed, cannot use `log_evidence_partition_meanfield` method."
        )

    get_labels = (
        model.get_labels if not model.nested else model.get_nested_labels
    )
    set_labels = (
        model.set_labels if not model.nested else model.set_nested_labels
    )
    original = model.get_labels()

    if not kwargs.get("start_from_original", False):
        model.sample_only_labels()
    partitions = [original]
    callback = lambda model: partitions.append(get_labels())

    mcmc_on_labels(model, callback=callback, **kwargs)

    pmodes = ModeClusterState(partitions, nested=model.nested)
    if kwargs.get("equilibriate_mode_cluster", False):
        mcmc_equilibrate(
            pmodes, force_niter=1, verbose=kwargs.get("verbose", False)
        )
    samples = []
    for p in partitions:
        set_labels(p)
        samples.append(model.log_joint() + loggamma(1 + len(np.unique(p))))
    if kwargs.get("reset_original", True):
        set_labels(original)
    return np.mean(samples) + pmodes.posterior_entropy()


def log_evidence_annealed(
    model: RandomGraph, graph: core.UndirectedMultigraph, **kwargs
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
        mcmc_on_labels(model, callback=callback, **kwargs)
        logp_k = (ub - lb) * np.array(likelihoods)
        samples.append(log_mean_exp(logp_k))

    return sum(samples)
