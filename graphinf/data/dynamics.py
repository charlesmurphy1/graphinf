from __future__ import annotations
import numpy as np

from typing import List, Optional
from graphinf import _graphinf
from graphinf.graph import RandomGraph
from graphinf.wrapper import Wrapper

_dynamics = _graphinf.data.dynamics
from graphinf.data import DataModelWrapper as _DataModelWrapper

__all__ = (
    "SISDynamics",
    "GlauberDynamics",
    "CowanDynamics",
    "VoterDynamics",
)


class Dynamics(_DataModelWrapper):
    def set_state(self, state: np.ndarray | List[List[int]], future: Optional[np.ndarray | List[List[int]]] = None):
        if future is None:
            past = [x[:-1] for x in state]
            future = [x[1:] for x in state]
        else:
            past = state

        if isinstance(past, np.ndarray):
            past = past.tolist()
        if isinstance(future, np.ndarray):
            future = future.tolist()

        self.wrap.set_state(past, future)


class SISDynamics(Dynamics):
    constructor = _dynamics.SISDynamics
    _param_list = [
        "infection_prob",
        "recovery_prob",
        "auto_activation_prob",
        "auto_deactivation_prob",
    ]

    def __init__(
        self,
        prior: RandomGraph | Wrapper = None,
        length: int = 10,
        infection_prob: float = 0.5,
        recovery_prob: float = 0.5,
        auto_activation_prob: float = 1e-06,
        auto_deactivation_prob: float = 0.0,
    ):
        super().__init__(
            prior=prior,
            length=length,
            infection_prob=infection_prob,
            recovery_prob=recovery_prob,
            auto_activation_prob=auto_activation_prob,
            auto_deactivation_prob=auto_deactivation_prob,
        )


class GlauberDynamics(Dynamics):
    constructor = _dynamics.GlauberDynamics
    _param_list = [
        "coupling",
        "auto_activation_prob",
        "auto_deactivation_prob",
    ]

    def __init__(
        self,
        prior: RandomGraph = None,
        length: int = 10,
        coupling: float = 1,
        auto_activation_prob: float = 0.0,
        auto_deactivation_prob: float = 0.0,
    ):
        super().__init__(
            prior=prior,
            length=length,
            coupling=coupling,
            auto_activation_prob=auto_activation_prob,
            auto_deactivation_prob=auto_deactivation_prob,
        )


class CowanDynamics(Dynamics):
    constructor = _dynamics.CowanDynamics
    _param_list = [
        "nu",
        "a",
        "mu",
        "eta",
        "auto_activation_prob",
        "auto_deactivation_prob",
    ]

    def __init__(
        self,
        prior: RandomGraph = None,
        length: int = 10,
        nu: float = 1,
        a: float = 1,
        mu: float = 1,
        eta: float = 0.5,
        auto_activation_prob: float = 0.0,
        auto_deactivation_prob: float = 0.0,
    ):
        super().__init__(
            prior=prior,
            length=length,
            nu=nu,
            a=a,
            mu=mu,
            eta=eta,
            auto_activation_prob=auto_activation_prob,
            auto_deactivation_prob=auto_deactivation_prob,
        )


class VoterDynamics(Dynamics):
    constructor = _dynamics.VoterDynamics
    _param_list = [
        "random_flip_prob",
        "auto_activation_prob",
        "auto_deactivation_prob",
    ]

    def __init__(
        self,
        prior: RandomGraph = None,
        random_flip_prob: float = 0.0,
        length: int = 10,
        auto_activation_prob: float = 0.0,
        auto_deactivation_prob: float = 0.0,
    ):
        super().__init__(
            prior=prior,
            random_flip_prob=random_flip_prob,
            length=length,
            auto_activation_prob=auto_activation_prob,
            auto_deactivation_prob=auto_deactivation_prob,
        )


class DegreeDynamics(Dynamics):
    constructor = _dynamics.DegreeDynamics

    def __init__(self, prior: RandomGraph = None, length: int = 10, C: float = 10):
        super().__init__(prior=prior, length=length, C=C)
