from __future__ import annotations
from graphinf import _graphinf
from graphinf.graph import RandomGraph
from graphinf.wrapper import Wrapper

_dynamics = _graphinf.data.dynamics
from graphinf.data import DataModelWrapper as _DataModelWrapper

__all__ = (
    "Dynamics",
    "SISDynamics",
    "GlauberDynamics",
    "CowanDynamics",
)

Dynamics = _dynamics.Dynamics

class SISDynamics(_DataModelWrapper):
    constructor = _dynamics.SISDynamics
    def __init__(
        self,
        graph_prior: RandomGraph | Wrapper = None,
        length: int = 10,
        infection_prob: float = 0.5,
        recovery_prob: float = 0.5,
        auto_activation_prob: float = 1e-06,
        auto_deactivation_prob: float = 0.0,
    ):
        super().__init__(
            graph_prior=graph_prior,
            length=length,
            infection_prob=infection_prob,
            recovery_prob=recovery_prob,
            auto_activation_prob=auto_activation_prob,
            auto_deactivation_prob=auto_deactivation_prob,
        )


class GlauberDynamics(_DataModelWrapper):
    constructor = _dynamics.GlauberDynamics

    def __init__(
        self,
        graph_prior: RandomGraph = None,
        length: int = 10,
        coupling: float = 1,
        auto_activation_prob: float = 0.0,
        auto_deactivation_prob: float = 0.0,
    ):
        super().__init__(
            graph_prior=graph_prior,
            length=length,
            coupling=coupling,
            auto_activation_prob=auto_activation_prob,
            auto_deactivation_prob=auto_deactivation_prob,
        )


class CowanDynamics(_DataModelWrapper):
    constructor = _dynamics.CowanDynamics

    def __init__(
        self,
        graph_prior: RandomGraph = None,
        length: int = 10,
        nu: float = 1,
        a: float = 1,
        mu: float = 1,
        eta: float = 0.5,
        auto_activation_prob: float = 0.0,
        auto_deactivation_prob: float = 0.0,
    ):
        super().__init__(
            graph_prior=graph_prior,
            length=length,
            nu=nu,
            a=a,
            mu=mu,
            eta=eta,
            auto_activation_prob=auto_activation_prob,
            auto_deactivation_prob=auto_deactivation_prob,
        )


class DegreeDynamics(_DataModelWrapper):
    constructor = _dynamics.DegreeDynamics

    def __init__(
        self, graph_prior: RandomGraph = None, length: int = 10, C: float = 10
    ):
        super().__init__(graph_prior=graph_prior, length=length, C=C)
