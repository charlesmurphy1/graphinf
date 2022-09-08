from __future__ import annotations

from _graphinf.data import dynamics as _dynamics
from _graphinf.data.dynamics import (
    Dynamics,
    BlockLabeledDynamics,
    NestedBlockLabeledDynamics,
)
from .__init__ import DataModelWrapper as _DataModelWrapper

__all__ = (
    "Dynamics",
    "BlockLabeledDynamics",
    "NestedBlockLabeledDynamics",
    "SISDynamics",
    "GlauberDynamics",
    "CowanDynamics",
)


class SISDynamics(_DataModelWrapper):
    constructors = {
        "normal": _dynamics.SISDynamics,
        "labeled": _dynamics.BlockLabeledSISDynamics,
        "nested": _dynamics.NestedBlockLabeledSISDynamics,
    }

    def __init__(
        self,
        graph_prior: Union[RandomGraph, Wrapper] = None,
        length: int = 10,
        infection_prob: float = 1,
        recovery_prob: float = 0.5,
        past_length: int = 0,
        auto_activation_prob: float = 1e-06,
        auto_deactivation_prob: float = 0.0,
        normalize: bool = True,
        async_mode: bool = False,
        num_active: int = 1,
    ):
        super().__init__(
            graph_prior=graph_prior,
            length=length,
            infection_prob=infection_prob,
            recovery_prob=recovery_prob,
            past_length=past_length,
            auto_activation_prob=auto_activation_prob,
            auto_deactivation_prob=auto_deactivation_prob,
            normalize=normalize,
            async_mode=async_mode,
            num_active=num_active,
        )


class GlauberDynamics(_DataModelWrapper):
    constructors = {
        "normal": _dynamics.GlauberDynamics,
        "labeled": _dynamics.BlockLabeledGlauberDynamics,
        "nested": _dynamics.NestedBlockLabeledGlauberDynamics,
    }

    def __init__(
        self,
        graph_prior: RandomGraph = None,
        length: int = 10,
        coupling: float = 1,
        past_length: int = 0,
        auto_activation_prob: float = 0.0,
        auto_deactivation_prob: float = 0.0,
        normalize: bool = True,
        async_mode: bool = False,
        num_active: int = -1,
    ):
        super().__init__(
            graph_prior=graph_prior,
            length=length,
            coupling=coupling,
            past_length=past_length,
            auto_activation_prob=auto_activation_prob,
            auto_deactivation_prob=auto_deactivation_prob,
            normalize=normalize,
            async_mode=async_mode,
            num_active=num_active,
        )


class CowanDynamics(_DataModelWrapper):
    constructors = {
        "normal": _dynamics.CowanDynamics,
        "labeled": _dynamics.BlockLabeledCowanDynamics,
        "nested": _dynamics.NestedBlockLabeledCowanDynamics,
    }

    def __init__(
        self,
        graph_prior: RandomGraph = None,
        length: int = 10,
        nu: float = 1,
        a: float = 1,
        mu: float = 1,
        eta: float = 0.5,
        past_length: int = 0,
        auto_activation_prob: float = 0.0,
        auto_deactivation_prob: float = 0.0,
        normalize: bool = True,
        async_mode: bool = False,
        num_active: int = -1,
    ):
        super().__init__(
            graph_prior=graph_prior,
            length=length,
            nu=nu,
            a=a,
            mu=mu,
            eta=eta,
            past_length=past_length,
            auto_activation_prob=auto_activation_prob,
            auto_deactivation_prob=auto_deactivation_prob,
            normalize=normalize,
            async_mode=async_mode,
            num_active=num_active,
        )


class DegreeDynamics(_DataModelWrapper):
    constructors = {
        "normal": _dynamics.DegreeDynamics,
        "labeled": _dynamics.BlockLabeledDegreeDynamics,
        "nested": _dynamics.NestedBlockLabeledDegreeDynamics,
    }

    def __init__(
        self, graph_prior: RandomGraph = None, length: int = 10, C: float = 10
    ):
        super().__init__(graph_prior=graph_prior, length=length, C=C)
