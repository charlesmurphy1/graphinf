from __future__ import annotations
from graphinf import _graphinf
from graphinf.graph import RandomGraph
from graphinf.wrapper import Wrapper

_uncertain = _graphinf.data.uncertain
# from .__init__ import DataModelWrapper as _DataModelWrapper
from graphinf.data import DataModelWrapper as _DataModelWrapper

__all__ = (
    "UncertainGraph",
    "PoissonUncertainGraph",
)

UncertainGraph = _uncertain.UncertainGraph


class PoissonUncertainGraph(_DataModelWrapper):
    constructor = _uncertain.PoissonUncertainGraph

    def __init__(
        self,
        prior: RandomGraph | Wrapper = None,
        mu: int = 5,
        mu_no_edge: float = 0.0,
    ):
        super().__init__(
            prior=prior,
            mu=mu,
            mu_no_edge=mu_no_edge,
        )
