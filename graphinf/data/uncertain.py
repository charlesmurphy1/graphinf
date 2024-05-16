from __future__ import annotations
import numpy as np
from graphinf import _graphinf
from graphinf.graph import RandomGraph
from basegraph import core as bg
from graphinf.wrapper import Wrapper

_uncertain = _graphinf.data.uncertain
from .util import adj_matrix_to_graph
from graphinf.data import DataModelWrapper as _DataModelWrapper

__all__ = (
    "UncertainGraph",
    "PoissonUncertainGraph",
)


class UncertainGraph(_DataModelWrapper):
    def set_state(self, state: np.ndarray | bg.UndirectedMultigraph) -> None:
        if isinstance(state, bg.UndirectedMultigraph):
            self.wrap.set_state(state)
            return

        g = adj_matrix_to_graph(state)
        self.wrap.set_state(g)

    def state(self, to_array: bool = False):
        state = self.wrap.state()
        if to_array:
            return np.array(state.get_adjacency_matrix(True))
        return state


class PoissonUncertainGraph(UncertainGraph):
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
