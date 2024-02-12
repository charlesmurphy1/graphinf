import pytest
import graphinf

from itertools import product

graphs = {
    "erdosrenyi": graphinf.graph.ErdosRenyiModel,
    "configuration": graphinf.graph.ConfigurationModelFamily,
    "stochastic_block_model": graphinf.graph.StochasticBlockModelFamily,
    "nested_stochastic_block_model": lambda N, E: graphinf.graph.StochasticBlockModelFamily(
        N, E, label_graph_prior_type="nested"
    ),
}


@pytest.fixture
def graph_prior():
    N, E = 100, 250
    return graphinf.graph.ErdosRenyiModel(N, E)


@pytest.mark.parametrize(
    "data, graph",
    [
        pytest.param(getattr(graphinf.data.dynamics, d), graphs[g], id=f"{d}-{g}")
        for d, g in product(graphinf.data.dynamics.__all__, graphs)
        if issubclass(getattr(graphinf.data.dynamics, d), graphinf.wrapper.Wrapper)
    ],
)
def test_dynamics(data, graph):
    if data.constructor is None:
        return
    N, E, T = 100, 250, 26
    g = graph(N, E)
    d = data(g, length=T)
    assert d.size() == N
    assert d.length() == T
    assert d.labeled == g.labeled
    assert d.nested == g.nested


if __name__ == "__main__":
    pass
