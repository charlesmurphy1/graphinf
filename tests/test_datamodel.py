import pytest
import graphinf

from itertools import product

random_graphs = {
    "erdosrenyi": graphinf.random_graph.ErdosRenyiModel,
    "configuration": graphinf.random_graph.ConfigurationModelFamily,
    "stochastic_block_model": graphinf.random_graph.StochasticBlockModelFamily,
    "nested_stochastic_block_model": lambda N, E: graphinf.random_graph.StochasticBlockModelFamily(
        N, E, label_graph_prior_type="nested"
    ),
}


@pytest.fixture
def graph_prior():
    N, E = 100, 250
    return graphinf.random_graph.ErdosRenyiModel(N, E)


@pytest.mark.parametrize(
    "data, graph",
    [
        pytest.param(
            getattr(graphinf.data_model.dynamics, d), random_graphs[g], id=f"{d}-{g}"
        )
        for d, g in product(graphinf.data_model.dynamics.__all__, random_graphs)
        if issubclass(
            getattr(graphinf.data_model.dynamics, d), graphinf.wrapper.Wrapper
        )
    ],
)
def test_dynamics(data, graph):
    N, E, T = 100, 250, 26
    g = graph(N, E)
    d = data(g, length=T)
    assert d.get_size() == N
    assert d.get_length() == T
    assert d.labeled == g.labeled
    assert d.nested == g.nested


if __name__ == "__main__":
    pass
