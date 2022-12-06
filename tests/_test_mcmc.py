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

data_models = {
    "sis": graphinf.data_model.dynamics.SISDynamics,
    "glauber": graphinf.data_model.dynamics.GlauberDynamics,
    "cowan": graphinf.data_model.dynamics.CowanDynamics,
}


@pytest.mark.parametrize(
    "graph, data",
    [
        pytest.param(v, vv, id=f"{k}-{kk}")
        for (k, v), (kk, vv) in product(random_graphs.items(), data_models.items())
    ],
)
def test_graph_reconstruction(graph, data):
    N, E, T = (10, 25, 5)
    g = graph(N, E)
    d = data(g, length=T)
    mcmc = graphinf.mcmc.GraphReconstructionMCMC(d)
    mcmc.do_MH_sweep(100)


@pytest.mark.parametrize(
    "graph",
    [pytest.param(v, id=k) for k, v in random_graphs.items() if v(2, 2).labeled],
)
def test_partition_reconstruction(graph):
    N, E = 10, 25
    g = graph(10, 25)
    mcmc = graphinf.mcmc.PartitionReconstructionMCMC(g)
    mcmc.do_MH_sweep(100)


if __name__ == "__main__":
    pass
