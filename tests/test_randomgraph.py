import pytest
import graphinf
from itertools import product


def test_erdosrenyi():
    N, E = 100, 250
    g = graphinf.random_graph.ErdosRenyiModel(N, E)
    g.sample()
    assert g.get_size() == N
    assert g.get_edge_count() == E


@pytest.mark.parametrize(
    "prior_type",
    [pytest.param(p, id=p) for p in ["uniform", "hyper"]],
)
def test_configuration(prior_type):
    N, E = 100, 250
    g = graphinf.random_graph.ConfigurationModelFamily(
        N, E, degree_prior_type=prior_type
    )
    assert g.get_size() == N
    assert g.get_edge_count() == E


def test_poisson():
    N, E = 100, 250
    g = graphinf.random_graph.PoissonModel(N, E)
    assert g.get_size() == N


def test_nbinom():
    N, E, h = 100, 250, 1
    g = graphinf.random_graph.NegativeBinomialModel(N, E, h)
    assert g.get_size() == N


@pytest.mark.parametrize(
    "block_prior_type, label_graph_prior_type, likelihood_type",
    [
        pytest.param(b, e, l, id=f"{b}-{e}-{l}")
        for (b, e, l) in product(
            ["uniform", "hyper"],
            # ["uniform", "planted", "nested"],
            ["uniform", "nested"],
            ["uniform", "stub_labeled"],
        )
    ],
)
def test_stochastic_block_model(
    block_prior_type, label_graph_prior_type, likelihood_type
):
    N, E = 100, 250
    g = graphinf.random_graph.StochasticBlockModelFamily(
        N,
        E,
        block_prior_type=block_prior_type,
        label_graph_prior_type=label_graph_prior_type,
        likelihood_type=likelihood_type,
    )
    assert g.get_size() == N
    assert g.get_edge_count() == E


@pytest.mark.parametrize(
    "block_prior_type, label_graph_prior_type",
    [
        pytest.param(b, e, id=f"{b}-{e}")
        for (b, e) in product(
            ["uniform", "hyper"],
            ["uniform", "nested"],
            # ["uniform", "planted", "nested"],
        )
    ],
)
def test_degree_corrected_stochastic_block_model(
    block_prior_type, label_graph_prior_type
):
    N, E = 100, 250
    g = graphinf.random_graph.StochasticBlockModelFamily(
        N,
        E,
        block_prior_type=block_prior_type,
        label_graph_prior_type=label_graph_prior_type,
        likelihood_type="degree_corrected",
    )
    g.sample()
    assert g.get_size() == N
    assert g.get_edge_count() == E


# @pytest.mark.parametrize(
#     "stub_labeled", [pytest.param(s) for s in [True, False]]
# )
# def test_planted_partition(stub_labeled):
#     N, E, B = 100, 250, 5
#     g = graphinf.random_graph.PlantedPartitionModel(
#         N, E, B, stub_labeled=stub_labeled
#     )
#     assert g.get_size() == N
#     assert g.get_edge_count() == E
#     assert g.get_label_count() == B


if __name__ == "__main__":
    pass
