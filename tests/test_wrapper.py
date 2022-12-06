import pytest

from _graphinf import random_graph as _random_graph
from graphinf.wrapper import Wrapper


@pytest.fixture
def wrapper():
    erdos = _random_graph.ErdosRenyiModel(10, 10)
    configuration = _random_graph.ConfigurationModelFamily(10, 10)
    return Wrapper(
        erdos,
        configuration=configuration,
    )


def test_access_wrapped_method(wrapper):
    assert wrapper.get_size() == 10
    wrapper.sample()


def test_wrap(wrapper):
    assert isinstance(wrapper.wrap, _random_graph.ErdosRenyiModel)


def test_other(wrapper):
    assert isinstance(wrapper.others["configuration"], _random_graph.ConfigurationModelFamily)
    assert isinstance(wrapper.other("configuration"), _random_graph.ConfigurationModelFamily)
    assert isinstance(wrapper.configuration, _random_graph.ConfigurationModelFamily)


if __name__ == "__main__":
    pass
