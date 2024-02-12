import pytest

from graphinf.wrapper import Wrapper
from graphinf._graphinf import graph as _graph


@pytest.fixture
def wrapper():
    erdos = _graph.ErdosRenyiModel(10, 10)
    configuration = _graph.ConfigurationModelFamily(10, 10)
    return Wrapper(
        erdos,
        configuration=configuration,
    )


def test_access_wrapped_method(wrapper):
    assert wrapper.size() == 10
    wrapper.sample()


def test_wrap(wrapper):
    assert isinstance(wrapper.wrap, _graph.ErdosRenyiModel)


def test_other(wrapper):
    assert isinstance(wrapper.others["configuration"], _graph.ConfigurationModelFamily)
    assert isinstance(wrapper.other("configuration"), _graph.ConfigurationModelFamily)
    assert isinstance(wrapper.configuration, _graph.ConfigurationModelFamily)


if __name__ == "__main__":
    pass
