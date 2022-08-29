import pytest

from _graphinf.random_graph import ErdosRenyiModel, ConfigurationModelFamily
from graphinf.wrapper import Wrapper


@pytest.fixture
def wrapper():
    erdos = ErdosRenyiModel(10, 10)
    configuration = ConfigurationModelFamily(10, 10)
    return Wrapper(
        erdos,
        configuration=configuration,
    )


def test_access_wrapped_method(wrapper):
    assert wrapper.get_size() == 10
    wrapper.sample()


def test_wrap(wrapper):
    assert isinstance(wrapper.wrap, ErdosRenyiModel)


def test_other(wrapper):
    assert isinstance(wrapper.others["configuration"], ConfigurationModelFamily)
    assert isinstance(wrapper.other("configuration"), ConfigurationModelFamily)
    assert isinstance(wrapper.configuration, ConfigurationModelFamily)


if __name__ == "__main__":
    pass
