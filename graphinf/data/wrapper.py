from graphinf.wrapper import Wrapper as _Wrapper
from graphinf.graph import (
    RandomGraphWrapper as _RandomGraphWrapper,
    ErdosRenyiModel as _ErdosRenyiModel,
)

class DataModelWrapper(_Wrapper):
    constructor = None

    def __init__(self, graph_prior: _RandomGraphWrapper = None, **kwargs):
        graph_prior = (
            _ErdosRenyiModel(100, 250) if graph_prior is None else graph_prior
        )
        self.labeled = graph_prior.labeled
        self.nested = graph_prior.nested
        data_model = self.constructor(graph_prior.wrap, **kwargs)
        data_model.sample()
        super().__init__(data_model, graph_prior=graph_prior, params=kwargs)

    def __repr__(self):
        str_format = f"{self.__class__.__name__ }(\n\tprior={self.graph_prior.__class__.__name__},"

        for k, v in self.params.items():
            str_format += f"\n\t{k}={v},"
        str_format += "\n)"
        return str_format
    @property
    def dtype(self):
        if self.nested:
            return "nested"
        elif self.labeled:
            return "labeled"
        return "normal"

    def set_graph_prior(self, graph_prior: _RandomGraphWrapper):
        self.labeled = graph_prior.labeled
        self.nested = graph_prior.nested
        self.graph_prior = graph_prior
        self.wrap.set_graph_prior(graph_prior.wrap)
        self.__wrapped__.sample()
