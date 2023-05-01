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
        self.wrap.set_graph_prior(graph_prior)
        self.__wrapped__.sample()
