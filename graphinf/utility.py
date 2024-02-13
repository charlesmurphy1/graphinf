import importlib
from collections import defaultdict
from typing import Optional, Union, Tuple, List
from itertools import combinations_with_replacement

import pandas as pd
from importlib.util import find_spec
import networkx as nx
import numpy as np
from basegraph import core as bg

from graphinf._graphinf.utility import *


def logbase(x: Union[np.ndarray, float], base: float = np.e):
    return np.log(x) / np.log(base)


def to_nary(x, base=2, dim=None):
    if type(x) is int or type(x) is float:
        x = np.array([x])
    if dim is None:
        max_val = base ** np.floor(logbase(np.max(x), base) + 1)
        dim = int(logbase(max_val, base))
    y = np.zeros([dim, *x.shape])
    for idx, xx in np.ndenumerate(x):
        r = np.zeros(dim)
        r0 = xx
        while r0 > 0:
            b = int(np.floor(logbase(r0, base)))
            r[b] += 1
            r0 -= base**b
        y.T[idx] = r[::-1]
    return y


def reduce_partition(p: Union[np.ndarray, float], max_label: Optional[int] = None):
    max_label = np.max(p) if max_label is None else max_label
    b = np.array(p)
    n = np.array([np.sum(b == r) for r in np.arange(max_label + 1)])
    index_map = {}

    s = 0
    for r, _n in enumerate(n):
        if _n > 0:
            index_map[r] = s
            s += 1
    return tuple(index_map[_p] for _p in p)


def enumerate_all_partitions(size: int, block_count: Optional[int] = None, reduce: bool = True):
    B = size if block_count is None else block_count
    s = set()
    for i in range(B**size):
        p = tuple(to_nary(i, B, dim=size).squeeze().astype("int").tolist())
        labels = np.unique(p)
        if block_count is not None and len(labels) != block_count:
            continue

        if reduce:
            p = reduce_partition(p)
            if p in s or (block_count is not None and len(labels) != block_count):
                continue
            s.add(p)
        yield p


def log_sum_exp(x: Union[np.ndarray, float]):
    x = np.array(x)
    b = np.max(x)
    return b + np.log(np.sum(np.exp(x - b)))


def log_mean_exp(x: Union[np.ndarray, float]):
    x = np.array(x)
    b = np.max(x)
    return b + np.log(np.mean(np.exp(x - b)))


class EdgeCollector:
    def __init__(
        self,
        graphs: Optional[Union[List[bg.UndirectedMultigraph], bg.UndirectedMultigraph]] = None,
        epsilon: float = 0.0,
    ):
        self.epsilon = epsilon
        self.clear()
        if graphs is None:
            return
        if isinstance(graphs, list):
            for graph in graphs:
                self.update(graph)
        else:
            self.update(graphs)

    @property
    def total_count(self):
        return self._total_count

    @property
    def node_count(self):
        return self._node_count

    def clear(self):
        self.multiplicities = defaultdict(lambda: defaultdict(int))
        self.counts = defaultdict(int)
        self._total_count = 0
        self._node_count = 0
        self._graph_collection = []

    def update(self, graph: bg.UndirectedMultigraph, keep_graph: bool = False) -> None:
        self._total_count += 1
        self._node_count = max(self.node_count, graph.get_size())
        for edge in graph.edges():
            self.multiplicities[edge][graph.get_edge_multiplicity(*edge)] += 1
            self.counts[edge] += 1
        if keep_graph:
            self._graph_collection.append(graph)

    def mle(self, edge: Tuple[int, int], multiplicity: Optional[int] = None) -> float:
        if edge not in self.counts:
            return 1.0 if multiplicity == 0 else self.epsilon / (self.total_count + self.epsilon)
        p0 = 1 - (self.counts[edge] + self.epsilon) / (self.total_count + self.epsilon)
        if multiplicity == 0:
            return p0
        if multiplicity is None:
            return 1 - p0
        return (self.multiplicities[edge][multiplicity] + self.epsilon) / (self.total_count + self.epsilon)

    def log_prob_estimate(self, graph: bg.UndirectedMultigraph) -> float:
        logp = 0
        for edge in combinations_with_replacement(range(graph.get_size()), 2):
            m = graph.get_edge_multiplicity(*edge)
            if self.epsilon <= 0:
                if edge not in self.counts and m > 0:
                    return -np.inf
                elif edge not in self.counts:
                    continue
            logp += np.log(self.mle(edge, m))
        return logp

    def entropy(self) -> float:
        entropy = 0
        for edge in combinations_with_replacement(range(self.node_count), 2):
            mult = self.multiplicities[edge]
            if self.counts[edge] < self.total_count:
                p = self.mle(edge, 0)
                entropy -= p * np.log(p)
            for m, c in mult.items():
                p = self.mle(edge, m)
                entropy -= p * np.log(p)
        return entropy

    def sample_from_collection(self):
        return self._graph_collection[np.random.randint(len(self._graph_collection))]

    def sample(
        self,
        edge_count: Optional[int] = None,
    ) -> bg.UndirectedMultigraph:
        graph = bg.UndirectedMultigraph(self.node_count)
        assert self.total_count > 0, "No data to sample from."
        if edge_count is not None:
            edges = [e for e in combinations_with_replacement(range(self.node_count), 2) if self.mle(e) > 0]
            weights = np.array([self.mle(e) for e in edges])

            for edge_idx in np.random.choice(len(edges), size=edge_count, p=weights / weights.sum(), replace=True):
                graph.add_edge(*edges[edge_idx])

        else:
            for edge in combinations_with_replacement(range(self.node_count), 2):
                mults = self.multiplicities[edge]
                if len(mults) == 0 and np.random.rand() < self.epsilon:
                    graph.add_edge(*edge)
                    continue
                m = np.array([0] + list(mults.keys()))
                probs = np.array([self.mle(edge, _m) for _m in m])
                probs /= probs.sum(keepdims=True)
                m = np.random.choice(m, p=probs)
                if m > 0:
                    graph.add_multiedge(*edge, m)
        return graph


def load_graph(path: str):
    data = pd.read_pickle(path)
    edgelist = data["edgelist"]
    nodelist = data["nodelist"]
    g = bg.UndirectedMultigraph(max(nodelist) + 1)
    for e in edgelist:
        i, j, m = e
        g.add_multiedge(i, j, m)
    return g


def save_graph(graph, path):
    nodelist = np.array([v for v in graph])
    edgelist = np.array([(*e, graph.get_edge_multiplicity(*e)) for e in graph.edges()])
    pd.to_pickle(dict(nodelist=nodelist, edgelist=edgelist), path)


def convert_basegraph_to_networkx(
    bs_graph: bg.UndirectedMultigraph,
) -> nx.Graph:
    nx_graph = nx.Graph()
    for v in bs_graph:
        nx_graph.add_node(v)
        for u in bs_graph.get_out_neighbours(v):
            if v > u:
                continue
            nx_graph.add_edge(v, u)
    return nx_graph


def convert_basegraph_to_graphtool(bs_graph: bg.UndirectedMultigraph):
    if find_spec("graph_tool") is not None:
        from graph_tool import Graph
    else:
        raise RuntimeError("Could not find `graph_tool`.")

    gt_graph = Graph(directed=False)
    gt_graph.add_vertex(n=bs_graph.get_size())
    for e in bs_graph.edges():
        for m in range(bs_graph.get_edge_multiplicity(*e)):
            gt_graph.add_edge(*e)
    return gt_graph


def convert_graphtool_to_basegraph(gt_graph) -> bg.UndirectedMultigraph:
    bs_graph = bg.UndirectedMultigraph(gt_graph.num_vertices())
    for e in gt_graph.edges():
        bs_graph.add_edge(*e)
    return bs_graph
