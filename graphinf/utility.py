import importlib
from collections import defaultdict
from itertools import combinations_with_replacement

import networkx as nx
import numpy as np
from basegraph import core as bs

from graphinf._graphinf.utility import *


def logbase(x, base=np.e):
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


def reduce_partition(p, max_label=None):
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


def enumerate_all_partitions(size, block_count=None, reduce=True):
    B = size if block_count is None else block_count
    s = set()
    for i in range(B**size):
        p = tuple(to_nary(i, B, dim=size).squeeze().astype("int").tolist())
        labels = np.unique(p)
        if block_count is not None and len(labels) != block_count:
            continue

        if reduce:
            p = reduce_partition(p)
            if p in s or (
                block_count is not None and len(labels) != block_count
            ):
                continue
            s.add(p)
        yield p


def log_sum_exp(x):
    x = np.array(x)
    b = np.max(x)
    return b + np.log(np.sum(np.exp(x - b)))


def log_mean_exp(x):
    x = np.array(x)
    b = np.max(x)
    return b + np.log(np.mean(np.exp(x - b)))


class EdgeCollector:
    def __init__(self):
        self.multiplicities = defaultdict(lambda: defaultdict(int))
        self.counts = defaultdict(int)
        self.total_count = 0

    def update(self, graph: bs.UndirectedMultigraph):
        self.total_count += 1
        for edge in graph.edges():
            self.multiplicities[edge][graph.get_edge_multiplicity(*edge)] += 1
            self.counts[edge] += 1

    def mle(self, edge, multiplicity=1):
        if self.counts[edge] == 0:
            return 0

        if multiplicity == 0:
            return 1 - self.counts[edge] / self.total_count
        return self.multiplicities[edge][multiplicity] / self.total_count

    def log_prob_estimate(self, graph):
        logp = 0
        for edge in combinations_with_replacement(range(graph.get_size()), 2):
            m = graph.get_edge_multiplicity(*edge)
            if edge not in self.counts and m > 0:
                return -np.inf
            if edge not in self.counts and m == 0:
                continue
            logp += np.log(self.mle(edge, m))
        return logp

    def entropy(self):
        entropy = 0
        for e, mult in self.multiplicities.items():
            if self.counts[e] < self.total_count:
                p = self.mle(e, 0)
                entropy -= p * np.log(p)
            for m, c in mult.items():
                p = self.mle(e, m)
                entropy -= p * np.log(p)
        return entropy


def convert_basegraph_to_networkx(
    bs_graph: bs.UndirectedMultigraph,
) -> nx.Graph:
    nx_graph = nx.Graph()
    for v in bs_graph:
        nx_graph.add_node(v)
        for u in bs_graph.get_out_neighbours(v):
            if v > u:
                continue
            nx_graph.add_edge(v, u)
    return nx_graph


def convert_basegraph_to_graphtool(bs_graph: bs.UndirectedMultigraph):
    if importlib.find_spec("graph_tool"):
        import graph_tool.all as gt
    else:
        raise RuntimeError("Could not find `graph_tool`.")

    gt_graph = gt.Graph(directed=False)
    for e in bs_graph.edges():
        for m in range(bs_graph.get_edge_multiplicity(*e)):
            gt_graph.add_edge(*e)
    return gt_graph


def convert_graphtool_to_basegraph(gt_graph) -> bs.UndirectedMultigraph:
    bs_graph = bs.UndirectedMultigraph(gt_graph.num_vertices())
    for e in gt_graph.edges():
        bs_graph.add_edge(*e)
    return bs_graph
