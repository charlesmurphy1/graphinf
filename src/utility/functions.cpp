#include <iostream>
#include <math.h>
#include <list>

#include "GraphInf/types.h"
#include "GraphInf/utility/functions.h"
#include "GraphInf/utility/polylog2_integral.h"
#include "GraphInf/exceptions.h"

using namespace std;

namespace GraphInf
{

    const size_t MAX_INTEGER_THRESHOLD = 5000;

    double logFactorial(size_t n)
    {
        return lgamma(n + 1);
        // if (n < MAX_INTEGER_THRESHOLD)
        //     return lgamma(n + 1);
        // else
        //     return 0.5 * sqrt(2 * PI * n) + n * (log(n) - 1);
    }

    double logDoubleFactorial(size_t n)
    {
        size_t k;
        if (n % 2 == 0)
        {
            k = n / 2;
            return k * log(2) + logFactorial(k);
        }
        else
        {
            k = (n + 1) / 2;
            return logFactorial(2 * k) - k * log(2) - logFactorial(k);
        }
    }

    double logBinomialCoefficient(size_t n, size_t k, bool force)
    {

        if (n < k)
        {
            if (force)
                return -INFINITY;
            throw invalid_argument(
                "logBinomialCoefficient: `n` (" + to_string(n) +
                ") must be greater or equal to `k` (" + to_string(k) + ").");
        }
        return logFactorial(n) - logFactorial(k) - logFactorial(n - k);
    }

    double logMultinomialCoefficient(std::list<size_t> sequence)
    {
        size_t sumSequence = 0;
        double sumLFactorialSequence = 0;
        for (size_t element : sequence)
        {
            sumSequence += element;
            sumLFactorialSequence += logFactorial(element);
        }
        return logFactorial(sumSequence) - sumLFactorialSequence;
    }

    double logMultinomialCoefficient(std::vector<size_t> sequence)
    {
        size_t sumSequence = 0;
        double sumLFactorialSequence = 0;
        for (size_t element : sequence)
        {
            sumSequence += element;
            sumLFactorialSequence += logFactorial(element);
        }
        return logFactorial(sumSequence) - sumLFactorialSequence;
    }

    double logMultisetCoefficient(size_t n, size_t k)
    {
        if (n == 0)
        {
            return 0;
        }
        else
        {
            return logBinomialCoefficient(n + k - 1, k);
        }
    }

    double logPoissonPMF(size_t x, double mean)
    {
        return x * log(mean) - logFactorial(x) - mean;
    }

    double logZeroTruncatedPoissonPMF(size_t x, double mean)
    {
        return x * log(mean) - logFactorial(x) - mean - log(1 - exp(-mean));
    }

    BaseGraph::Edge getOrderedEdge(const BaseGraph::Edge &edge)
    {
        if (edge.first < edge.second)
            return edge;
        return {edge.second, edge.first};
    }

    std::list<BaseGraph::Edge> getEdgeList(const MultiGraph &graph)
    {
        std::list<BaseGraph::Edge> edgeList;
        for (const auto &edge : graph.edges())
            edgeList.push_back(edge);
        return edgeList;
    }

    std::map<BaseGraph::Edge, size_t> getWeightedEdgeList(const MultiGraph &graph)
    {
        std::map<BaseGraph::Edge, size_t> edgeList;
        for (const auto &edge : graph.edges())
            edgeList.insert({edge, graph.getEdgeMultiplicity(edge.first, edge.second)});
        return edgeList;
    }

    void assertValidProbability(double probability)
    {
        if (probability > 1 || probability < 0)
            throw ConsistencyError("Probability " + std::to_string(probability) + " is not between 0 and 1.");
    }

    std::pair<size_t, size_t> getUndirectedPairFromIndex(size_t index, size_t n)
    {
        const size_t i = floor(-.5 + sqrt(.25 + 2 * index));
        const size_t j = index - i * (i + 1) * .5;
        return {j, i};
    }

    MultiGraph getSubGraphByBlocks(const MultiGraph &graph, const BlockSequence &blocks, BlockIndex r, BlockIndex s)
    {
        MultiGraph subGraph(graph.getSize());

        for (auto vertex : graph)
        {
            for (auto neighbor : graph.getOutNeighbours(vertex))
            {
                if ((vertex < neighbor) && (blocks[vertex] == r && blocks[neighbor] == s))
                    subGraph.setEdgeMultiplicity(vertex, neighbor, graph.getEdgeMultiplicity(vertex, neighbor));
            }
        }

        return subGraph;
    }

    double clip(double x, double min, double max)
    {
        if (x < min)
            return min;
        else if (x > max)
            return max;
        else
            return x;
    }

    double clipProb(double p, double epsilon) { return clip(p, epsilon, 1 - epsilon); }

    void displayNeighborhood(const MultiGraph &graph, const BaseGraph::VertexIndex &v)
    {
        std::cout << "vertex " << v << ": ";
        for (auto neighbor : graph.getOutNeighbours(v))
            std::cout << "(" << neighbor << ", " << graph.getEdgeMultiplicity(v, neighbor) << ") ";
        std::cout << std::endl;
    }
    void displayGraph(const MultiGraph &graph, std::string name)
    {
        std::cout << name << ":" << std::endl;
        for (auto v : graph)
        {
            std::cout << "\t";
            displayNeighborhood(graph, v);
        }
    }

    std::list<MultiGraph> enumerateAllGraphs(size_t N, size_t E, bool withSelfLoops, bool withParallelEdges)
    {
        std::list<BaseGraph::VertexIndex> allVertices;
        for (auto i = 0; i < N; ++i)
            allVertices.push_back(i);
        std::list<std::list<BaseGraph::VertexIndex>> allEdges = combinations(allVertices, 2, withSelfLoops);
        std::list<std::list<std::list<BaseGraph::VertexIndex>>> allEdgeLists = combinations(allEdges, E, withParallelEdges);

        std::list<MultiGraph> graphs;
        for (auto edges : allEdgeLists)
        {
            MultiGraph g(N);
            for (auto e : edges)
            {
                g.addEdge(e.front(), e.back());
            }
            graphs.push_back(g);
        }
        return graphs;
    }

    int testingBaseGraph()
    {
        BaseGraph::UndirectedMultigraph graph(10);
        graph.addEdge(0, 1);
        graph.addEdge(2, 3);
        graph.addEdge(2, 4);
        graph.addEdge(3, 4);

        BaseGraph::UndirectedMultigraph graph2 = graph;

        std::cout << graph << std::endl;
        std::cout << "The degree of vertex 2 is: " << graph.getDegree(2)
                  << "The degree of vertex 2 is: " << graph2.getDegree(2)
                  << std::endl;
        return 0;
    }

} // namespace GraphInf
