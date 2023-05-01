#ifndef GRAPH_INF_UTIL_FUNCTIONS_H
#define GRAPH_INF_UTIL_FUNCTIONS_H

#include <list>
#include <vector>
#include <utility>
#include <iostream>
#include <algorithm>
#include "GraphInf/types.h"
#include "GraphInf/exceptions.h"

namespace GraphInf
{

    // static const double INFINITY = std::numeric_limits<double>::infinity();

    double logFactorial(size_t);
    double logDoubleFactorial(size_t);
    double logBinomialCoefficient(size_t, size_t, bool force = true);
    double logPoissonPMF(size_t x, double mean);
    double logZeroTruncatedPoissonPMF(size_t x, double mean);
    double logMultinomialCoefficient(std::list<size_t> sequence);
    double logMultinomialCoefficient(std::vector<size_t> sequence);
    double logMultisetCoefficient(size_t n, size_t k);

    double logRestrictedPartitionNumber(size_t n, size_t k);
    double logRestrictedPartitionNumber(size_t n, size_t k);
    double logApproxRestrictedPartitionNumber(size_t n, size_t k);

    MultiGraph getSubGraphByBlocks(const MultiGraph &graph, const BlockSequence &blocks, BlockIndex r, BlockIndex s);
    double clip(double x, double min, double max);
    double clipProb(double p, double epsilon = 1e-15);

    std::list<MultiGraph> enumerateAllGraphs(size_t N, size_t E, bool withSelfLoops = true, bool withParallelEdges = true);

    template <typename ContainerType>
    double logSumExp(ContainerType xSeq)
    {
        auto xMax = *std::max_element(xSeq.begin(), xSeq.end());
        double s = 0;
        for (auto x : xSeq)
            s += exp(x - xMax);
        return log(s) + xMax;
    }

    template <typename ContainerType>
    double logMeanExp(ContainerType xSeq)
    {
        auto N = xSeq.size();
        return logSumExp(xSeq) - log(N);
    }

    template <typename T>
    std::list<std::list<T>> combinations(const std::list<T> elements, size_t repeat, bool withReplacement = false, std::list<T> prefix = {})
    {
        std::list<std::list<T>> combs;
        if (repeat == 1)
        {
            for (auto x : elements)
            {
                std::list<T> _prefix = prefix;
                _prefix.push_back(x);
                combs.push_back(_prefix);
            }
            return combs;
        }
        std::list<T> subElements = elements;
        for (auto x : elements)
        {
            if (not withReplacement)
                subElements.pop_front();
            std::list<T> _prefix = prefix;

            _prefix.push_back(x);

            for (std::list<T> o : combinations(subElements, repeat - 1, withReplacement, _prefix))
                combs.push_back(o);

            if (withReplacement)
                subElements.pop_front();
        }
        return combs;
    }

    template <typename T>
    std::vector<T> listToVec(std::list<T> other)
    {
        std::vector<T> myVec;
        for (auto x : other)
            myVec.push_back(x);
        return myVec;
    }

    template <typename T>
    std::list<T> vecToList(std::vector<T> other)
    {
        std::list<T> myList;
        for (auto x : other)
            myList.push_back(x);
        return myList;
    }

    template <typename T>
    std::pair<T, T> getOrderedPair(const std::pair<T, T> &myPair)
    {
        if (myPair.first < myPair.second)
            return myPair;
        return {myPair.second, myPair.first};
    }
    BaseGraph::Edge getOrderedEdge(const BaseGraph::Edge &);
    inline size_t choose2(size_t n) { return n * (n - 1) / 2; }
    std::pair<size_t, size_t> getUndirectedPairFromIndex(size_t index, size_t n);

    std::list<BaseGraph::Edge> getEdgeList(const MultiGraph &graph);
    std::map<BaseGraph::Edge, size_t> getWeightedEdgeList(const MultiGraph &graph);

    // MultiGraph convertMatrixToMultiGraph(Matrix<size_t> matrix){
    //     MultiGraph graph(matrix.size());
    //
    //     for (size_t i=0; i<matrix.size(); ++i){
    //         if (matrix[i][i] > 0)
    //             graph.addMultiedgeIdx(i, i, matrix[i][i]/2);
    //         for (size_t j=i + 1; j<matrix[i].size(); ++j){
    //             if (matrix[i][j] > 0)
    //                 graph.addMultiedgeIdx(i, j, matrix[i][j]);
    //         }
    //     }
    //     return graph;
    // }

    template <typename T>
    T sumElementsOfMatrix(Matrix<T> mat, T init)
    {
        T sum = init;
        for (auto rows = mat.begin(); rows != mat.end(); rows++)
        {
            for (auto cols = rows->begin(); cols != rows->end(); cols++)
            {
                sum += *cols;
            }
        }
        return sum;
    }

    template <typename T>
    static void verifyHasSize(
        const T &actual,
        const T &expected,
        const std::string &className,
        const std::string &vectorName,
        const std::string &sizeName)
    {
        if (actual != expected)
            throw ConsistencyError(className + ": " + vectorName + " has size " +
                                   std::to_string(actual) + " while there are " +
                                   std::to_string(expected) + " " + sizeName + ".");
    }

    template <typename T>
    static void verifyHasAtLeastSize(
        const T &actual,
        const T &expected,
        const std::string &className,
        const std::string &vectorName,
        const std::string &sizeName)
    {
        if (actual != expected)
            throw ConsistencyError(className + ": " + vectorName + " has size " +
                                   std::to_string(actual) + " while there are " +
                                   std::to_string(expected) + " " + sizeName + ".");
    }

    template <typename T>
    std::string displayMatrix(const Matrix<T> &matrix, std::string name = "m", bool toConsole = false)
    {
        std::stringstream ss;
        ss << name << " = [" << std::endl;
        for (auto row : matrix)
        {
            ss << "  [ ";
            for (auto col : row)
            {
                ss << std::to_string(col) << ", ";
            }
            ss << "]," << std::endl;
        }
        ss << "]";

        if (toConsole)
            std::cout << ss.str() << std::endl;
        return ss.str();
    }
    template <typename T>
    std::string displayVector(const std::vector<T> &vec, std::string name = "v", bool toConsole = false)
    {
        std::stringstream ss;
        ss << name << " = [ ";
        for (auto row : vec)
        {
            ss << std::to_string(row) << ", ";
        }
        ss.seekp(-1, ss.cur);
        ss.seekp(-1, ss.cur);
        ss << "]";
        if (toConsole)
            std::cout << ss.str() << std::endl;

        return ss.str();
    }

    template <typename T>
    std::string displaySet(const std::set<T> &s, std::string name = "s", bool toConsole = false)
    {
        std::stringstream ss;
        ss << name << " = {";
        for (auto row : s)
        {
            ss << std::to_string(row) << ", ";
        }
        ss.seekp(-1, ss.cur);
        ss.seekp(-1, ss.cur);
        ss << "}";
        if (toConsole)
            std::cout << ss.str() << std::endl;

        return ss.str();
    }

    void displayNeighborhood(const MultiGraph &, const BaseGraph::VertexIndex &);
    void displayGraph(const MultiGraph &graph, std::string name = "g");

    template <typename T1, typename T2>
    std::string pairToString(const std::pair<T1, T2> &p)
    {
        std::stringstream ss;
        ss << "<" << p.first << ", " << p.second << ">" << std::endl;
        return ss.str();
    }

} // namespace GraphInf

#endif
