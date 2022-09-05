#include "gtest/gtest.h"

#include "GraphInf/utility/functions.h"

namespace GraphInf{

TEST(GetPoissonPMF, anyIntegerAndMeanCombination_returnCorrectLogPoissonPMF) {
    for (auto x: {0, 2, 10, 100})
        for (auto mu: {.0001, 1., 10., 1000.})
            EXPECT_DOUBLE_EQ(logPoissonPMF(x, mu), x*log(mu) - lgamma(x+1) - mu);
}

TEST(combinations, listOfIntegers_returnAllCombinations){
    std::list<int> xInt = {1, 2, 3, 4, 5};

    std::list<std::list<int>> cInt;
    cInt = combinations(xInt, 2, false);
    EXPECT_EQ(cInt.size(), 10);
    cInt = combinations(xInt, 2, true);
    EXPECT_EQ(cInt.size(), 15);

    std::list<std::string> xStr = {"a", "b", "c", "d", "e", "f"};

    std::list<std::list<std::string>> cStr;
    cStr = combinations(xStr, 3, false);
    EXPECT_EQ(cStr.size(), 20);
    cStr = combinations(xStr, 3, true);
    EXPECT_EQ(cStr.size(), 56);
}

TEST(enumerateAllGraphs, returnAllGraphs){
    size_t N=3, E=3;
    std::list<MultiGraph> graphs = enumerateAllGraphs(N, E);
    EXPECT_EQ(graphs.size(), 56);
}

}
