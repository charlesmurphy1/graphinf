#include "gtest/gtest.h"

#include "GraphInf/utility/functions.h"


TEST(GetPoissonPMF, anyIntegerAndMeanCombination_returnCorrectLogPoissonPMF) {
    for (auto x: {0, 2, 10, 100})
        for (auto mu: {.0001, 1., 10., 1000.})
            EXPECT_DOUBLE_EQ(GraphInf::logPoissonPMF(x, mu),
                                x*log(mu) - lgamma(x+1) - mu);
}
