#include "gtest/gtest.h"

#include "GraphInf/utility/functions.h"
#include "GraphInf/graph/erdosrenyi.h"
#include "GraphInf/graph/configuration.h"
#include "GraphInf/data/dynamics/glauber.h"
#include "BaseGraph/undirected_multigraph.hpp"

int meanCount(const std::vector<int> &vec)
{
    int sum = 0;
    for (auto i : vec)
        sum += i;
    return sum / vec.size();
}
namespace GraphInf
{
    TEST(MiscTest, test)
    {

        seed(time(NULL));
        auto graph = ConfigurationModelFamily(105, 441, false, false);
        graph.sample();
        for (size_t i = 0; i < 1000; i++)
        {
            std::cout << i << std::endl;
            graph.metropolisGraphSweep(100000);
            graph.checkConsistency();
        }
    }
}