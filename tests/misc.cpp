#include "gtest/gtest.h"

#include "GraphInf/utility/functions.h"
#include "GraphInf/graph/erdosrenyi.h"
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
        auto uniform01 = std::uniform_real_distribution<double>(0, 1);
        auto graph = ErdosRenyiModel(100, 100, true, true, true);
        graph.setGraphMoveType("single_edge");
        std::cout << graph.getSingleEdgeProposer().getEdgeSampler().getTotalWeight() << std::endl;

        CounterMap<std::string> counter;
        for (size_t i = 0; i < 1000; i++)
        {
            auto move = graph.proposeGraphMove();
            if (move.addedEdges.size() == 1)
            {
                counter.increment("add");
            }
            else
            {
                counter.increment("remove");
            }
        }
        std::cout << "add: " << counter["add"] << " remove: " << counter["remove"] << std::endl;
    }
}