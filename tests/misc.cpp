#include "gtest/gtest.h"

#include "GraphInf/utility/functions.h"
#include "GraphInf/graph/erdosrenyi.h"
#include "GraphInf/data/dynamics/glauber.h"

namespace GraphInf
{
    TEST(MiscTest, test)
    {
        seed(time(NULL));
        auto graph = ErdosRenyiModel(100, 100, true, true, true);
        auto model = GlauberDynamics(graph, 100, 1.0);
        model.sample();
        model.metropolisGraphSweep(10000);
    }
}