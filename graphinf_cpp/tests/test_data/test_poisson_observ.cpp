#include "gtest/gtest.h"

#include "GraphInf/random_graph/erdosrenyi.h"
#include "GraphInf/data/uncertain/poisson.hpp"

namespace GraphInf
{

    class TestUncertainPoissonModel : public ::testing::Test
    {
    public:
        double m_noEdgeAverage = 0.1;
        double m_edgeAverage = 5;
        ErdosRenyiModel prior = ErdosRenyiModel(3, 3);
        MultiGraph graph = MultiGraph(3);
        MultiGraph m_observations = MultiGraph(3);
        UncertainPoissonModel m_model =
            UncertainPoissonModel{prior, m_noEdgeAverage, m_edgeAverage};

        void SetUp()
        {
            graph.addMultiedgeIdx(0, 1, 2);
            graph.addMultiedgeIdx(1, 2, 1);
            prior.setState(graph);

            m_observations.setEdgeMultiplicityIdx(0, 1, 20);
            m_observations.setEdgeMultiplicityIdx(0, 2, 3);
            m_observations.setEdgeMultiplicityIdx(1, 2, 10);
            m_model.setState(m_observations);
        }
    };

    TEST(_TestUncertainPoissonModel, setState_observationsDifferentSize_throwLogicError)
    {
        MultiGraph graph(2), observations(3);
        ErdosRenyiModel prior(3, 3);
        UncertainPoissonModel model = {prior, 0, 0};

        EXPECT_THROW(model.setState(observations), std::logic_error);
    }

    TEST_F(TestUncertainPoissonModel, getAverage_multiplicity0_returnNoEdgeAverage)
    {
        EXPECT_EQ(m_model.getAverage(0), m_noEdgeAverage);
    }

    TEST_F(TestUncertainPoissonModel, getAverage_nonzeroMultiplicity_returnMultiplicityTimesAverage)
    {
        EXPECT_EQ(m_model.getAverage(1), m_edgeAverage);
        EXPECT_EQ(m_model.getAverage(2), 2 * m_edgeAverage);
    }

    static double poissonLogPDF(size_t x, double average)
    {
        return x * log(average) - average - lgamma(x + 1);
    }

    TEST_F(TestUncertainPoissonModel, getLogLikelihood_returnCorrectValue)
    {
        double actual = m_model.getLogLikelihood();
        double expected = 0;

        expected += poissonLogPDF(m_observations.getEdgeMultiplicityIdx(0, 1), m_model.getAverage(graph.getEdgeMultiplicityIdx(0, 1)));
        expected += poissonLogPDF(m_observations.getEdgeMultiplicityIdx(0, 2), m_model.getAverage(graph.getEdgeMultiplicityIdx(0, 2)));
        expected += poissonLogPDF(m_observations.getEdgeMultiplicityIdx(1, 2), m_model.getAverage(graph.getEdgeMultiplicityIdx(1, 2)));

        EXPECT_EQ(actual, expected);
    }

    TEST_F(TestUncertainPoissonModel, computeLogLikelihoodRatioOfPair_addInexistentEdge_returnCorrectValue)
    {
        EXPECT_EQ(m_model.computeLogLikelihoodRatioOfPair(0, 2, 1),
                  m_observations.getEdgeMultiplicityIdx(0, 2) * (log(m_edgeAverage) - log(m_noEdgeAverage)) - m_edgeAverage + m_noEdgeAverage);
    }

    TEST_F(TestUncertainPoissonModel, computeLogLikelihoodRatioOfPair_addExistentEdge_returnCorrectValue)
    {
        EXPECT_EQ(m_model.computeLogLikelihoodRatioOfPair(1, 2, 1),
                  m_observations.getEdgeMultiplicityIdx(1, 2) * (log(2 * m_edgeAverage) - log(m_edgeAverage)) - m_edgeAverage);
    }

    TEST_F(TestUncertainPoissonModel, computeLogLikelihoodRatioOfPair_removeEdge_returnCorrectValue)
    {
        EXPECT_EQ(m_model.computeLogLikelihoodRatioOfPair(1, 2, 0),
                  m_observations.getEdgeMultiplicityIdx(1, 2) * (log(m_noEdgeAverage) - log(m_edgeAverage)) - m_noEdgeAverage + m_edgeAverage);
    }

    TEST_F(TestUncertainPoissonModel, computeLogLikelihoodRatioOfPair_removeEdgeFromMultiedge_returnCorrectValue)
    {
        EXPECT_EQ(m_model.computeLogLikelihoodRatioOfPair(0, 1, 0),
                  m_observations.getEdgeMultiplicityIdx(0, 1) * (log(m_edgeAverage) - log(2 * m_edgeAverage)) + m_edgeAverage);
    }

    TEST_F(TestUncertainPoissonModel, getLogLikelihoodRatioFromGraphMove_returnSumOfLogRatios)
    {
        GraphMove move({{1, 2}}, {{0, 1}});
        EXPECT_EQ(m_model.getLogLikelihoodRatioFromGraphMove(move),
                  m_model.computeLogLikelihoodRatioOfPair(1, 2, 0) + m_model.computeLogLikelihoodRatioOfPair(0, 1, 1));
    }

} // namespace GraphInf
