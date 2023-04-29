#ifndef GRAPH_INF_LIKELIHOOD_CONFIGURATION_H
#define GRAPH_INF_LIKELIHOOD_CONFIGURATION_H

#include "BaseGraph/types.h"
#include "GraphInf/graph/likelihood/likelihood.hpp"
#include "GraphInf/graph/prior/degree.h"
#include "GraphInf/utility/maps.hpp"
#include "GraphInf/generators.h"
#include "GraphInf/types.h"

namespace GraphInf
{

    class ConfigurationModelLikelihood : public GraphLikelihoodModel
    {
    public:
        const MultiGraph sample() const override
        {
            return generateCM((*m_degreePriorPtrPtr)->getState());
        }
        const double getLogLikelihood() const override;
        const double getLogLikelihoodRatioFromGraphMove(const GraphMove &move) const override;
        DegreePrior **m_degreePriorPtrPtr = nullptr;
    };

}

#endif
