#ifndef GRAPH_INF_UNCERTAIN_H
#define GRAPH_INF_UNCERTAIN_H

#include "GraphInf/data/data_model.hpp"

namespace GraphInf{

template<typename GraphPriorType=RandomGraph>
class UncertainGraphModel: public DataModel<GraphPriorType>{
protected:
    virtual void applyGraphMoveToSelf(const GraphMove& move) = 0;
public:
    virtual void sampleState() = 0;
    virtual const double getLogLikelihood() const = 0;
    virtual const double getLogLikelihoodRatioFromGraphMove(const GraphMove& move) const = 0;
};


}

#endif
