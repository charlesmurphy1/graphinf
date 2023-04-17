#ifndef GRAPH_INF_UNCERTAIN_H
#define GRAPH_INF_UNCERTAIN_H

#include "GraphInf/data/data_model.h"

namespace GraphInf
{

    class UncertainGraphModel : public DataModel
    {
    protected:
        MultiGraph m_state;
        virtual void applyGraphMoveToSelf(const GraphMove &move) = 0;

    public:
        UncertainGraphModel(RandomGraph &prior) : DataModel(prior) {}
        virtual void sampleState() = 0;
        virtual const double getLogLikelihood() const = 0;
        virtual const double getLogLikelihoodRatioFromGraphMove(const GraphMove &move) const = 0;
        void setState(const MultiGraph &observations) { m_state = observations; }
        const MultiGraph &getState() const { return m_state; }
    };

}

#endif
