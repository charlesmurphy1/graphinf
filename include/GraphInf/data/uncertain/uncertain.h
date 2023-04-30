#ifndef GRAPH_INF_UNCERTAIN_H
#define GRAPH_INF_UNCERTAIN_H

#include "GraphInf/data/data_model.h"

namespace GraphInf
{

    class UncertainGraph : public DataModel
    {
    protected:
        MultiGraph m_state;
        virtual void applyGraphMoveToSelf(const GraphMove &move) = 0;

    public:
        using DataModel::DataModel;
        void sample()
        {
            samplePrior();
            sampleState();
            computationFinished();
        }
        virtual void sampleState() = 0;
        virtual const double getLogLikelihood() const = 0;
        virtual const double getLogLikelihoodRatioFromGraphMove(const GraphMove &move) const = 0;
        void setState(const MultiGraph &observations)
        {
            if (observations.getSize() > m_graphPriorPtr->getSize())
                throw std::logic_error("State with size " + std::to_string(observations.getSize()) + " cannot be larger than graph with size " + std::to_string(m_graphPriorPtr->getSize()) + ".");
            m_state = observations;
            if (m_state.getSize() < m_graphPriorPtr->getSize())
                m_state.resize(m_graphPriorPtr->getSize());
        }
        const MultiGraph &getState() const { return m_state; }
    };

}

#endif
