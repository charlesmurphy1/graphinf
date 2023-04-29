#ifndef GRAPH_INF_UNIFORM_H
#define GRAPH_INF_UNIFORM_H

#include <map>
#include <utility>
#include <vector>

#include "BaseGraph/types.h"
#include "GraphInf/graph/erdosrenyi.h"

namespace GraphInf
{

    // class UniformMultiGraphFamily: public ErdosRenyiFamily{
    // protected:
    //     EdgeCountMultisetPrior m_edgeCountMultisetPrior;
    // public:
    //     UniformMultiGraphFamily(size_t graphSize, size_t maxEdgeCount):
    //     m_edgeCountMultisetPrior(maxEdgeCount),
    //     ErdosRenyiFamily(graphSize, m_edgeCountMultisetPrior){ }
    // };
    //
    // class UniformSimpleGraphFamily: public ErdosRenyiFamily{
    // protected:
    //     EdgeCountBinomialPrior m_edgeCountBinomialPrior;
    // public:
    //     UniformSimpleGraphFamily(size_t graphSize, size_t maxEdgeCount):
    //     m_edgeCountBinomialPrior(maxEdgeCount),
    //     ErdosRenyiFamily(graphSize, m_edgeCountBinomialPrior){ }
    // };

} // end GraphInf
#endif
