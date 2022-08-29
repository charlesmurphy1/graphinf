#ifndef GRAPH_INF_PROPOSER_HPP
#define GRAPH_INF_PROPOSER_HPP

#include "GraphInf/types.h"
#include "GraphInf/rv.hpp"

namespace GraphInf{


template<typename MoveType>
class Proposer: public NestedRandomVariable{
    public:
        virtual ~Proposer(){}
        virtual const MoveType proposeMove() const = 0;
        virtual void clear() {};
};

}

#endif
