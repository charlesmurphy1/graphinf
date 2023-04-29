#ifndef GRAPH_INF_LABELED_EDGE_PROPOSER_H
#define GRAPH_INF_LABELED_EDGE_PROPOSER_H

#include <map>
#include "edge_proposer.h"
#include "GraphInf/graph/proposer/sampler/edge_sampler.h"
#include "GraphInf/graph/proposer/sampler/vertex_sampler.h"
#include "GraphInf/graph/proposer/sampler/label_sampler.h"
#include "GraphInf/utility/functions.h"
#include "GraphInf/utility/maps.hpp"

namespace GraphInf
{

    using LabelPair = std::pair<BlockIndex, BlockIndex>;

    class LabeledEdgeProposer : public EdgeProposer
    {
        // protected:
        //     LabelPairSampler m_labelSampler;
        // public:
        //     LabeledEdgeProposer(
        //         bool allowSelfLoops=true,
        //         bool allowMultiEdges=true,
        //         double labelPairShift=1):
        //             EdgeProposer(allowSelfLoops, allowMultiEdges),
        //             m_labelSampler(labelPairShift){ }
        //     virtual ~LabeledEdgeProposer(){ }
        //     virtual void setUp( const RandomGraph& randomGraph ) {
        //         clear();
        //         m_labelSampler.setUp(randomGraph);
        //         setUpFromGraph(randomGraph.getGraph());
        //     }
        //     virtual void setUpFromGraph(const MultiGraph& graph){
        //         EdgeProposer::setUpFromGraph(graph);
        //     }
        //     virtual void onLabelCreation(const BlockMove& move) { };
        //     virtual void onLabelDeletion(const BlockMove& move) { };
        //     virtual void clear() { m_labelSampler.clear(); }
    };

}

#endif
