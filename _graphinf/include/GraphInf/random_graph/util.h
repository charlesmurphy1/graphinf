#ifndef GRAPH_INF_RANDOMGRAPH_UTIL_H
#define GRAPH_INF_RANDOMGRAPH_UTIL_H

#include <string>
#include <memory>
#include "GraphInf/types.h"
#include "prior/edge_count.h"
#include "prior/block_count.h"
#include "prior/block.h"
#include "prior/degree.h"
#include "prior/labeled_degree.h"
#include "likelihood/sbm.h"

#include "GraphInf/proposer/edge/edge_proposer.h"
#include "GraphInf/proposer/edge/single_edge.h"
#include "GraphInf/proposer/edge/hinge_flip.h"
#include "GraphInf/proposer/edge/double_edge_swap.h"

#include "GraphInf/proposer/label/base.hpp"
#include "GraphInf/proposer/label/uniform.hpp"
#include "GraphInf/proposer/label/mixed.hpp"

#include "GraphInf/proposer/nested_label/base.hpp"
#include "GraphInf/proposer/nested_label/uniform.hpp"
#include "GraphInf/proposer/nested_label/mixed.hpp"

namespace GraphInf{

MultiGraph getLabelGraphFromGraph(const MultiGraph& graph, const BlockSequence& blockSeq);

void checkGraphConsistencyWithLabelGraph(
    std::string namePrefix,
    const MultiGraph& graph,
    const BlockSequence& blockSeq,
    const MultiGraph& expectedEdgeMat
);
void checkGraphConsistencyWithDegreeSequence(
    std::string className,
    std::string expName,
    const MultiGraph& graph,
    std::string actName,
    const DegreeSequence& expectedDegreeSeq
);
EdgeCountPrior* makeEdgeCountPrior(double edgeCount, bool canonical=false);
BlockPrior* makeBlockPrior(size_t size, BlockCountPrior& blockCountPrior, bool hyperPrior=false);
LabelGraphPrior* makeLabelGraphPrior(EdgeCountPrior& edgeCountPrior, BlockPrior& blockPrior, bool plantedPrior=false);
DegreePrior* makeDegreePrior(size_t size, EdgeCountPrior& prior, bool hyperPrior=false);
VertexLabeledDegreePrior* makeVertexLabeledDegreePrior(LabelGraphPrior& prior, bool hyperPrior=false);
StochasticBlockModelLikelihood* makeSBMLikelihood(bool stubLabeled=true);
EdgeProposer* makeEdgeProposer(
    std::string proposerType="uniform",
    bool canonical=false,
    bool degreeConstrained=false,
    bool withSelfLoops=true,
    bool withParallelEdges=true
);
LabelProposer<BlockIndex>* makeBlockProposer(
    std::string proposerType="uniform",
    bool restricted=true,
    double sampleLabelCountProb=0.1,
    double labelCreationProb=0.5,
    double shift=1);

NestedLabelProposer<BlockIndex>* makeNestedBlockProposer(
    std::string proposerType="uniform",
    bool restricted=true,
    double sampleLabelCountProb=0.1,
    double labelCreationProb=0.5,
    double shift=1);
}

#endif
