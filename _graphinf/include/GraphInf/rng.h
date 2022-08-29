#ifndef GRAPH_INF_RNG_H
#define GRAPH_INF_RNG_H


#include <random>
#include "GraphInf/types.h"


namespace GraphInf {

extern RNG rng;
extern size_t SEED;

void seed(size_t n);
void seedWithTime();
const size_t& getSeed();

} // namespace GraphInf

#endif
