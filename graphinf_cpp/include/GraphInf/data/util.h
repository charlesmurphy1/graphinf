#ifndef GRAPH_INF_DYNAMICS_UTIL_H
#define GRAPH_INF_DYNAMICS_UTIL_H

#include <cmath>

namespace GraphInf{

static inline double sigmoid(double x) {
    return 1./(1.+exp(-x));
}

}

#endif
