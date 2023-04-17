#ifndef GRAPHINF_NBWRAPPER_INIT_UTILITY_H
#define GRAPHINF_NBWRAPPER_INIT_UTILITY_H

#include <nanobind/nanobind.h>

#include "init_maps.h"
#include "init_functions.h"
// #include "init_integerpartition.h"
// #include "init_distance.h"

namespace nb = nanobind;
namespace GraphInf
{

    void initUtility(nb::module_ &m)
    {
        initMaps(m);
        initFunctions(m);
        // initIntegerPartition(m);
        // initDistances(m);
    }

}

#endif
