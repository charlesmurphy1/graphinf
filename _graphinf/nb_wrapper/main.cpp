#include <nanobind/nanobind.h>
#include "utility/init_utility.h"

namespace nb = nanobind;
namespace GraphInf
{
    NB_MODULE(_graphinf, m)
    {
        nb::module_::import_("basegraph");

        auto utility = m.def_submodule("utility");
        initUtility(utility);
        // initGenerators(utility);
        // initRNG(utility);
        // initExceptions(utility);
    }

}
