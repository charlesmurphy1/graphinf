#ifndef GRAPHINF_PYTHON_DATA_PROPOSER_HPP
#define GRAPHINF_PYTHON_DATA_PROPOSER_HPP

#include "pybind11/pybind11.h"
#include "GraphInf/data/proposer.h"

namespace GraphInf
{

    template <typename BaseClass = ParamProposer>
    class PyParamProposer : public BaseClass
    {
    public:
        using BaseClass::BaseClass;
        ~PyParamProposer() override = default;
        double proposeMove() const override
        {
            PYBIND11_OVERRIDE_PURE(double, BaseClass, proposeMove, );
        }
        double logProposal(const double move) const override
        {
            PYBIND11_OVERRIDE_PURE(double, BaseClass, logProposal, move);
        }
    };

}
#endif