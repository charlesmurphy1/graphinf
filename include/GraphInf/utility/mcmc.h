#ifndef GRAPHINF_CALLBACK_H
#define GRAPHINF_CALLBACK_H

#include <string>
namespace GraphInf
{
    struct MCMCSummary
    {
        std::string move;
        double acceptProb;
        bool isAccepted;
    };

}
#endif