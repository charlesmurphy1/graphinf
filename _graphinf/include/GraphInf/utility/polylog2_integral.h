#ifndef GRAPH_INF_POLYLOG2_INTEGRAL_H
#define GRAPH_INF_POLYLOG2_INTEGRAL_H

#include "GraphInf/types.h"

namespace GraphInf{

    // static double MACHEP = 1.11022302462515654042E-16;    /* 2**-53 */
    static double PI = 3.14159265358979323846264338;

    static double A[8] = {
        4.65128586073990045278E-5,
        7.31589045238094711071E-3,
        1.33847639578309018650E-1,
        8.79691311754530315341E-1,
        2.71149851196553469920E0,
        4.25697156008121755724E0,
        3.29771340985225106936E0,
        1.00000000000000000126E0,
    };

    static double B[8] = {
        6.90990488912553276999E-4,
        2.54043763932544379113E-2,
        2.82974860602568089943E-1,
        1.41172597751831069617E0,
        3.63800533345137075418E0,
        5.03278880143316990390E0,
        3.54771340985225096217E0,
        9.99999999999999998740E-1,
    };
    double polylog2Integral(double x);

}
#endif // GRAPH_INF_POLYLOG2_INTEGRAL_H
