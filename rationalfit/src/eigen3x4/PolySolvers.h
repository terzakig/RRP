//
// Analytical polynomial solvers. Implementation by Manolis Lourakis (FORTH)
//
//

#ifndef __POLYSOLVERS_H__
#define __POLYSOLVERS_H__

#include <vector>
#include <math.h>

// #include <iostream>

namespace PolySolvers
{

    // see http://mathworld.wolfram.com/QuadraticFormula.html
    template <typename P>
    inline std::vector<P> SolveQuadratic(P a, P b, P c)
    {
        P delta = b * b - 4 * a * c;
        P inv_2a, sqrt_delta;
        P x1, x0;

        if (delta < 0)
            return std::vector<P>();
        if (a == 0)
        {
            // solve first order system
            x1 = 0;
            if (b != 0)
            {
                x0 = -c / b;
                return std::vector<P>({x0});
            }

            x0 = 0;
            return std::vector<P>();
        }

        inv_2a = 0.5 / a;
        if (delta == 0)
        {
            x0 = -b * inv_2a;
            x1 = x0;
            return std::vector<P>({x0});
        }

        sqrt_delta = sqrt(delta);
        x0 = (-b + sqrt_delta) * inv_2a;
        x1 = (-b - sqrt_delta) * inv_2a;
        return std::vector<P>({x0, x1});
    }

    /* see http://mathworld.wolfram.com/CubicEquation.html */
    template <typename P>
    inline std::vector<P> SolveCubic(P a, P b, P c, P d)
    {
        P inv_a, b_a, b_a2, c_a, d_a;
        P Q, R, Q3, D, b_a_3;
        P AD, BD;

        P x0, x1, x2;

        if (a == 0)
        {
            /* solve second order system */
            if (b == 0)
            {
                /* solve first order system */
                if (c == 0)
                    return std::vector<P>();

                x0 = -d / c;
                return std::vector<P>({x0});
            }

            x2 = 0;
            return SolveQuadratic<P>(b, c, d);
        }

        /* calculate the normalized form x^3 + a2 * x^2 + a1 * x + a0 = 0 */
        inv_a = 1.0 / a;
        b_a = inv_a * b;
        b_a2 = b_a * b_a;
        c_a = inv_a * c;
        d_a = inv_a * d;

        /* solve the cubic equation */
        Q = (3 * c_a - b_a2) / 9;
        R = (9 * b_a * c_a - 27 * d_a - 2 * b_a * b_a2) / 54;
        Q3 = Q * Q * Q;
        D = Q3 + R * R;
        b_a_3 = (1.0 / 3.0) * b_a;

        if (Q == 0)
        {
            if (R == 0)
            {
                x0 = x1 = x2 = -b_a_3;
                return std::vector<P>({x0, x1, x2});
            }
            else
            {
                x0 = pow(2 * R, 1 / 3.0) - b_a_3;
                return std::vector<P>({x0});
            }
        }

        if (D <= 0)
        {
            /* three real roots */
            P theta = acos(R / sqrt(-Q3));
#if 0
            P cs_theta3 = cos(theta / 3.0);
            P cs_theta3_2pi3 = cos((theta + 2 * M_PI) / 3.0);
            P cs_theta3_4pi3 = cos((theta + 4 * M_PI) / 3.0);
#else
	    // employ the equation cos(a + b) = cos(a)cos(b) - sin(a)sin(b) to use a single trig call (i.e., sincos)
            P sn_theta3, cs_theta3;
            sincos(theta / 3.0, &sn_theta3, &cs_theta3);
	    // if sincos() is not available, uncomment the following line:
	    // sn_theta3 = sin(theta / 3.0); cs_theta3 = cos(theta / 3.0);
	    // cos(2*M_PI/3)=cos(4*M_PI/3)=-0.5;  sin(2*M_PI/3)=-sin(4*M_PI/3)=0.866025403784438708
            P cs_theta3_2pi3 = -cs_theta3*0.5 - sn_theta3*0.866025403784438708;
            P cs_theta3_4pi3 = -cs_theta3*0.5 + sn_theta3*0.866025403784438708;
#endif

            P sqrt_Q = sqrt(-Q);
            x0 = 2 * sqrt_Q * cs_theta3 - b_a_3;
            x1 = 2 * sqrt_Q * cs_theta3_2pi3 - b_a_3;
            x2 = 2 * sqrt_Q * cs_theta3_4pi3 - b_a_3;

            return std::vector<P>({x0, x1, x2});
        }

        /* D > 0, only one real root */
        AD = pow(fabs(R) + sqrt(D), 1.0 / 3.0) * (R > 0 ? 1 : (R < 0 ? -1 : 0));
        BD = (AD == 0) ? 0 : -Q / AD;

        /* calculate the sole real root */
        x0 = AD + BD - b_a_3;

        return std::vector<P>({x0});
    }

    /* see http://mathworld.wolfram.com/QuarticEquation.html */
    template <typename P>
    inline std::vector<P> SolveQuartic(P a, P b, P c, P d, P e)
    {
        P inv_a, b2, bc, b3, b_4;
        P r0;
        int nb_real_roots;
        P R2, R_2, R, inv_R;
        P D2, E2;

        P x0 = 0, x1 = 0, x2 = 0, x3 = 0;

        if (a == 0)
        {
            x3 = 0;
            return SolveCubic<P>(b, c, d, e);
        }

        /* normalize coefficients */
        inv_a = 1.0 / a;
        b *= inv_a;
        c *= inv_a;
        d *= inv_a;
        e *= inv_a;
        b2 = b * b;
        bc = b * c;
        b3 = b2 * b;

        /* solve resultant cubic */
        auto solution3 = SolveCubic<P>(1, -c, d * b - 4 * e, 4 * c * e - d * d - b2 * e);
        int n = (solution3.size() == 0) ? 0 : solution3.size();

        if (n == 0)
            return solution3;

        r0 = solution3[0];
        /* calculate R^2 */
        R2 = 0.25 * b2 - c + r0;
        if (R2 < 0)
            return std::vector<P>();

        R = sqrt(R2);
        inv_R = 1.0 / R;

        nb_real_roots = 0;

        /* calculate D^2 and E^2 */
        if (R < 10E-12)
        {
            P temp = r0 * r0 - 4 * e;
            if (temp < 0)
            {
                D2 = E2 = -1;
            }
            else
            {
                P sqrt_temp = sqrt(temp);
                D2 = 0.75 * b2 - 2 * c + 2 * sqrt_temp;
                E2 = D2 - 4 * sqrt_temp;
            }
        }
        else
        {
            P u = 0.75 * b2 - 2 * c - R2, v = 0.25 * inv_R * (4 * bc - 8 * d - b3);
            D2 = u + v;
            E2 = u - v;
        }

        b_4 = 0.25 * b;
        R_2 = 0.5 * R;
        if (D2 >= 0)
        {
            P D = sqrt(D2);
            P D_2 = 0.5 * D;
            nb_real_roots = 2;
            x0 = R_2 + D_2 - b_4;
            x1 = x0 - D;
        }

        /* calculate E^2 */
        if (E2 >= 0)
        {
            P E = sqrt(E2);
            P E_2 = 0.5 * E;
            if (nb_real_roots == 0)
            {
                x0 = -R_2 + E_2 - b_4;
                x1 = x0 - E;
                nb_real_roots = 2;
            }
            else
            {
                x2 = -R_2 + E_2 - b_4;
                x3 = x2 - E;
                nb_real_roots = 4;
            }
        }
        switch (nb_real_roots)
        {
        // case 0:
        //     return new Tuple<int, double[]>(0, null);
        //     break; // covered by the "default" case
        case 1:
            return std::vector<P>({x0});
            // break;
        case 2:
            return std::vector<P>({x0, x1});
            // break;
        case 3:
            return std::vector<P>({x0, x1, x2});
            // break;
        case 4:
            return std::vector<P>({x0, x1, x2, x3});
            // break;
        default:
            return std::vector<P>(); // just to shut the compiler up....
                                     // break;
        }
    }

} // end namespace PolySolvers

#endif
