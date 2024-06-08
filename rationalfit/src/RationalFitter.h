/**
 *
 * Rational function fitter:
 *
 * f(t) = (a0 + a1*t + a2*t^2) / (b0 + b1*t + b3*t^2) s.t. b1^2-4*b1*b2 < 0
 *
 * George Terzakis, April 2023
 *
 **/

#ifndef RATIONAL_FITTER_H__
#define RATIONAL_FITTER_H__

#include <array>
#include <vector>
#include <unordered_set>

#include "eigen3x4/EigenDecompose.h"
#include "minialg.h"

namespace mini = minialg;

namespace ratfit
{

    template <int data_dim>
    class RationalFitter
    {

    public:
        static constexpr double LARGE_WEIGHT = 1000.0;

        template <typename Pt, typename Pd, typename Pw = double>
        RationalFitter(                                          //
            const std::vector<Pt> &time_instances,               //
            const std::vector<mini::Vector<Pd, data_dim>> &data, //
            const std::vector<size_t> &fixed_point_indexes,      //
            const std::vector<Pw> &weights = std::vector<Pw>())
        {
            assert(                                                                         //
                data.size() > 4 &&                                                          //
                time_instances.size() == data.size() &&                                     //
                ((!weights.empty() && weights.size() == data.size()) || weights.empty()) && //
                fixed_point_indexes.size() < 4);

            // Check what optimization to run (with/without fixed points)
            if (fixed_point_indexes.empty())
            {

                InitSimpleRegression(time_instances, data, weights);
            }
            else
            {
                InitRegressionWithFixedPoints(time_instances, data, fixed_point_indexes, weights);
            }
        }

        template <typename Pt, typename Pd, typename Pw = double>
        void InitRegressionWithFixedPoints(                      //
            const std::vector<Pt> &time_instances,               //
            const std::vector<mini::Vector<Pd, data_dim>> &data, //
            const std::vector<size_t> &fixed_point_indexes,      //
            const std::vector<Pw> &weights = std::vector<Pw>())
        {
            // Sum(y^2*(c*c'))
            std::array<mini::Matrix<double, 3, 3>, data_dim> As;
            // Sum(c*c')
            std::array<mini::Matrix<double, 3, 3>, data_dim> Bs;
            // Sum(y*c*c')
            std::array<mini::Matrix<double, 3, 3>, data_dim> Cs;

            // Time base
            time_base_ = time_instances[0];
            // Scale
            scale_ = 1.0 / (time_instances[time_instances.size() - 1] - time_base_);

            for (size_t d = 0; d < data_dim; d++)
            {
                // No need to Zero the matrices As[d], Bs[d], Cs[d] (default constructor zeros the matrices)
                for (size_t j = 0; j < data.size(); j++)
                {
                    const double dt = (time_instances[j] - time_base_) * scale_;
                    const double dt2 = dt * dt;
                    const double dt3 = dt2 * dt;
                    const double dt4 = dt3 * dt;
                    const double y = data[j][d];

                    const mini::Matrix<double, 3, 3> G({
                        1, dt, dt2,   //
                        dt, dt2, dt3, //
                        dt2, dt3, dt4 //
                    });
                    const double w = weights.empty() ? 1.0 : weights[j];

                    As[d] += (w * y * y) * G;
                    Bs[d] += w * G;
                    Cs[d] += (w * y) * G;
                }

                // Use eqs. (14), (15), (16) in paper.
                // A_hat = N'*A*N
                // B_hat = N'*(B - A*H*F)
                // C_hat = F*(H'*A*H*F + C-2*H'*B).

                // Take cases on the number of fixed points
                if (fixed_point_indexes.size() == 1)
                {
                    std::cout << "points fixed 1!!!\n";
                    // Fill constraint matrices E and D
                    mini::Matrix<double, 1, 3> D;
                    mini::Matrix<double, 1, 3> E;
                    for (size_t r = 0; r < fixed_point_indexes.size(); r++)
                    {
                        const size_t index = fixed_point_indexes[r];
                        const double dt = (time_instances[index] - time_base_) * scale_;
                        const double y = data[index][d];
                        E(r, 0) = y * (D(r, 0) = 1);
                        E(r, 1) = y * (D(r, 1) = dt);
                        E(r, 2) = y * (D(r, 2) = dt * dt);
                    }
                    // A. Null space is 2D and row space is 1D
                    mini::Vector<double, 3> u1 = D.GetRow(0);
                    u1 = mini::Normalize(u1);
                    mini::Matrix<double, 3, 1> H = u1;

                    // B. Row space is a 2D basis of the orthogonal space of u1
                    if (u1[2] < 0)
                    {
                        u1 = (-1) * u1;
                    }
                    // Obtain a basis of the orth. space of u1 as per. https://link.springer.com/article/10.1007/s10851-017-0765-x
                    mini::Matrix<double, 3, 2> N;
                    const double one_plus_rho = 1 + u1[2];
                    N(0, 0) = one_plus_rho - u1[0] * u1[0];
                    N(0, 1) = -u1[0] * u1[1];
                    N(1, 0) = -u1[1] * u1[0];
                    N(1, 1) = one_plus_rho - u1[1] * u1[1];
                    N(2, 0) = -one_plus_rho * u1[0];
                    N(2, 1) = -one_plus_rho * u1[1];

                    // Compute F
                    const double HtDtDH = mini::Dotp(H, D.t() * D * H);
                    mini::Matrix<double, 1, 3> F = (1.0 / HtDtDH) * mini::Dotp(D.GetRow(0), H) * E;

                    // Compute A, B, C hats
                    mini::Matrix<double, 2, 2> A_hat = N.t() * As[d] * N;
                    mini::Matrix<double, 2, 3> B_hat = N.t() * (Bs[d] - As[d] * H * F);
                    mini::Matrix<double, 3, 3> C_hat = (H * F).t() * As[d] * H * F + Cs[d] - 2 * (H * F).t() * Bs[d];

                    // Get the suymmetric part of C_hat to be n the safe side
                    C_hat = 0.5 * (C_hat + C_hat.t());

                    // Finally, the eigenvector matrix, Q
                    mini::Matrix<double, 2, 2> A_hatinv;
                    mini::Invert2x2(A_hat, A_hatinv);
                    mini::Matrix<double, 3, 3> Q = C_hat - B_hat.t() * A_hatinv * B_hat;

                    // The Qp = inv(K)*Q
                    // Qp = inv(K)*(symmetric(C_hat) - B_hat*
                    mini::Matrix<double, 3, 3> Qp({
                        //
                        -0.5 * Q(2, 0), -0.5 * Q(2, 1), -0.5 * Q(2, 2), //
                        Q(1, 0), Q(1, 1), Q(1, 2),                      //
                        -0.5 * Q(0, 0), -0.5 * Q(0, 1), -0.5 * Q(0, 2)  //
                    });

                    auto eigen_decomposition = Eigen34::EigenDecompose3x3(Qp.DataPtr());
                    std::vector<double> eigenvalues = eigen_decomposition.first;
                    std::vector<std::vector<double>> eigenvectors = eigen_decomposition.second;
                    double min_error = std::numeric_limits<double>::max(); // probably not necessary
                    if (eigenvalues.empty())
                    {
                        std::cerr << " NO EIGENVALUES!!!!!!!  NO EIGENVALUES!!!!!!! NO EIGENVALUES!!!!!!! \n";
                        exit(-1);
                    }
                    // Find the suitable eigenvalue-eigenvector
                    min_error = std::numeric_limits<double>::max(); // probably not necessary
                    for (size_t i = 0; i < eigenvalues.size(); i++)
                    {
                        mini::Vector<double, 3> b({eigenvectors[i][0], eigenvectors[i][1], eigenvectors[i][2]});
                        // std::cout << " b for d = " << d << std::endl
                        //           << ": " << b << std::endl;
                        b = (1 / mini::Norm2(b)) * b;
                        const double Delta = b[1] * b[1] - 4 * b[2] * b[0];
                        // std::cout << " Delta " << Delta << std::endl;

                        if (Delta < 0 || eigenvalues.size() == 1)
                        {
                            const double error = Dotp(b, C_hat * b);
                            if (min_error > error)
                            {
                                min_error = error;
                                denominators_[d] = b;
                            }
                        }
                    }

                    // Store solutions
                    numerators_[d] = N * (A_hatinv * B_hat * denominators_[d]) + H * (F * denominators_[d]);
                    if (eigenvalues.size() == 1)
                    {
                        std::cout << " numerator = " << F * denominators_[d] << std::endl;
                        std::cout << " denominaor = " << F * denominators_[d] << std::endl;
                    }
                }
                else if (fixed_point_indexes.size() == 2)
                {
                    // Fill constraint matrices E and D
                    mini::Matrix<double, 2, 3> D;
                    mini::Matrix<double, 2, 3> E;
                    for (size_t r = 0; r < fixed_point_indexes.size(); r++)
                    {
                        const size_t index = fixed_point_indexes[r];
                        const double dt = (time_instances[index] - time_base_) * scale_;
                        const double y = data[index][d];
                        E(r, 0) = y * (D(r, 0) = 1);
                        E(r, 1) = y * (D(r, 1) = dt);
                        E(r, 2) = y * (D(r, 2) = dt * dt);
                    }

                    // A. The row space is no simply the two rows of D
                    mini::Vector<double, 3> u1 = D.GetRow(0);
                    mini::Vector<double, 3> u2 = D.GetRow(1);
                    // Orthogonalize for numerical accuracy
                    u1 = mini::Normalize(u1);
                    u2 = u2 - mini::Dotp(u1, u2) * u1;
                    u2 = mini::Normalize(u2);

                    mini::Matrix<double, 3, 2> H;
                    H(0, 0) = u1[0];
                    H(1, 0) = u1[1];
                    H(2, 0) = u1[2];

                    H(0, 1) = u2[0];
                    H(1, 1) = u2[1];
                    H(2, 1) = u2[2];

                    // Obtain the null vector as the cross product of the row space vectors
                    mini::Vector<double, 3> N = mini::Cross(u1, u2);

                    // Compute F
                    mini::Matrix<double, 2, 2> HtDtDH = H.t() * D.t() * D * H;
                    mini::Matrix<double, 2, 2> HtDtDHinv;
                    mini::Invert2x2(HtDtDH, HtDtDHinv);
                    mini::Matrix<double, 2, 3> F = HtDtDHinv * (D * H).t() * E;

                    // Compute A, B, C hats
                    const double A_hat = mini::Dotp(N, As[d] * N);
                    mini::Matrix<double, 1, 3> B_hat = N.t() * (Bs[d] - As[d] * H * F);
                    mini::Matrix<double, 3, 3> C_hat = (H * F).t() * As[d] * H * F + Cs[d] - 2 * (H * F).t() * Bs[d];

                    // Get the suymmetric part of C_hat to be n the safe side
                    C_hat = 0.5 * (C_hat + C_hat.t());

                    // Finally, the eigenvector matrix, Q
                    mini::Matrix<double, 3, 3> Q = C_hat - (1.0 / A_hat) * B_hat.t() * B_hat;

                    // The Qp = inv(K)*Q
                    // Qp = inv(K)*(symmetric(C_hat) - B_hat*
                    mini::Matrix<double, 3, 3> Qp({
                        //
                        -0.5 * Q(2, 0), -0.5 * Q(2, 1), -0.5 * Q(2, 2), //
                        Q(1, 0), Q(1, 1), Q(1, 2),                      //
                        -0.5 * Q(0, 0), -0.5 * Q(0, 1), -0.5 * Q(0, 2)  //
                    });

                    auto eigen_decomposition = Eigen34::EigenDecompose3x3(Qp.DataPtr());
                    std::vector<double> eigenvalues = eigen_decomposition.first;
                    std::vector<std::vector<double>> eigenvectors = eigen_decomposition.second;
                    double min_error = std::numeric_limits<double>::max(); // probably not necessary
                    if (eigenvalues.empty())
                    {
                        std::cerr << " NO EIGENVALUES!!!!!!!  NO EIGENVALUES!!!!!!! NO EIGENVALUES!!!!!!! \n";
                        exit(-1);
                    }
                    // Find the suitable eigenvalue-eigenvector
                    min_error = std::numeric_limits<double>::max(); // probably not necessary
                    for (size_t i = 0; i < eigenvalues.size(); i++)
                    {
                        mini::Vector<double, 3> b({eigenvectors[i][0], eigenvectors[i][1], eigenvectors[i][2]});
                        // std::cout << " b for d = " << d << std::endl
                        //           << ": " << b << std::endl;
                        b = (1 / mini::Norm2(b)) * b;
                        const double Delta = b[1] * b[1] - 4 * b[2] * b[0];
                        // std::cout << " Delta " << Delta << std::endl;

                        if (Delta < 0 || eigenvalues.size() == 1)
                        {
                            const double error = Dotp(b, C_hat * b);
                            if (min_error > error)
                            {
                                min_error = error;
                                denominators_[d] = b;
                            }
                        }
                    }

                    // Store solutions
                    numerators_[d] = N * ((1.0 / A_hat) * B_hat * denominators_[d]) + (H * F * denominators_[d]);
                    if (eigenvalues.size() == 1)
                    {
                        std::cout << " numerator = " << F * denominators_[d] << std::endl;
                        std::cout << " denominaor = " << F * denominators_[d] << std::endl;
                    }
                }
                else if (fixed_point_indexes.size() == 3)
                {
                    // Fill constraint matrices E and D
                    mini::Matrix<double, 3, 3> D;
                    mini::Matrix<double, 3, 3> E;
                    for (size_t r = 0; r < fixed_point_indexes.size(); r++)
                    {
                        const size_t index = fixed_point_indexes[r];
                        const double dt = (time_instances[index] - time_base_) * scale_;
                        const double y = data[index][d];
                        E(r, 0) = y * (D(r, 0) = 1);
                        E(r, 1) = y * (D(r, 1) = dt);
                        E(r, 2) = y * (D(r, 2) = dt * dt);
                    }

                    mini::Matrix<double, 3, 3> H = mini::Matrix<double, 3, 3>::eye();

                    // Compute F
                    mini::Matrix<double, 3, 3> HtDtDH = H.t() * D.t() * D * H;
                    mini::Matrix<double, 3, 3> HtDtDHinv;
                    InvertSPD3x3(HtDtDH, HtDtDHinv);
                    mini::Matrix<double, 3, 3> F = HtDtDHinv * (D * H).t() * E;

                    // No A, B hats this time...
                    mini::Matrix<double, 3, 3> C_hat = (H * F).t() * As[d] * H * F + Cs[d] - 2 * (H * F).t() * Bs[d];

                    // Get the symmetric part of C_hat to be n the safe side
                    C_hat = 0.5 * (C_hat + C_hat.t());

                    // Finally, the eigenvector matrix, Q
                    mini::Matrix<double, 3, 3> Q = C_hat;

                    // The Qp = inv(K)*Q
                    // Qp = inv(K)*(symmetric(C_hat) - B_hat*
                    mini::Matrix<double, 3, 3> Qp({
                        //
                        -0.5 * Q(2, 0), -0.5 * Q(2, 1), -0.5 * Q(2, 2), //
                        Q(1, 0), Q(1, 1), Q(1, 2),                      //
                        -0.5 * Q(0, 0), -0.5 * Q(0, 1), -0.5 * Q(0, 2)  //
                    });

                    auto eigen_decomposition = Eigen34::EigenDecompose3x3(Qp.DataPtr());
                    std::vector<double> eigenvalues = eigen_decomposition.first;
                    std::vector<std::vector<double>> eigenvectors = eigen_decomposition.second;
                    double min_error = std::numeric_limits<double>::max(); // probably not necessary
                    if (eigenvalues.empty())
                    {
                        std::cerr << " NO EIGENVALUES!!!!!!!  NO EIGENVALUES!!!!!!! NO EIGENVALUES!!!!!!! \n";
                        exit(-1);
                    }
                    // Find the suitable eigenvalue-eigenvector
                    min_error = std::numeric_limits<double>::max(); // probably not necessary
                    for (size_t i = 0; i < eigenvalues.size(); i++)
                    {
                        mini::Vector<double, 3> b({eigenvectors[i][0], eigenvectors[i][1], eigenvectors[i][2]});
                        // std::cout << " b for d = " << d << std::endl
                        //           << ": " << b << std::endl;
                        b = (1 / mini::Norm2(b)) * b;
                        const double Delta = b[1] * b[1] - 4 * b[2] * b[0];
                        // std::cout << " Delta " << Delta << std::endl;

                        if (Delta < 0 || eigenvalues.size() == 1)
                        {
                            const double error = Dotp(b, C_hat * b);
                            if (min_error > error)
                            {
                                min_error = error;
                                denominators_[d] = b;
                            }
                        }
                    }

                    // Store solutions
                    numerators_[d] = F * denominators_[d];
                    if (eigenvalues.size() == 1)
                    {
                        std::cout << " numerator = " << F * denominators_[d] << std::endl;
                        std::cout << " denominaor = " << F * denominators_[d] << std::endl;
                    }
                }
                else if (fixed_point_indexes.size() == 4)
                {
                    // Fill constraint matrices E and D
                    mini::Matrix<double, 4, 3> D;
                    mini::Matrix<double, 4, 3> E;
                    for (size_t r = 0; r < fixed_point_indexes.size(); r++)
                    {
                        const size_t index = fixed_point_indexes[r];
                        const double dt = (time_instances[index] - time_base_) * scale_;
                        const double y = data[index][d];
                        E(r, 0) = y * (D(r, 0) = 1);
                        E(r, 1) = y * (D(r, 1) = dt);
                        E(r, 2) = y * (D(r, 2) = dt * dt);
                    }

                    mini::Matrix<double, 3, 3> H = mini::Matrix<double, 3, 3>::eye();

                    // Compute F
                    mini::Matrix<double, 3, 3> HtDtDH = H.t() * D.t() * D * H;
                    mini::Matrix<double, 3, 3> HtDtDHinv;
                    InvertSPD3x3(HtDtDH, HtDtDHinv);
                    mini::Matrix<double, 3, 3> F = HtDtDHinv * (D * H).t() * E;

                    // No A, B hats this time...
                    mini::Matrix<double, 3, 3> C_hat = (H * F).t() * As[d] * H * F + Cs[d] - 2 * (H * F).t() * Bs[d];

                    // Get the symmetric part of C_hat to be n the safe side
                    C_hat = 0.5 * (C_hat + C_hat.t());

                    // Finally, the eigenvector matrix, Q
                    mini::Matrix<double, 3, 3> Q = C_hat;

                    // The Qp = inv(K)*Q
                    // Qp = inv(K)*(symmetric(C_hat) - B_hat*
                    mini::Matrix<double, 3, 3> Qp({
                        //
                        -0.5 * Q(2, 0), -0.5 * Q(2, 1), -0.5 * Q(2, 2), //
                        Q(1, 0), Q(1, 1), Q(1, 2),                      //
                        -0.5 * Q(0, 0), -0.5 * Q(0, 1), -0.5 * Q(0, 2)  //
                    });

                    auto eigen_decomposition = Eigen34::EigenDecompose3x3(Qp.DataPtr());
                    std::vector<double> eigenvalues = eigen_decomposition.first;
                    std::vector<std::vector<double>> eigenvectors = eigen_decomposition.second;
                    double min_error = std::numeric_limits<double>::max(); // probably not necessary
                    if (eigenvalues.empty())
                    {
                        std::cerr << " NO EIGENVALUES!!!!!!!  NO EIGENVALUES!!!!!!! NO EIGENVALUES!!!!!!! \n";
                        exit(-1);
                    }
                    // Find the suitable eigenvalue-eigenvector
                    min_error = std::numeric_limits<double>::max(); // probably not necessary
                    for (size_t i = 0; i < eigenvalues.size(); i++)
                    {
                        mini::Vector<double, 3> b({eigenvectors[i][0], eigenvectors[i][1], eigenvectors[i][2]});
                        // std::cout << " b for d = " << d << std::endl
                        //           << ": " << b << std::endl;
                        b = (1 / mini::Norm2(b)) * b;
                        const double Delta = b[1] * b[1] - 4 * b[2] * b[0];
                        // std::cout << " Delta " << Delta << std::endl;

                        if (Delta < 0 || eigenvalues.size() == 1)
                        {
                            const double error = Dotp(b, C_hat * b);
                            if (min_error > error)
                            {
                                min_error = error;
                                denominators_[d] = b;
                            }
                        }
                    }

                    // Store solutions
                    numerators_[d] = F * denominators_[d];
                    if (eigenvalues.size() == 1)
                    {
                        std::cout << " numerator = " << F * denominators_[d] << std::endl;
                        std::cout << " denominaor = " << F * denominators_[d] << std::endl;
                    }
                }
            }
        }

        // @brief Initialize polynomials for simple regression
        template <typename Pt, typename Pd, typename Pw>
        inline void InitSimpleRegression(                        //
            const std::vector<Pt> &time_instances,               //
            const std::vector<mini::Vector<Pd, data_dim>> &data, //
            const std::vector<Pw> &weights = std::vector<Pw>())
        {
            // Sum(y^2*(c*c'))
            std::array<mini::Matrix<double, 3, 3>, data_dim> As;
            // Sum(c*c')
            std::array<mini::Matrix<double, 3, 3>, data_dim> Bs;
            // Sum(y*c*c')
            std::array<mini::Matrix<double, 3, 3>, data_dim> Cs;

            // Time base
            time_base_ = time_instances[0];
            // Scale
            scale_ = 1.0 / (time_instances[time_instances.size() - 1] - time_base_);

            for (size_t d = 0; d < data_dim; d++)
            {
                // No need to Zero the matrices As[d], Bs[d], Cs[d] (default constructor zeros the matrices)
                for (size_t j = 0; j < data.size(); j++)
                {
                    const double dt = (time_instances[j] - time_base_) * scale_;
                    const double dt2 = dt * dt;
                    const double dt3 = dt2 * dt;
                    const double dt4 = dt3 * dt;
                    const double y = data[j][d];

                    const mini::Matrix<double, 3, 3> G({
                        1, dt, dt2,   //
                        dt, dt2, dt3, //
                        dt2, dt3, dt4 //
                    });
                    const double w = weights.empty() ? 1.0 : weights[j];
                    As[d] += (w * y * y) * G;
                    Bs[d] += w * G;
                    Cs[d] += (w * y) * G;
                }

                mini::Matrix<double, 3, 3> Binv;
                InvertSPD3x3(Bs[d], Binv);

                mini::Matrix<double, 3, 3> Q;
                Q = As[d] - Cs[d] * Binv * Cs[d];
                //  Qp = iK*Q;
                mini::Matrix<double, 3, 3> Qp({                                                //
                                               -0.5 * Q(2, 0), -0.5 * Q(2, 1), -0.5 * Q(2, 2), //
                                               Q(1, 0), Q(1, 1), Q(1, 2),                      //
                                               -0.5 * Q(0, 0), -0.5 * Q(0, 1), -0.5 * Q(0, 2)});
                // eigen-decompose Q
                auto eigen_decomposition = Eigen34::EigenDecompose3x3(Qp.DataPtr());
                const std::vector<double> eigenvalues = eigen_decomposition.first;
                const std::vector<std::vector<double>> eigenvectors = eigen_decomposition.second;
                double min_error = std::numeric_limits<double>::max(); // probably not necessary
                if (eigenvalues.empty())
                {
                    std::cerr << " NO EIGENVALUES!!!!!!!  NO EIGENVALUES!!!!!!! NO EIGENVALUES!!!!!!! \n";
                    exit(-1);
                }

                for (size_t i = 0; i < eigenvalues.size(); i++)
                {
                    mini::Vector<double, 3> b({eigenvectors[i][0], eigenvectors[i][1], eigenvectors[i][2]});
                    b = (1 / Norm2(b)) * b;
                    const double Delta = b[1] * b[1] - 4 * b[2] * b[0];
                    if (Delta < 0)
                    {
                        const double error = Dotp(b, Q * b);
                        if (min_error > error)
                        {
                            min_error = error;
                            denominators_[d] = b;
                        }
                    }
                }
                // Store solutions
                numerators_[d] = Binv * Cs[d] * denominators_[d];
            }
        }

        /**
         * @brief Get the value of the of the rational function at t
         */
        template <typename P>
        inline mini::Vector<P, data_dim> Value(const P &t)
        {
            const double dt = (t - time_base_) * scale_;
            mini::Vector<P, data_dim> val;
            const mini::Vector<double, 3> c({1, dt, dt * dt});
            for (size_t d = 0; d < data_dim; d++)
            {
                val[d] = static_cast<P>(Dotp(numerators_[d], c) / Dotp(denominators_[d], c));
            }
            return val;
        }

    private:
        //! Numerator coefficients
        std::array<mini::Vector<double, 3>, data_dim> numerators_;
        //! Denominator coefficients
        std::array<mini::Vector<double, 3>, data_dim> denominators_;
        //! The time base
        double time_base_;
        //! Scale of the range of data, i.e., 1 / (t_n - t_1)
        double scale_;

        /* Inverse of SPD 3x3 A.
         * Involves computing a lower triangular sqrt-free Cholesky factor A=L*D*L'
         * (L has ones on the diagonal, D is diagonal)
         * and using it to solve 3 systems with the canonical basis vectors as rhs.
         *
         * Only the lower triangular part of A is accessed.
         *
         * The function returns 0 if successful, non-zero otherwise
         *
         * see http://euler.nmt.edu/~brian/ldlt.html
         */
        inline static int InvertSPD3x3(const mini::Matrix<double, 3, 3> &A, mini::Matrix<double, 3, 3> &A1)
        {
            double L[3 * 3], D[3], v[2], x[3];

            v[0] = D[0] = A(0, 0);
            if (v[0] <= 1E-10)
                return 1;
            v[1] = 1.0 / v[0];
            L[3] = A(1, 0) * v[1];
            L[6] = A(2, 0) * v[1];
            // L[0]=1.0;
            // L[1]=L[2]=0.0;

            v[0] = L[3] * D[0];
            v[1] = D[1] = A(1, 1) - L[3] * v[0];
            if (v[1] <= 1E-10)
                return 2;
            L[7] = (A(2, 1) - L[6] * v[0]) / v[1];
            // L[4]=1.0;
            // L[5]=0.0;

            v[0] = L[6] * D[0];
            v[1] = L[7] * D[1];
            D[2] = A(2, 2) - L[6] * v[0] - L[7] * v[1];
            // L[8]=1.0;

            D[0] = 1.0 / D[0];
            D[1] = 1.0 / D[1];
            D[2] = 1.0 / D[2];

#if 0
    /* Forward solve Lx = b */
    x[0]= b[0];
    x[1]=(b[1]-L[3]*x[0]);
    x[2]=(b[2]-L[6]*x[0]-L[7]*x[1]);

    /* Backward solve D*L'x = y */
    x[2]=x[2]*D[2];
    x[1]=x[1]*D[1]-L[7]*x[2];
    x[0]=x[0]*D[0]-L[3]*x[1]-L[6]*x[2];
#endif

            /* use the factorization to solve for A*x_i = e_i */

            /* Forward solve Lx = e0 */
            // x[0]=1.0;
            x[1] = -L[3];
            x[2] = -L[6] + L[7] * L[3];

            /* Backward solve D*L'x = y */
            A1(0, 2) = x[2] = x[2] * D[2];
            A1(0, 1) = x[1] = x[1] * D[1] - L[7] * x[2];
            A1(0, 0) = D[0] - L[3] * x[1] - L[6] * x[2];

            /* Forward solve Lx = e1 */
            // x[0]=0.0;
            // x[1]=1.0;
            x[2] = -L[7];

            /* Backward solve D*L'x = y */
            A1(1, 2) = x[2] = x[2] * D[2];
            A1(1, 1) = x[1] = D[1] - L[7] * x[2];
            A1(1, 0) = -L[3] * x[1] - L[6] * x[2];

            /* Forward solve Lx = e2 */
            // x[0]=0.0;
            // x[1]=0.0;
            // x[2]=1.0;

            /* Backward solve D*L'x = y */
            A1(2, 2) = x[2] = D[2];
            A1(2, 1) = x[1] = -L[7] * x[2];
            A1(2, 0) = -L[3] * x[1] - L[6] * x[2];

            return 0;
        }
    };

} // namespace ORB_SLAM2
#endif
