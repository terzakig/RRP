/**
 *
 * Mini algebra lib for fixed-size matrices
 *
 *
 * George Terzakis, June, 2024
 *
 **/

#ifndef MINIALG_HPP_
#define MINIALG_HPP_

#include <array>
#include <assert.h>
#include <cstdarg>
#include <iostream>
#include <initializer_list>
#include <math.h>

namespace minialg
{

    //@brief Minialg matrix
    template <typename P, int rows, int cols>
    class Matrix
    {
    public:
        static constexpr int N = rows * cols;

        inline static Matrix<P, rows, cols> eye()
        {
            Matrix<P, rows, cols> m;
            const size_t min_dim = std::min(rows, cols);
            for (size_t i = 0; i < min_dim; i++)
            {
                m(i, i) = 1;
            }
            return m;
        }

        inline static Matrix<P, rows, cols> zeros()
        {
            Matrix<P, rows, cols> m;
            return m;
        }

        inline Matrix() : mat_{0}
        {
        }

        inline Matrix(const std::array<P, N> &a) : mat_(a)
        {
        }

        inline P *DataPtr()
        {
            return &mat_[0];
        }

        inline Matrix<P, cols, rows> t()
        {
            Matrix<P, cols, rows> T;
            for (size_t r = 0; r < rows; r++)
            {
                for (size_t c = 0; c < cols; c++)
                {
                    T(c, r) = mat_[r * cols + c];
                }
            }
            return T;
        }

        inline P &operator()(const size_t &r, const size_t &c)
        {
            return *(&mat_[0] + r * cols + c);
        }

        inline const P &operator()(const size_t &r, const size_t &c) const
        {
            return *(&mat_[0] + r * cols + c);
        }

        inline P &operator[](const size_t &i)
        {
            return *(&mat_[0] + i);
        }

        inline const P &operator[](const size_t &i) const
        {
            return *(&mat_[0] + i);
        }

        //@brief Get column as vector (rows x 1)
        inline Matrix<P, rows, 1> GetColumn(const size_t &c) const
        {
            Matrix<P, rows, 1> col;
            for (size_t r = 0; r < rows; r++)
            {
                col[r] = mat_[r * cols + c];
            }
            return col;
        }

        //@brief Get row as vector (cols x 1)
        inline Matrix<P, cols, 1> GetRow(const size_t &r) const
        {
            Matrix<P, cols, 1> row;
            for (size_t c = 0; c < cols; c++)
            {
                row[c] = mat_[r * cols + c];
            }
            return row;
        }

        inline Matrix<P, rows, cols> operator-()
        {
            Matrix<P, rows, cols> N;
            for (size_t r = 0; r < rows; r++)
            {
                for (size_t c = 0; c < cols; c++)
                {
                    N(r, c) = -mat_[r * cols + c];
                }
            }
            return *this;
        }

        inline Matrix<P, rows, cols> &operator=(const Matrix<P, rows, cols> &a)
        {
            for (size_t i = 0; i < N; i++)
            {
                mat_[i] = a.mat_[i];
            }
            return *this;
        }

        inline Matrix<P, rows, cols> &operator+=(const Matrix<P, rows, cols> &a)
        {
            for (size_t i = 0; i < N; i++)
            {
                mat_[i] += a.mat_[i];
            }
            return *this;
        }

        inline Matrix<P, rows, cols> &operator-=(const Matrix<P, rows, cols> &a)
        {
            for (size_t i = 0; i < N; i++)
            {
                mat_[i] -= a.mat_[i];
            }
            return *this;
        }

    protected:
        std::array<P, N> mat_;
    };

    template <typename P, int rows, int cols>
    std::ostream &operator<<(std::ostream &os, const Matrix<P, rows, cols> &m)
    {
        os << "[";
        for (int r = 0; r < rows; r++)
        {
            for (int c = 0; c < cols; c++)
            {
                os << m(r, c);
                if (c < cols - 1)
                {
                    os << ", ";
                }
                else if (r < rows - 1)
                {
                    os << ";\n";
                }
            }
        }
        os << "]";
        return os;
    }

    template <typename P, int rows, int cols>
    Matrix<P, rows, cols> operator+(const Matrix<P, rows, cols> &A, const Matrix<P, rows, cols> &B)
    {
        Matrix<P, rows, cols> C;
        for (size_t r = 0; r < rows; r++)
        {
            for (size_t c = 0; c < cols; c++)
            {
                C(r, c) = A(r, c) + B(r, c);
            }
        }
        return C;
    }

    template <typename P, int rows, int cols>
    Matrix<P, rows, cols> operator-(const Matrix<P, rows, cols> &A, const Matrix<P, rows, cols> &B)
    {
        Matrix<P, rows, cols> C;
        for (size_t r = 0; r < rows; r++)
        {
            for (size_t c = 0; c < cols; c++)
            {
                C(r, c) = A(r, c) - B(r, c);
            }
        }
        return C;
    }

    template <typename P, int rows, int cols>
    Matrix<P, rows, cols> operator*(const double &a, const Matrix<P, rows, cols> &B)
    {
        Matrix<P, rows, cols> C;
        for (size_t r = 0; r < rows; r++)
        {
            for (size_t c = 0; c < cols; c++)
            {
                C(r, c) = a * B(r, c);
            }
        }
        return C;
    }

    template <typename P, int rows, int cols>
    Matrix<P, rows, cols> operator*(const Matrix<P, rows, cols> &A, const double &b)
    {
        Matrix<P, rows, cols> C;
        for (size_t r = 0; r < rows; r++)
        {
            for (size_t c = 0; c < cols; c++)
            {
                C(r, c) = b * A(r, c);
            }
        }
        return C;
    }

    template <typename P, int rowsA, int dim, int colsB>
    Matrix<P, rowsA, colsB> operator*(const Matrix<P, rowsA, dim> &A, const Matrix<P, dim, colsB> &B)
    {
        Matrix<P, rowsA, colsB> C;
        for (size_t r = 0; r < rowsA; r++)
        {
            for (size_t c = 0; c < colsB; c++)
            {
                P sum = 0;
                for (size_t k = 0; k < dim; k++)
                {
                    sum += A(r, k) * B(k, c);
                }
                C(r, c) = sum;
            }
        }
        return C;
    }

    //! Invert a 2x2 matrix
    template <typename P>
    bool Invert2x2(const Matrix<P, 2, 2> &A, Matrix<P, 2, 2> &Ainv)
    {
        const P detA = A(0, 0) * A(1, 1) - A(1, 0) * A(0, 1);
        if (abs(detA) < 1e-5)
        {
            Ainv = Matrix<P, 2, 2>::zeros();
            return false;
        }
        const P inv_det = static_cast<P>(1.0 / detA);

        // Invert elements and swap diagonal
        Ainv(0, 0) = inv_det * A(1, 1);
        Ainv(0, 1) = inv_det * -A(0, 1);
        Ainv(1, 0) = inv_det * -A(1, 0);
        Ainv(1, 1) = inv_det * A(0, 0);
        return true;
    }

    template <typename P, int rows>
    using Vector = Matrix<P, rows, 1>;

    template <typename P>
    Matrix<P, 3, 3> CrossProductMatrix(const Vector<P, 3> &u)
    {
        Matrix<P, 3, 3> Ux;
        Ux(0, 0) = Ux(1, 1) = Ux(2, 2) = 0;
        Ux(0, 1) = -u[2];
        Ux(0, 2) = u[1];
        Ux(1, 0) = u[2];
        Ux(1, 2) = -u[0];
        Ux(2, 0) = -u[1];
        Ux(2, 1) = u[0];

        return Ux;
    }

    template <typename P>
    Vector<P, 3> Cross(const Vector<P, 3> &a, const Vector<P, 3> &b)
    {
        return Vector<P, 3>( //
            {
                -a[2] * b[1] + a[1] * b[2], //
                a[2] * b[0] - a[0] * b[2],  //
                -a[1] * b[0] + a[0] * b[1]  //
            });
    }

    template <typename P, int rows>
    double Dotp(const Vector<P, rows> &u, const Vector<P, rows> &v)
    {
        double sum = 0;
        for (size_t r = 0; r < rows; r++)
        {
            sum += u[r] * v[r];
        }
        return sum;
    }

    template <typename P, int rows>
    P Norm2(const Vector<P, rows> &u)
    {
        return sqrt(Dotp(u, u));
    }

    template <typename P, int rows>
    Vector<P, rows> Normalize(const Vector<P, rows> &v)
    {
        const P norm_v = Norm2(v);

        // assert(norm_v > 1e-7); // Not too s

        Vector<P, rows> ret = v;
        for (size_t i = 0; i < rows; i++)
        {
            ret[i] /= norm_v;
        }
        return ret;
    }

} // namespace minialg

#endif
