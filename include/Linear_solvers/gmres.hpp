#ifndef __GMRES_HPP_
#define __GMRES_HPP_

#include <cmath>
#include <functional>
#include <vector>

namespace Kadath {
namespace linear_solvers {

/// Result of a GMRES solve.
struct GmresResult {
    bool converged =
        false;  ///< ||b - A x|| <= tolerance * ||b|| reached within max_iterations.
    int iterations = 0;  ///< Krylov iterations (operator applications).
    double relative_residual =
        0.;  ///< ||b - A x|| / ||b|| estimated by the Arnoldi recursion.
};

/// Unrestarted, right-preconditioned GMRES for A x = b (Saad, Iterative Methods for Sparse Linear Systems,
/// 2nd ed., Alg. 9.5) with modified Gram-Schmidt and Givens rotations.
/// @param apply_a  y = A v
/// @param apply_m  z = M^{-1} v (the preconditioner); the solution is x = M^{-1} (V y)
/// @param b right-hand side; @param x output solution (initial guess 0)
/// @param tolerance relative residual target; @param max_iterations Krylov dimension.
/// Every vector has the same length; the caller makes the operators consistent across MPI ranks.
inline GmresResult gmres(
    std::function<void(std::vector<double> const&, std::vector<double>&)> const&
        apply_a,
    std::function<void(std::vector<double> const&, std::vector<double>&)> const&
        apply_m,
    std::vector<double> const& b,
    std::vector<double>& x,
    double tolerance,
    int max_iterations) {
    std::size_t const n = b.size();
    auto dot = [n](std::vector<double> const& u, std::vector<double> const& v) {
        double s = 0.;
        for (std::size_t i = 0; i < n; ++i)
            s += u[i] * v[i];
        return s;
    };

    GmresResult result;
    x.assign(n, 0.);
    double const beta = std::sqrt(dot(b, b));
    if (beta == 0.) {
        result.converged = true;
        return result;
    }

    int const m = max_iterations;
    std::vector<std::vector<double>> v(m + 1, std::vector<double>(n, 0.));
    std::vector<std::vector<double>> h(m + 1, std::vector<double>(m, 0.));
    std::vector<double> cs(m, 0.), sn(m, 0.), g(m + 1, 0.);
    for (std::size_t i = 0; i < n; ++i)
        v[0][i] = b[i] / beta;
    g[0] = beta;

    std::vector<double> z(n), w(n);
    int k = 0;
    for (; k < m; ++k) {
        apply_m(v[k], z);
        apply_a(z, w);
        for (int j = 0; j <= k; ++j) {
            h[j][k] = dot(w, v[j]);
            for (std::size_t i = 0; i < n; ++i)
                w[i] -= h[j][k] * v[j][i];
        }
        double const next_norm =
            std::sqrt(dot(w, w));  // before the rotation zeroes h[k + 1][k]
        h[k + 1][k] = next_norm;
        if (next_norm > 0.)
            for (std::size_t i = 0; i < n; ++i)
                v[k + 1][i] = w[i] / next_norm;

        for (int j = 0; j < k;
             ++j) {  // apply the previous rotations to the new column
            double const t = cs[j] * h[j][k] + sn[j] * h[j + 1][k];
            h[j + 1][k] = -sn[j] * h[j][k] + cs[j] * h[j + 1][k];
            h[j][k] = t;
        }
        double const r = std::hypot(h[k][k], h[k + 1][k]);
        cs[k] = h[k][k] / r;
        sn[k] = h[k + 1][k] / r;
        h[k][k] = r;
        h[k + 1][k] = 0.;
        g[k + 1] = -sn[k] * g[k];
        g[k] = cs[k] * g[k];

        result.relative_residual = std::fabs(g[k + 1]) / beta;
        if (result.relative_residual <= tolerance ||
            next_norm == 0.) {  // converged or lucky breakdown
            ++k;
            result.converged = result.relative_residual <= tolerance;
            break;
        }
    }
    if (k == m)
        result.converged = result.relative_residual <= tolerance;
    result.iterations = k;

    std::vector<double> y(k, 0.);  // back substitution H y = g
    for (int i = k - 1; i >= 0; --i) {
        double s = g[i];
        for (int j = i + 1; j < k; ++j)
            s -= h[i][j] * y[j];
        y[i] = s / h[i][i];
    }
    std::vector<double> u(n, 0.);
    for (int j = 0; j < k; ++j)
        for (std::size_t i = 0; i < n; ++i)
            u[i] += y[j] * v[j][i];
    apply_m(u, x);
    return result;
}

}  // namespace linear_solvers
}  // namespace Kadath

#endif
