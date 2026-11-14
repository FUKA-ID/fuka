/*
 * T6: right-preconditioned GMRES (include/Linear_solvers/gmres.hpp) on dense test systems.
 *  - no preconditioner: converges, and the true residual |b - A x| / |b| matches the requested tolerance
 *  - exact preconditioner M = A: one iteration
 *  - stale preconditioner M = A + 1e-3 E (a stand-in for an old Newton Jacobian): few iterations
 *  - iteration cap: reports non-convergence after exactly max_iterations
 *  - b = 0: x = 0 without iterations
 * Serial; returns 0 if all checks pass.
 */
#include <cmath>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "Linear_solvers/gmres.hpp"

using namespace Kadath::linear_solvers;
using Dense = std::vector<std::vector<double>>;
using Op =
    std::function<void(std::vector<double> const&, std::vector<double>&)>;

namespace {

int failures = 0;

std::string sci(double x) {
    std::ostringstream os;
    os.precision(3);
    os << std::scientific << x;
    return os.str();
}

void check(bool ok, std::string const& what) {
    if (!ok)
        ++failures;
    std::cout << (ok ? "ok:   " : "FAIL: ") << what << std::endl;
}

Dense random_matrix(int n, unsigned seed, double diagonal) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<double> value(-1., 1.);
    Dense a(n, std::vector<double>(n));
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            a[i][j] = value(gen) / std::sqrt(1. * n) + (i == j ? diagonal : 0.);
    return a;
}

Op multiply_by(Dense const& a) {
    return [&a](std::vector<double> const& x, std::vector<double>& y) {
        y.assign(x.size(), 0.);
        for (std::size_t i = 0; i < a.size(); ++i)
            for (std::size_t j = 0; j < x.size(); ++j)
                y[i] += a[i][j] * x[j];
    };
}

/// y = M^{-1} x by Gaussian elimination with partial pivoting on a copy of m (fine for test sizes).
Op solve_with(Dense const& m) {
    return [&m](std::vector<double> const& x, std::vector<double>& y) {
        int const n = static_cast<int>(m.size());
        Dense a = m;
        y = x;
        for (int k = 0; k < n; ++k) {
            int p = k;
            for (int i = k + 1; i < n; ++i)
                if (std::fabs(a[i][k]) > std::fabs(a[p][k]))
                    p = i;
            std::swap(a[k], a[p]);
            std::swap(y[k], y[p]);
            for (int i = k + 1; i < n; ++i) {
                double const f = a[i][k] / a[k][k];
                for (int j = k; j < n; ++j)
                    a[i][j] -= f * a[k][j];
                y[i] -= f * y[k];
            }
        }
        for (int i = n - 1; i >= 0; --i) {
            for (int j = i + 1; j < n; ++j)
                y[i] -= a[i][j] * y[j];
            y[i] /= a[i][i];
        }
    };
}

double true_residual(Dense const& a,
                     std::vector<double> const& x,
                     std::vector<double> const& b) {
    std::vector<double> ax;
    multiply_by(a)(x, ax);
    double r = 0., s = 0.;
    for (std::size_t i = 0; i < b.size(); ++i) {
        r += (ax[i] - b[i]) * (ax[i] - b[i]);
        s += b[i] * b[i];
    }
    return std::sqrt(r / s);
}

}  // namespace

int main() {
    int const n = 150;
    Dense const a = random_matrix(n, 7, 2.);
    std::vector<double> b(n);
    for (int i = 0; i < n; ++i)
        b[i] = std::sin(0.3 * i) + 0.1;
    Op const identity = [](std::vector<double> const& x,
                           std::vector<double>& y) {
        y = x;
    };
    std::vector<double> x;

    GmresResult r = gmres(multiply_by(a), identity, b, x, 1e-12, n);
    double const res = true_residual(a, x, b);
    check(r.converged && res < 1e-11,
          "no preconditioner: " + std::to_string(r.iterations) +
              " iterations, estimate " + sci(r.relative_residual) + ", true " +
              sci(res));

    r = gmres(multiply_by(a), solve_with(a), b, x, 1e-12, n);
    check(r.converged && r.iterations == 1 && true_residual(a, x, b) < 1e-11,
          "exact preconditioner: " + std::to_string(r.iterations) +
              " iteration(s), true residual " + sci(true_residual(a, x, b)));

    Dense stale = a;
    Dense const e = random_matrix(n, 8, 0.);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            stale[i][j] += 1e-3 * e[i][j] * std::sqrt(1. * n);
    r = gmres(multiply_by(a), solve_with(stale), b, x, 1e-12, n);
    check(r.converged && r.iterations <= 10 && true_residual(a, x, b) < 1e-11,
          "stale preconditioner: " + std::to_string(r.iterations) +
              " iterations, true residual " + sci(true_residual(a, x, b)));

    r = gmres(multiply_by(a), identity, b, x, 1e-14, 3);
    check(!r.converged && r.iterations == 3,
          "iteration cap: " + std::to_string(r.iterations) +
              " iterations, converged = " + std::to_string(r.converged));

    std::vector<double> const zero(n, 0.);
    r = gmres(multiply_by(a), identity, zero, x, 1e-12, n);
    bool all_zero = true;
    for (double v : x)
        all_zero = all_zero && v == 0.;
    check(r.converged && r.iterations == 0 && all_zero, "zero right-hand side");

    std::cout << (failures == 0 ? "test_gmres: PASS" : "test_gmres: FAIL")
              << " (" << failures << " failures)" << std::endl;
    return failures == 0 ? 0 : 1;
}
