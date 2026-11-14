/*
 * T3/T4: sparse (MUMPS) vs dense (ScaLAPACK) Newton on a real Kadath System_of_eqs.
 *
 * Problem (tutorial 6 plus an unknown constant): shells 1-2 and the compactified domain of a Space_spheric,
 *   Lap(P) + grad(P).grad(P) = 0,  dn(P) + lam / a = 0 at r = a,  P = 0 at infinity,
 *   P(M) = log(1 + a / r_M) at one point M of domain 1  (an eq_int row; lam is a var_double).
 * exp(P) is harmonic, so P = log(1 + c / r) and lam = c / (a + c); the point condition selects c = a,
 * hence lam = 1/2 and P = log(1 + a / r) exactly.
 *
 * Checks:
 *  T3  do_col_J is deterministic (bitwise identical when recomputed, and on every rank); reports nnz fraction
 *  T3b the matrix-free do_JX(e_j) equals do_col_J(j), for the number unknown (lam) and field columns, after
 *      do_col_J calls that leave other derivative state behind; with lam in the last domain too
 *  T3d do_col_J skipping definitions/metric of untouched domains is bitwise identical to recomputing all
 *  T3c the same with variable domains (shape unknowns): Space_spheric_adapted (adapted shells, Galerkin
 *      shape coefficients) and Space_adapted_bh (homothetic shells, as in the BH solver)
 *  T4a one Newton update: MUMPS (centralized and distributed input) vs ScaLAPACK, relative difference < 1e-10
 *  T4b full solves with KADATH_LINSOLVE=mumps and =scalapack from the same guess: same iteration count,
 *      matching error history, same lam and P, and both agree with the analytic solution
 *  T4d Newton-Krylov (GMRES + kept MUMPS factors) converges to the exact-Newton solution within tolerance
 *  T4f T4d with KADATH_KRYLOV_REFRESH=lazy (GMRES trial before refactoring): same solution
 *  T4e T4d with a preconditioner from a coarsely dropped Jacobian (KADATH_KRYLOV_DROP=1e-3): same solution
 *  T4c a rejected MUMPS solution (residual tolerance 0) falls back to ScaLAPACK step by step: bitwise identical
 *
 * Run with 1 and 4 MPI ranks; returns 0 if all checks pass (77 = skipped, built without MUMPS).
 */
#include <mpi.h>

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "Linear_solvers/newton_linear_solve.hpp"
#ifdef KADATH_USE_MUMPS
#include "Linear_solvers/mumps_solver.hpp"
#endif
#include "Linear_solvers/sparse_triplets.hpp"
#include "kadath.hpp"
#include "adapted.hpp"
#include "adapted_bh.hpp"

using namespace Kadath;
using namespace Kadath::linear_solvers;

#ifdef KADATH_USE_MUMPS
namespace {

int rank_ = 0;
int nproc_ = 1;
int failures = 0;

std::string sci(double x) {
    std::ostringstream os;
    os.precision(3);
    os << std::scientific << x;
    return os.str();
}

void check(bool ok, std::string const& what) {
    int local = ok ? 0 : 1, any = 0;
    MPI_Allreduce(&local, &any, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    if (any) {
        ++failures;
        if (rank_ == 0)
            std::cout << "FAIL: " << what << std::endl;
    } else if (rank_ == 0)
        std::cout << "ok:   " << what << std::endl;
}

Dim_array resolution() {
    Dim_array res(3);
    res.set(0) = 11;
    res.set(1) = 7;
    res.set(2) = 6;
    return res;
}

Point origin() {
    Point center(3);
    for (int i = 1; i <= 3; ++i)
        center.set(i) = 0.;
    return center;
}

Array<double> domain_bounds() {
    Array<double> bounds(3);
    bounds.set(0) = 1.;
    bounds.set(1) = 2.;
    bounds.set(2) = 4.;
    return bounds;
}

Point cartesian(double x, double y, double z) {
    Point p(3);
    p.set(1) = x;
    p.set(2) = y;
    p.set(3) = z;
    return p;
}

double radius(Point const& p) {
    return std::sqrt(p(1) * p(1) + p(2) * p(2) + p(3) * p(3));
}

/// Owns the space, the unknowns and the system (the system keeps references to the unknowns).
struct Problem {
    Space_spheric space;
    Scalar field;
    double lam = 0.3;
    double a = 0.;
    System_of_eqs syst;

    /// @param lam_outer outer BC P + lam - 1/2 = 0 instead of P = 0 (same solution, lam = 1/2): puts the
    /// number unknown into the last domain of the system, where do_JX once left its derivative stale.
    explicit Problem(bool lam_outer = false)
        : space(CHEB_TYPE, origin(), resolution(), domain_bounds()),
          field(space),
          syst(space, 1, 3) {
        field = 1.;
        field.std_base();

        Index pos(space.get_domain(1)->get_nbr_points());
        a = space.get_domain(1)->get_radius()(pos);
        Point const point = cartesian(1.0, 0.7, 0.5);

        syst.add_var("P", field);
        syst.add_var("lam", lam);
        syst.add_cst("a", a);
        syst.add_cst("Pref", std::log(1. + a / radius(point)));
        syst.add_def("dP2 = scal(grad(P), grad(P))");

        for (int d = 1; d <= 3; ++d)
            syst.add_eq_inside(d, "Lap(P) + dP2 = 0");
        syst.add_eq_bc(1, INNER_BC, "dn(P) + lam / a = 0");
        syst.add_eq_bc(3, OUTER_BC, lam_outer ? "P + lam - 0.5 = 0" : "P = 0");
        for (int d = 1; d < 3; ++d) {
            syst.add_eq_matching(d, OUTER_BC, "P");
            syst.add_eq_matching(d, OUTER_BC, "dn(P)");
        }
        syst.add_eq_point(1, "P - Pref", point);
    }
};

bool bitwise_equal(Array<double> const& x, Array<double> const& y) {
    if (x.get_size(0) != y.get_size(0))
        return false;
    for (int i = 0; i < x.get_size(0); ++i)
        if (!(x(i) == y(i)))
            return false;
    return true;
}

double rel_diff(Array<double> const& x, Array<double> const& ref) {
    double diff = 0., scale = 0.;
    for (int i = 0; i < ref.get_size(0); ++i) {
        diff = std::max(diff, std::fabs(x(i) - ref(i)));
        scale = std::max(scale, std::fabs(ref(i)));
    }
    return scale > 0. ? diff / scale : diff;
}

void test_jacobian_columns() {
    Problem problem;
    System_of_eqs& syst = problem.syst;
    syst.vars_to_terms();
    Array<double> const second(syst.sec_member());
    int const n = syst.get_nbr_unknowns();
    int const m = second.get_size(0);
    check(n == m,
          "square system: n = " + std::to_string(n) +
              ", m = " + std::to_string(m));

    std::vector<int> const probes = {0, 1, n / 2, n - 1};
    std::vector<Array<double>> first;
    for (int c : probes)
        first.push_back(syst.do_col_J(c));
    bool deterministic = true;
    for (std::size_t k = 0; k < probes.size(); ++k)
        deterministic = deterministic &&
                        bitwise_equal(syst.do_col_J(probes[k]), first[k]);
    check(deterministic,
          "T3 do_col_J bitwise identical when recomputed in a different order");

    bool same_on_ranks = true;
    for (auto const& col : first) {
        Array<double> root(col);
        MPI_Bcast(root.set_data(),
                  root.get_size(0),
                  MPI_DOUBLE,
                  0,
                  MPI_COMM_WORLD);
        same_on_ranks = same_on_ranks && bitwise_equal(root, col);
    }
    check(same_on_ranks, "T3 do_col_J bitwise identical on every rank");

    SparseTriplets local;
    for (int c :
         block_cyclic_columns(n, newton_block_size(n, nproc_), rank_, nproc_))
        append_column(syst.do_col_J(c), c, local);
    std::int64_t nnz = 0, local_nnz = local.size();
    MPI_Allreduce(&local_nnz, &nnz, 1, MPI_INT64_T, MPI_SUM, MPI_COMM_WORLD);
    check(nnz > 0 && nnz < static_cast<std::int64_t>(n) * n,
          "T3 Jacobian nnz = " + std::to_string(nnz) + " (" +
              sci(100. * nnz / (1. * n * n)) + " % of dense)");
}

/// T3d: do_col_J with skip_clean_domains (definitions and metric recomputed only in touched domains) must be
/// bitwise identical to recomputing everything, for columns visited in a scrambled order with do_JX and
/// sec_member calls in between (they invalidate the tracked state).
void check_skip_clean(System_of_eqs& syst, std::string const& label) {
    int const n = syst.get_nbr_unknowns();
    std::vector<int> columns;
    for (int c = 0; c < std::min(n, 4); ++c)
        columns.push_back(c);
    for (int k = 1; k <= 24; ++k)
        columns.push_back(static_cast<int>((static_cast<long long>(n - 1) * k) / 24));
    syst.skip_clean_domains = false;
    syst.sec_member();
    std::vector<Array<double>> reference;
    for (int c : columns)
        reference.push_back(syst.do_col_J(c));

    syst.skip_clean_domains = true;
    syst.sec_member();
    bool same = true;
    std::size_t const m = columns.size();
    for (std::size_t k = 0; k < 3 * m; ++k) {
        std::size_t const idx = (k * 7 + 3) % m;  // 7 is coprime with m = 28: every column, scrambled, 3 times
        if (k == m) {
            Array<double> e(n);
            e = 1.;
            syst.do_JX(e);
        }
        if (k == 2 * m)
            syst.sec_member();
        same = same && bitwise_equal(syst.do_col_J(columns[idx]), reference[idx]);
    }
    check(same, "T3d " + label + ": do_col_J with skipped clean domains bitwise identical (" +
                    std::to_string(m) + " columns, 3 scrambled passes)");
}

/// T3b: J e_j from do_JX must equal the assembled column j (Newton-Krylov relies on it), whatever the previous
/// do_col_J left in the Term_eq derivatives (column 0 is lam: no variable domains, the number comes first).
void test_jx_columns(bool lam_outer) {
    Problem problem(lam_outer);
    System_of_eqs& syst = problem.syst;
    syst.vars_to_terms();
    Array<double> const second(syst.sec_member());
    int const n = syst.get_nbr_unknowns();
    std::string const t = std::string("T3b") + (lam_outer ? " (lam in last domain)" : "");
    // (column assembled before, column probed): every number/field combination of stale state
    std::vector<std::pair<int, int>> const cases = {{n - 1, 0}, {0, 1}, {0, n / 2}, {1, 0}, {0, n - 1}};
    double worst = 0.;
    for (auto const& [before, probe] : cases) {
        syst.do_col_J(before);
        Array<double> e(n);
        e = 0.;
        e.set(probe) = 1.;
        Array<double> const jx(syst.do_JX(e));
        Array<double> const col(syst.do_col_J(probe));
        double const d = rel_diff(jx, col);
        worst = std::max(worst, d);
        if (rank_ == 0 && d > 1e-13)
            std::cout << t << " do_JX(e_" << probe << ") after do_col_J(" << before
                      << ") differs from do_col_J(" << probe << "): " << sci(d) << std::endl;
    }
    check(worst <= 1e-13, t + " do_JX(e_j) equals do_col_J(j): max rel. diff = " + sci(worst));
    check_skip_clean(syst, lam_outer ? "spheric, lam in last domain" : "spheric");
}

/// T3c: do_JX(e_j) == do_col_J(j) when domains 1 and 2 have shape unknowns (the first columns). The field is
/// non-constant so that moving the collocation points changes it (update_term_eq is not trivially zero).
template <class SpaceT>
void test_jx_variable_domains(std::string const& label) {
    std::vector<double> const bounds = {1., 2., 4., 6.};  // nucleus, 2 variable shells, shell, compact
    SpaceT space(CHEB_TYPE, origin(), resolution(), bounds);
    int const ndom = space.get_nbr_domains();
    Scalar field(space);
    for (int d = 0; d < ndom - 1; ++d) {
        Val_domain const r(space.get_domain(d)->get_radius());
        field.set_domain(d) = 1. + 0.1 * r * r;
    }
    field.set_domain(ndom - 1) = 1.;
    field.std_base();
    double scale = 1.;
    System_of_eqs syst(space, 1, ndom - 2);
    syst.add_var("P", field);
    syst.add_var("lam", scale);
    for (int d = 1; d <= ndom - 2; ++d)
        syst.add_eq_inside(d, "Lap(P) + lam * P = 0");
    syst.add_eq_bc(1, INNER_BC, "dn(P) = 0");
    for (int d = 1; d < ndom - 2; ++d) {
        syst.add_eq_matching(d, OUTER_BC, "P");
        syst.add_eq_matching(d, OUTER_BC, "dn(P)");
    }
    syst.add_eq_bc(ndom - 2, OUTER_BC, "P = 1");
    syst.vars_to_terms();
    Array<double> const second(syst.sec_member());
    int const n = syst.get_nbr_unknowns();
    int const nvd = space.nbr_unknowns_from_variable_domains();

    std::vector<int> columns;
    for (int c = 0; c < std::min(n, nvd + 3); ++c)  // every shape unknown, lam and the first field columns
        columns.push_back(c);
    for (int k = 1; k <= 8; ++k)
        columns.push_back(static_cast<int>((static_cast<long long>(n - 1) * k) / 8));
    double worst = 0.;
    int worst_col = -1;
    for (std::size_t k = 0; k < columns.size(); ++k) {
        syst.do_col_J(columns[(k + 1) % columns.size()]);  // leave another column's derivative state behind
        Array<double> e(n);
        e = 0.;
        e.set(columns[k]) = 1.;
        Array<double> const jx(syst.do_JX(e));
        Array<double> const col(syst.do_col_J(columns[k]));
        double const d = rel_diff(jx, col);
        if (d > worst) {
            worst = d;
            worst_col = columns[k];
        }
    }
    // A combination of shape unknowns at once (the Galerkin j = 0 coefficient sums over j): do_JX(v) vs
    // sum_j v_j do_col_J(j) over the shape columns.
    Array<double> v(n), sum(second.get_size(0));
    v = 0.;
    sum = 0.;
    for (int c = 0; c < nvd; ++c) {
        v.set(c) = 1. + 0.37 * c;
        Array<double> const col(syst.do_col_J(c));
        for (int i = 0; i < sum.get_size(0); ++i)
            sum.set(i) += v(c) * col(i);
    }
    double const combo = nvd > 0 ? rel_diff(syst.do_JX(v), sum) : 0.;
    check(nvd > 0 && worst <= 1e-12 && combo <= 1e-12,
          "T3c " + label + ": " + std::to_string(nvd) + " shape unknowns, do_JX(e_j) vs do_col_J(j) max rel. diff = " +
              sci(worst) + " (column " + std::to_string(worst_col) + "), shape combination " + sci(combo));
    check_skip_clean(syst, label);
}

void test_single_update() {
    Problem problem;
    System_of_eqs& syst = problem.syst;
    syst.vars_to_terms();
    Array<double> const second(syst.sec_member());

    Array<double> const dense(
        newton_update_scalapack(syst, second, System_of_eqs::NEWTON_RAPHSON));
    for (char const* input : {"centralized", "distributed"}) {
        setenv("KADATH_MUMPS_INPUT", input, 1);
        Array<double> const sparse(newton_update_mumps(syst, second));
        double const d = rel_diff(sparse, dense);
        check(d < 1e-10,
              std::string("T4a Newton update, ") + input +
                  " input: rel. diff = " + sci(d));
    }
    unsetenv("KADATH_MUMPS_INPUT");
}

struct SolveResult {
    int iterations = 0;
    bool converged = false;
    std::vector<double> errors;
    double lam = 0.;
    double a = 0.;
    std::vector<double> samples;
};

std::vector<Point> sample_points() {
    return {cartesian(1.1, 0.2, 0.3),
            cartesian(-0.5, 2.5, 0.8),
            cartesian(3.0, -6.0, 7.0)};
}

SolveResult solve_with(char const* backend) {
    setenv("KADATH_LINSOLVE", backend, 1);
    Problem problem;
    SolveResult result;
    result.a = problem.a;
    double error = 0.;
    for (int it = 0; it < 20 && !result.converged; ++it) {
        result.converged = problem.syst.do_newton(1e-11, error);
        result.errors.push_back(error);
        result.iterations = it;
    }
    result.lam = problem.lam;
    for (auto const& p : sample_points())
        result.samples.push_back(problem.field.val_point(p));
    unsetenv("KADATH_LINSOLVE");
    return result;
}

/// Returns the ScaLAPACK reference solve for test_fallback.
SolveResult test_full_solve() {
    SolveResult const sparse = solve_with("mumps");
    SolveResult dense = solve_with("scalapack");

    check(sparse.converged,
          "T4b MUMPS solve converged in " + std::to_string(sparse.iterations) +
              " steps");
    check(dense.converged,
          "T4b ScaLAPACK solve converged in " +
              std::to_string(dense.iterations) + " steps");
    check(sparse.iterations == dense.iterations,
          "T4b same number of Newton steps");

    bool history = sparse.errors.size() == dense.errors.size();
    for (std::size_t i = 0; history && i < dense.errors.size(); ++i)
        history = std::fabs(sparse.errors[i] - dense.errors[i]) <=
                  1e-8 * dense.errors[i] + 1e-12;
    check(history, "T4b error history agrees (rel. 1e-8 or abs. 1e-12)");

    check(std::fabs(sparse.lam - dense.lam) < 1e-12,
          "T4b lam agrees: |dlam| = " + sci(std::fabs(sparse.lam - dense.lam)));
    check(std::fabs(sparse.lam - 0.5) < 1e-6,
          "T4b lam analytic: |lam - 1/2| = " +
              sci(std::fabs(sparse.lam - 0.5)));

    auto const points = sample_points();
    for (std::size_t k = 0; k < points.size(); ++k) {
        double const exact = std::log(1. + sparse.a / radius(points[k]));
        check(std::fabs(sparse.samples[k] - dense.samples[k]) < 1e-10,
              "T4b P agrees at r = " + sci(radius(points[k])));
        check(std::fabs(sparse.samples[k] - exact) < 1e-6,
              "T4b P analytic at r = " + sci(radius(points[k])) +
                  ": err = " + sci(std::fabs(sparse.samples[k] - exact)));
    }
    return dense;
}

/// T4c: with KADATH_MUMPS_RESIDUAL_TOL=0 every attempt of the retry ladder is rejected by the residual check; with
/// KADATH_MUMPS_FALLBACK=scalapack each Newton step must then be redone densely, reproducing the ScaLAPACK solve.
void test_fallback(SolveResult const& dense) {
    setenv("KADATH_MUMPS_RESIDUAL_TOL", "0", 1);
    setenv("KADATH_MUMPS_FALLBACK", "scalapack", 1);
    {
        Problem problem;
        problem.syst.vars_to_terms();
        Array<double> const second(problem.syst.sec_member());
        int status = 0;
        newton_update_mumps(problem.syst, second, &status);
        check(status == kInaccurateSolution,
              "T4c MUMPS solution rejected by the residual check: status " +
                  std::to_string(status));
    }
    SolveResult const fallback = solve_with("mumps");
    unsetenv("KADATH_MUMPS_RESIDUAL_TOL");
    unsetenv("KADATH_MUMPS_FALLBACK");

    check(fallback.converged && fallback.iterations == dense.iterations,
          "T4c fallback solve converged in the same " +
              std::to_string(dense.iterations) + " steps");
    bool same = fallback.errors == dense.errors && fallback.lam == dense.lam &&
                fallback.samples == dense.samples;
    check(same,
          "T4c fallback solve bitwise identical to the ScaLAPACK solve "
          "(errors, lam, P)");
}

/// T4d: Newton-Krylov (KADATH_NEWTON_METHOD=krylov) must converge to the exact-Newton solution within the Newton
/// tolerance (not bitwise: its iterates differ), with at least one step solved by preconditioned GMRES.
void test_krylov(SolveResult const& dense, std::string const& t, char const* drop = nullptr,
                 bool lazy = false) {
    reset_newton_krylov();
    setenv("KADATH_NEWTON_METHOD", "krylov", 1);
    if (drop != nullptr)
        setenv("KADATH_KRYLOV_DROP", drop, 1);
    if (lazy)
        setenv("KADATH_KRYLOV_REFRESH", "lazy", 1);
    SolveResult const krylov = solve_with("mumps");
    unsetenv("KADATH_NEWTON_METHOD");
    unsetenv("KADATH_KRYLOV_DROP");
    unsetenv("KADATH_KRYLOV_REFRESH");
    NewtonKrylovStats const stats = newton_krylov_stats();
    if (rank_ == 0)
        std::cout << t << " Newton-Krylov: " << krylov.iterations
                  << " Newton steps, " << stats.krylov_steps << " by GMRES ("
                  << stats.krylov_iterations << " iterations), "
                  << stats.refactorizations << " refactorizations, "
                  << stats.gmres_failures << " GMRES failures, "
                  << stats.trial_successes << " refactorizations avoided by trials" << std::endl;

    check(krylov.converged,
          t + " Newton-Krylov converged in " +
              std::to_string(krylov.iterations) + " steps");
    check(stats.krylov_steps >= 1,
          t + " at least one step solved by GMRES with a kept preconditioner");
    check(std::fabs(krylov.lam - dense.lam) < 1e-9,
          t + " lam agrees with exact Newton: |dlam| = " +
              sci(std::fabs(krylov.lam - dense.lam)));
    double worst = 0.;
    for (std::size_t k = 0; k < dense.samples.size(); ++k)
        worst = std::max(worst,
                         std::fabs(krylov.samples[k] - dense.samples[k]) /
                             std::fabs(dense.samples[k]));
    check(worst < 1e-9,
          t + " P agrees with exact Newton: max rel. diff = " + sci(worst));
    // No reset here: the kept MUMPS instance must be released inside MPI_Finalize (MPI_COMM_SELF attribute);
    // releasing it after MPI_Finalize makes the process exit with an MPI error and the test fail.
}

}  // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
    MPI_Comm_size(MPI_COMM_WORLD, &nproc_);

    test_jacobian_columns();
    test_jx_columns(false);
    test_jx_columns(true);
    test_jx_variable_domains<Space_spheric_adapted>("adapted shells");
    test_jx_variable_domains<Space_adapted_bh>("homothetic shells (BH)");
    test_single_update();
    SolveResult const dense = test_full_solve();
    test_fallback(dense);
    test_krylov(dense, "T4e", "1e-3");
    test_krylov(dense, "T4f", nullptr, true);
    test_krylov(dense, "T4d");  // last: exercises the release in MPI_Finalize

    if (rank_ == 0)
        std::cout << (failures == 0 ? "test_newton_kadath: PASS"
                                    : "test_newton_kadath: FAIL")
                  << " (" << failures << " failures, " << nproc_ << " ranks)"
                  << std::endl;
    MPI_Finalize();
    return failures == 0 ? 0 : 1;
}

#else
int main() {
    std::cout << "test_newton_kadath: SKIP (built without KADATH_USE_MUMPS)"
              << std::endl;
    return 77;
}
#endif
