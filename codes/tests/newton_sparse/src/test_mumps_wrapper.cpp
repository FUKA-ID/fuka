/*
 * T2: MumpsSolver on a synthetic system shaped like a Kadath Newton Jacobian.
 *
 * A has ndom dense diagonal blocks (spectral coupling inside a domain), sparse couplings between
 * neighbouring blocks (boundary matching), nglob dense bordering rows and columns (integral equations
 * and var_double unknowns), and a row permutation inside every block so the diagonal has zeros
 * (equations are not aligned with unknowns in Kadath). Every rank builds the same A from a fixed seed
 * and keeps the columns it would own in do_newton (1-D block cyclic).
 *
 * Checks, for centralized and distributed input (and ParMETIS ordering when nproc > 1):
 *  - relative residual and error against a known solution
 *  - the solution is identical on every rank and ignores the rhs given on non-root ranks
 *  - a second solve with the same instance and new values
 *  - the same matrix on a fresh instance (separates instance-reuse bugs from pivoting accuracy)
 *  - the third matrix needs the CNTL(1) = 0.1 default: with the MUMPS default 0.01 it only reaches ~1e-11
 *  - centralized input: repeated solves and other column distributions agree to roundoff (<= 1e-12)
 *  - a solution failing the residual check is rejected (kInaccurateSolution)
 *  - retry ladder: failing first attempt (too small configured budget) recovered by the next configuration
 *  - a structurally singular matrix (an unknown in no equation: empty column) is reported (-6 or -10)
 *
 * Run with 1, 2 and 4 MPI ranks; returns 0 if all checks pass (77 = skipped, built without MUMPS).
 */
#include <mpi.h>

#include <cmath>
#include <iostream>
#include <limits>
#include <map>
#include <random>
#include <sstream>
#include <vector>

#include "Linear_solvers/sparse_triplets.hpp"

#ifdef KADATH_USE_MUMPS
#include "Linear_solvers/mumps_solver.hpp"

using namespace Kadath::linear_solvers;

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
    }
}

/// Global matrix stored by columns: column j -> (row, value).
using Matrix = std::vector<std::map<int, double>>;

Matrix make_matrix(int ndom,
                   int block,
                   int nglob,
                   unsigned seed,
                   bool singular) {
    int const n = ndom * block + nglob;
    std::mt19937 gen(seed);
    std::uniform_real_distribution<double> value(-1., 1.);
    std::bernoulli_distribution coupling(0.05);
    Matrix a(n);

    for (int d = 0; d < ndom; ++d) {
        int const off = d * block;
        for (int jj = 0; jj < block; ++jj)
            for (int ii = 0; ii < block; ++ii) {
                int const permuted_row =
                    off +
                    (ii + 1) % block;  // zero-free diagonal needs a permutation
                double v = value(gen);
                if (ii == jj)
                    v +=
                        3. *
                        block;  // well conditioned after the permutation is undone
                if (permuted_row != off + jj)
                    a[off + jj][permuted_row] =
                        v;  // structural zero on the diagonal
            }
        if (d + 1 < ndom)
            for (int jj = 0; jj < block; ++jj)
                for (int ii = 0; ii < block; ++ii) {
                    if (coupling(gen))
                        a[off + block + jj][off + ii] = value(gen);
                    if (coupling(gen))
                        a[off + jj][off + block + ii] = value(gen);
                }
    }
    int const g0 = ndom * block;
    for (int g = 0; g < nglob; ++g) {
        for (int j = 0; j < n; ++j)
            a[j][g0 + g] = 0.1 * value(gen);  // dense bordering row
        for (int i = 0; i < g0; ++i)
            a[g0 + g][i] = 0.1 * value(gen);  // dense bordering column
        a[g0 + g][g0 + g] = 5. + value(gen);
    }
    if (singular)
        a[1].clear();  // unknown 1 appears in no equation
    return a;
}

std::vector<double> multiply(Matrix const& a, std::vector<double> const& x) {
    std::vector<double> b(x.size(), 0.);
    for (std::size_t j = 0; j < a.size(); ++j)
        for (auto const& [i, v] : a[j])
            b[i] += v * x[j];
    return b;
}

/// This rank's columns of @p a in a 1-D block-cyclic distribution (block 0 = the ScaLAPACK-path block size).
SparseTriplets local_columns(Matrix const& a, int bsize = 0) {
    int const n = static_cast<int>(a.size());
    if (bsize == 0)
        bsize = newton_block_size(n, nproc_);
    SparseTriplets t;
    for (int j : block_cyclic_columns(n, bsize, rank_, nproc_))
        for (auto const& [i, v] : a[j]) {
            t.rows.push_back(i + 1);
            t.cols.push_back(j + 1);
            t.values.push_back(v);
        }
    return t;
}

double max_abs(std::vector<double> const& v) {
    double m = 0.;
    for (double x : v)
        m = std::max(m, std::fabs(x));
    return m;
}

void solve_and_check(MumpsSolver& solver,
                     Matrix const& a,
                     std::string const& label) {
    int const n = static_cast<int>(a.size());
    std::vector<double> x_true(n);
    for (int i = 0; i < n; ++i)
        x_true[i] = std::sin(0.37 * i) + 0.5;
    std::vector<double> const b = multiply(a, x_true);

    std::vector<double> x =
        (rank_ == 0)
            ? b
            : std::vector<double>(n, std::numeric_limits<double>::quiet_NaN());
    SparseTriplets local = local_columns(a);
    int const status = solver.solve(n, local, x);
    check(status == 0, label + ": solve status " + std::to_string(status));
    if (status != 0)
        return;

    std::vector<double> r = multiply(a, x);
    std::vector<double> err(n);
    for (int i = 0; i < n; ++i) {
        r[i] -= b[i];
        err[i] = x[i] - x_true[i];
    }
    double const rel_res = max_abs(r) / max_abs(b);
    double const rel_err = max_abs(err) / max_abs(x_true);
    check(rel_res < 1e-12, label + ": relative residual " + sci(rel_res));
    check(rel_err < 1e-10, label + ": relative error " + sci(rel_err));

    std::vector<double> x_root = x;
    MPI_Bcast(x_root.data(), n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    bool same = true;
    for (int i = 0; i < n; ++i)
        same = same && (x_root[i] == x[i]);
    check(same, label + ": identical solution on every rank");

    if (rank_ == 0)
        std::cout << label << ": n = " << n << ", nnz = " << solver.stats().nnz
                  << ", factors = " << solver.stats().factor_entries
                  << ", rel. residual = " << sci(rel_res)
                  << ", rel. error = " << sci(rel_err) << std::endl;
}

void run_case(MumpsOptions options, std::string const& label) {
    options.verbosity = 1;
    MumpsSolver solver(MPI_COMM_WORLD, options);
    solve_and_check(solver,
                    make_matrix(6, 40, 3, 11, false),
                    label + " [first]");
    solve_and_check(solver,
                    make_matrix(6, 40, 3, 12, false),
                    label + " [reuse instance]");
    solve_and_check(solver,
                    make_matrix(9, 25, 2, 13, false),
                    label + " [reuse, new size]");
    MumpsSolver fresh(MPI_COMM_WORLD, options);
    solve_and_check(fresh,
                    make_matrix(9, 25, 2, 13, false),
                    label + " [fresh, same matrix]");
}

/// Kept analysis (MumpsOptions::reuse_analysis): new values on the analysed pattern and a subset of it reuse the
/// analysis (the subset with explicit zeros), an entry outside the pattern or reuse_analysis = false analyse again.
void check_kept_analysis(MumpsOptions options, std::string const& label) {
    options.verbosity = 1;
    Matrix a = make_matrix(6, 40, 3, 21, false);
    MumpsSolver solver(MPI_COMM_WORLD, options);
    solve_and_check(solver, a, label + " [analysed]");
    check(!solver.stats().analysis_reused, label + ": first solve analyses");

    int k = 0;
    for (auto& column : a)
        for (auto& [i, v] : column)
            v *= 1. + 0.01 * std::sin(1. * ++k);
    solve_and_check(solver, a, label + " [new values]");
    check(solver.stats().analysis_reused == options.reuse_analysis,
          label + ": new values keep the analysis iff reuse_analysis");

    Matrix subset = a;
    int removed = 0;
    for (std::size_t j = 0; j + 1 < subset.size() && removed < 20; j += 7)
        for (auto it = subset[j].begin(); it != subset[j].end(); ++it)
            if (std::fabs(it->second) < 0.5 &&
                it->first != static_cast<int>(j)) {
                subset[j].erase(it);
                ++removed;
                break;
            }
    solve_and_check(solver, subset, label + " [subset of the pattern]");
    check(solver.stats().analysis_reused == options.reuse_analysis,
          label + ": a subset keeps the analysis iff reuse_analysis");
    if (options.reuse_analysis)
        check(solver.stats().pattern_zeros == removed,
              label + ": one explicit zero per removed entry");

    Matrix grown = a;
    grown[3][static_cast<int>(a.size()) - 1 - 50] = 0.25;
    if (grown[3].size() == a[3].size())
        grown[5][static_cast<int>(a.size()) - 1 - 60] = 0.25;
    solve_and_check(solver, grown, label + " [entry outside the pattern]");
    check(!solver.stats().analysis_reused,
          label + ": a new entry gives a new analysis");
    // other = subset plus a different new entry: the union analysis (not ParMETIS) then also covers grown.
    Matrix other = subset;
    other[8][static_cast<int>(a.size()) - 1 - 70] = 0.25;
    solve_and_check(solver, other, label + " [other new entry]");
    check(!solver.stats().analysis_reused,
          label + ": the other new entry gives a new analysis");
    solve_and_check(solver, grown, label + " [back to the grown matrix]");
    check(solver.stats().analysis_reused ==
              (options.reuse_analysis && !options.parallel_analysis),
          label + ": the union pattern covers the earlier matrix");

    solver.release_factors();
    check(!solver.has_factors(), label + ": release_factors frees the factors");
    solve_and_check(solver, grown, label + " [after release_factors]");
    check(solver.stats().analysis_reused == options.reuse_analysis,
          label + ": the analysis survives release_factors");
}

/// Different column distributions hand MUMPS the same entries in a different order; solutions for different
/// distributions, and repeated solves, must agree to roundoff (bit identity is not required); the measured
/// differences are printed.
void check_distribution_independence(MumpsOptions options,
                                     std::string const& label) {
    options.verbosity = 1;
    Matrix const a = make_matrix(6, 40, 3, 31, false);
    int const n = static_cast<int>(a.size());
    std::vector<double> x_true(n);
    for (int i = 0; i < n; ++i)
        x_true[i] = std::cos(0.21 * i);
    std::vector<double> const b = multiply(a, x_true);

    std::vector<std::vector<double>> solutions;
    for (int bsize :
         {0, 0, 1, 7}) {  // the repeated 0 measures run-to-run reproducibility
        MumpsSolver solver(MPI_COMM_WORLD, options);
        std::vector<double> x = b;
        SparseTriplets local = local_columns(a, bsize);
        check(solver.solve(n, local, x) == 0,
              label + ": solve with column block " + std::to_string(bsize));
        solutions.push_back(x);
    }
    auto rel = [](std::vector<double> const& x, std::vector<double> const& y) {
        double d = 0., s = 0.;
        for (std::size_t i = 0; i < x.size(); ++i) {
            d = std::max(d, std::fabs(x[i] - y[i]));
            s = std::max(s, std::fabs(y[i]));
        }
        return d / s;
    };
    double const repeat = rel(solutions[1], solutions[0]);
    double across = 0.;
    for (std::size_t k = 2; k < solutions.size(); ++k)
        across = std::max(across, rel(solutions[k], solutions[0]));
    if (rank_ == 0)
        std::cout << label << ": repeated solve differs by " << sci(repeat)
                  << (repeat == 0. ? " (bitwise)" : "")
                  << ", other column distributions by " << sci(across)
                  << (across == 0. ? " (bitwise)" : "") << std::endl;
    check(repeat <= 1e-12 && across <= 1e-12,
          label + ": solutions agree to roundoff (repeat " + sci(repeat) +
              ", distributions " + sci(across) + ")");
}

/// A solution failing the residual check must be rejected (tolerance 0 cannot be met in floating point).
void check_residual_rejection(MumpsOptions options, std::string const& label) {
    options.verbosity = 0;
    options.residual_tolerance = 0.;
    Matrix const a = make_matrix(4, 20, 1, 41, false);
    int const n = static_cast<int>(a.size());
    std::vector<double> x(n, 1.);
    SparseTriplets local = local_columns(a);
    MumpsSolver solver(MPI_COMM_WORLD, options);
    int const status = solver.solve(n, local, x);
    check(status == kInaccurateSolution &&
              solver.stats().relative_residual > 0.,
          label + ": inaccurate solution rejected, status " +
              std::to_string(status) + ", residual " +
              sci(solver.stats().relative_residual));
}

/// Retry ladder: a configured budget too small for the matrix (no workspace retries) must fail the first attempt;
/// the recovery attempts (automatic budget) must then succeed on the same triplets.
void check_retry_ladder(MumpsOptions base, std::string const& label) {
    base.verbosity = 0;
    MumpsOptions no_retry = base;
    no_retry.retry = false;
    check(mumps_retry_ladder(no_retry).size() == 1,
          label + ": retry = false gives a single attempt");
    auto const ladder = mumps_retry_ladder(base);
    check(ladder.size() == 3 && ladder[1].options.permutation == 1 &&
              ladder[1].options.centralized && !ladder[2].options.centralized &&
              ladder[1].options.max_mem_mb == 0,
          label +
              ": ladder = configured, structural permutation, distributed "
              "input (automatic budget)");

    base.max_mem_mb = 1;
    base.max_mem_retries = 0;
    auto const tight = mumps_retry_ladder(base);
    Matrix const a = make_matrix(12, 80, 3, 51, false);
    int const n = static_cast<int>(a.size());
    std::vector<double> x_true(n, 1.);
    std::vector<double> const b = multiply(a, x_true);
    SparseTriplets local = local_columns(a);

    std::vector<int> status;
    for (auto const& attempt : tight) {
        MumpsSolver solver(MPI_COMM_WORLD, attempt.options);
        std::vector<double> x = b;
        status.push_back(solver.solve(n, local, x));
        if (status.back() == 0)
            break;
    }
    check(status.size() >= 2 && status[0] != 0 && status.back() == 0,
          label + ": 1 MB budget fails attempt 1 (status " +
              std::to_string(status[0]) + "), attempt " +
              std::to_string(status.size()) + " succeeds");
}

void run_singular(MumpsOptions options, std::string const& label) {
    options.verbosity = 0;
    MumpsSolver solver(MPI_COMM_WORLD, options);
    Matrix const a = make_matrix(4, 20, 1, 21, true);
    int const n = static_cast<int>(a.size());
    std::vector<double> x(n, 1.);
    SparseTriplets local = local_columns(a);
    int const status = solver.solve(n, local, x);
    check(status == -10 || status == -6,
          label + ": singular matrix reported, status " +
              std::to_string(status));
}

}  // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
    MPI_Comm_size(MPI_COMM_WORLD, &nproc_);

    MumpsOptions centralized;
    centralized.centralized = true;
    MumpsOptions distributed;
    distributed.centralized = false;

    run_case(centralized, "centralized/METIS");
    run_case(distributed, "distributed/METIS");
    if (nproc_ > 1) {
        MumpsOptions parallel = distributed;
        parallel.parallel_analysis = true;
        run_case(parallel, "distributed/ParMETIS");
    }
    check_kept_analysis(centralized, "centralized kept analysis");
    check_kept_analysis(distributed, "distributed kept analysis");
    if (nproc_ > 1) {
        MumpsOptions parallel = distributed;
        parallel.parallel_analysis = true;
        check_kept_analysis(parallel, "ParMETIS kept analysis");
    }
    {
        MumpsOptions off = centralized;
        off.reuse_analysis = false;
        check_kept_analysis(off, "no kept analysis");
    }
    check_distribution_independence(centralized,
                                    "centralized distribution independence");
    check_residual_rejection(centralized, "centralized residual check");
    check_retry_ladder(centralized, "retry ladder");
    check_residual_rejection(distributed, "distributed residual check");
    run_singular(centralized, "centralized singular");
    run_singular(distributed, "distributed singular");

    if (rank_ == 0)
        std::cout << (failures == 0 ? "test_mumps_wrapper: PASS"
                                    : "test_mumps_wrapper: FAIL")
                  << " (" << failures << " failures, " << nproc_ << " ranks)"
                  << std::endl;
    MPI_Finalize();
    return failures == 0 ? 0 : 1;
}

#else
int main() {
    std::cout << "test_mumps_wrapper: SKIP (built without KADATH_USE_MUMPS)"
              << std::endl;
    return 77;
}
#endif
