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
 *  - distributed input with the column permutation computed here (matching_entries), incl. ParMETIS
 *  - a structurally singular matrix (an unknown in no equation: empty column) is reported (-6, -10, or
 *    kNoMatching with the matching)
 *
 * Run with 1, 2 and 4 MPI ranks; returns 0 if all checks pass (77 = skipped, built without MUMPS).
 */
#include <mpi.h>

#include <cmath>
#include <cstdlib>
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
    MumpsOptions matched_base = base;
    matched_base.centralized = false;
    matched_base.matching_entries = 16;
    auto const matched_ladder = mumps_retry_ladder(matched_base);
    check(matched_ladder.size() == 4 && matched_ladder[0].options.matching_entries > 0 &&
              matched_ladder[1].options.centralized && matched_ladder[1].options.permutation != 1 &&
              matched_ladder[2].options.centralized && matched_ladder[2].options.permutation == 1 &&
              !matched_ladder[3].options.centralized && matched_ladder[3].options.matching_entries == 0,
          label + ": ladder from distributed input = matched distributed, centralized, structural, "
                  "distributed without permutation");
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

/// Automatic out-of-core: with 1 MB of "available" memory per rank the factors cannot fit in core; auto must
/// switch to disk (decision only: the workspace is set explicitly) and still solve exactly, never must return kInsufficientMemory, and with the real memory
/// auto must stay in core.
void check_out_of_core_decision(MumpsOptions options, std::string const& label) {
    options.verbosity = 1;
    Matrix const a = make_matrix(6, 40, 3, 51, false);
    int const n = static_cast<int>(a.size());
    std::vector<double> x_true(n, 1.);
    std::vector<double> const b = multiply(a, x_true);
    {
        MumpsOptions o = options;
        o.out_of_core = kOutOfCoreAuto;
        std::vector<double> x = b;
        SparseTriplets local = local_columns(a);
        MumpsSolver solver(MPI_COMM_WORLD, o);
        int const status = solver.solve(n, local, x);
        check(status == 0 && !solver.stats().out_of_core,
              label + " auto, enough memory: in core (status " + std::to_string(status) + ")");
    }
    {
        MumpsOptions o = options;
        o.out_of_core = kOutOfCoreAuto;
        o.available_mem_mb = 1;
        o.max_mem_mb = 256;  // workspace given explicitly: a 1 MB budget is too small even out of core
        std::vector<double> x = b;
        SparseTriplets local = local_columns(a);
        MumpsSolver solver(MPI_COMM_WORLD, o);
        int const status = solver.solve(n, local, x);
        double err = 0.;
        for (int i = 0; i < n; ++i)
            err = std::max(err, std::fabs(x[i] - 1.));
        check(status == 0 && solver.stats().out_of_core && err < 1e-10,
              label + " auto, 1 MB per rank: out of core, status " + std::to_string(status) + ", max error " + sci(err));
    }
    {
        MumpsOptions o = options;
        o.out_of_core = kOutOfCoreNever;
        o.available_mem_mb = 1;
        std::vector<double> x = b;
        SparseTriplets local = local_columns(a);
        MumpsSolver solver(MPI_COMM_WORLD, o);
        int const status = solver.solve(n, local, x);
        check(status == kInsufficientMemory,
              label + " off, 1 MB per rank: kInsufficientMemory (status " + std::to_string(status) + ")");
    }
}

/// Block low-rank factors (MumpsOptions::blr_tolerance) are approximate: the solve must succeed with the
/// relaxed residual check of a preconditioner, the reported compressed size must not exceed the full-rank one,
/// and the retry ladder must fall back to full-rank factors.
void check_blr(MumpsOptions options, std::string const& label) {
    options.verbosity = 1;
    options.blr_tolerance = 1e-8;
    options.residual_tolerance = 1e-6;
    Matrix const a = make_matrix(6, 40, 3, 41, false);
    int const n = static_cast<int>(a.size());
    std::vector<double> x_true(n, 1.);
    std::vector<double> const b = multiply(a, x_true);
    std::vector<double> x = b;
    SparseTriplets local = local_columns(a);
    MumpsSolver solver(MPI_COMM_WORLD, options);
    int const status = solver.solve(n, local, x);
    check(status == 0, label + ": solve status " + std::to_string(status));
    MumpsStats const& s = solver.stats();
    check(s.relative_residual >= 0. && s.relative_residual <= 1e-6,
          label + ": relative residual " + sci(s.relative_residual));
    check(s.compressed_factor_entries > 0 && s.compressed_factor_entries <= s.factor_entries,
          label + ": " + std::to_string(s.compressed_factor_entries) + " of " +
              std::to_string(s.factor_entries) + " factor entries stored");
    std::vector<MumpsAttempt> const ladder = mumps_retry_ladder(options);
    bool full_rank_retries = ladder.size() > 1;
    for (std::size_t k = 1; k < ladder.size(); ++k)
        full_rank_retries = full_rank_retries && ladder[k].options.blr_tolerance == 0.;
    check(full_rank_retries, label + ": the retries use full-rank factors");
}

void run_singular(MumpsOptions options, std::string const& label) {
    options.verbosity = 0;
    MumpsSolver solver(MPI_COMM_WORLD, options);
    Matrix const a = make_matrix(4, 20, 1, 21, true);
    int const n = static_cast<int>(a.size());
    std::vector<double> x(n, 1.);
    SparseTriplets local = local_columns(a);
    int const status = solver.solve(n, local, x);
    check(status == -10 || status == -6 || status == kNoMatching,
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
    distributed.matching_entries = 0;  // MUMPS without any column permutation

    run_case(centralized, "centralized/METIS");
    run_case(distributed, "distributed/METIS");
    if (nproc_ > 1) {
        MumpsOptions parallel = distributed;
        parallel.parallel_analysis = true;
        parallel.parallel_analysis_min_n = 0;
        run_case(parallel, "distributed/ParMETIS");
        MumpsOptions scotch = parallel;
        scotch.parallel_ordering = 1;
        run_case(scotch, "distributed/PT-Scotch");
    }
    // Column permutation computed in the wrapper (pruned gather + maximum-product matching), as the Newton
    // solvers use it with distributed input; with 2 entries per column the retry with more entries may be needed.
    MumpsOptions matched = distributed;
    matched.matching_entries = 16;
    run_case(matched, "distributed+matching/METIS");
    MumpsOptions matched_small = distributed;
    matched_small.matching_entries = 2;
    run_case(matched_small, "distributed+matching (2 entries per column)/METIS");
    // Row candidates, gradual growth of the stuck columns and the cold restart of every round.
    MumpsOptions rows = matched_small;
    rows.matching_row_entries = 2;
    run_case(rows, "distributed+matching (2 per column, 2 per row)/METIS");
    MumpsOptions grown = matched_small;
    grown.matching_growth = 2;
    run_case(grown, "distributed+matching (2 per column, growth 2)/METIS");
    MumpsOptions cold = grown;
    cold.matching_warm_start = false;
    run_case(cold, "distributed+matching (2 per column, growth 2, no warm start)/METIS");
    MumpsOptions minimal = matched_small;
    minimal.matching_entries = 1;
    minimal.matching_row_entries = 1;
    minimal.matching_growth = 1;
    run_case(minimal, "distributed+matching (1 per column, 1 per row, growth 1)/METIS");
    // The fallback after INFOG(1) = -51 (graph too large for the 32-bit METIS): internal AMF ordering.
    run_case(with_internal_ordering(matched), "distributed+matching/AMF");
    run_case(with_internal_ordering(centralized), "centralized/AMF");
    // ParMETIS on this small matrix: only on 2 ranks; on 4 METIS gets an empty local subgraph and divides by
    // zero (see MumpsOptions::parallel_analysis_min_n).
    if (nproc_ == 2) {
        MumpsOptions matched_parallel = matched;
        matched_parallel.parallel_analysis = true;
        matched_parallel.parallel_analysis_min_n = 0;
        run_case(matched_parallel, "distributed+matching/ParMETIS");
    }
    // Out-of-core factors (exact): files in $TMPDIR (or /tmp), removed by MUMPS at JOB = -2.
    MumpsOptions out_of_core = centralized;
    char const* tmpdir = std::getenv("TMPDIR");
    out_of_core.ooc_dir = tmpdir != nullptr && tmpdir[0] != 0 ? tmpdir : "/tmp";
    out_of_core.out_of_core = kOutOfCoreAlways;
    run_case(out_of_core, "centralized/METIS out-of-core");
    MumpsOptions out_of_core_distributed = distributed;
    out_of_core_distributed.ooc_dir = out_of_core.ooc_dir;
    out_of_core_distributed.out_of_core = kOutOfCoreAlways;
    run_case(out_of_core_distributed, "distributed/METIS out-of-core");
    check_out_of_core_decision(centralized, "centralized");
    check_out_of_core_decision(distributed, "distributed");
    check_blr(centralized, "centralized BLR 1e-8");
    check_blr(distributed, "distributed BLR 1e-8");
    check_distribution_independence(centralized,
                                    "centralized distribution independence");
    check_residual_rejection(centralized, "centralized residual check");
    check_retry_ladder(centralized, "retry ladder");
    check_residual_rejection(distributed, "distributed residual check");
    run_singular(centralized, "centralized singular");
    run_singular(distributed, "distributed singular");
    run_singular(matched, "distributed+matching singular");

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
