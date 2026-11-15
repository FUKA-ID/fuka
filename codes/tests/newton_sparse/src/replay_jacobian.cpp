/*
 * Replays a Newton Jacobian dumped with KADATH_NEWTON_DUMP_PATTERN=<prefix> through MumpsSolver.
 *
 * Every rank reads the dump files r, r + nproc, ... (any rank count works), the right-hand side is b = A x_true
 * with x_true_i = 1 + 0.5 sin(i), and the solve uses the MUMPS options from the KADATH_MUMPS_* environment
 * variables. Reports the status, pivoting statistics, relative residual and relative error, so factorization
 * failures and accuracy can be studied offline on real matrices.
 *
 * usage: mpirun -np N replay_jacobian <prefix>.step<k>
 * Returns 0 if the solve succeeds with relative residual < 1e-12, 1 otherwise.
 */
#include <mpi.h>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "Linear_solvers/sparse_triplets.hpp"

#ifdef KADATH_USE_MUMPS
#include "Linear_solvers/mumps_solver.hpp"

using namespace Kadath::linear_solvers;

namespace {

/// Reads one dump file; returns false if it does not exist.
bool read_dump(std::string const& name,
               int& n,
               int& nproc_dump,
               SparseTriplets& t) {
    std::ifstream in(name, std::ios::binary);
    if (!in)
        return false;
    std::int64_t count = 0;
    in.read(reinterpret_cast<char*>(&n), sizeof(n));
    in.read(reinterpret_cast<char*>(&nproc_dump), sizeof(nproc_dump));
    in.read(reinterpret_cast<char*>(&count), sizeof(count));
    std::size_t const old = t.values.size();
    t.rows.resize(old + count);
    t.cols.resize(old + count);
    t.values.resize(old + count);
    in.read(reinterpret_cast<char*>(t.rows.data() + old), count * sizeof(int));
    in.read(reinterpret_cast<char*>(t.cols.data() + old), count * sizeof(int));
    in.read(reinterpret_cast<char*>(t.values.data() + old),
            count * sizeof(double));
    if (!in) {
        std::cerr << "truncated " << name << std::endl;
        std::exit(1);
    }
    return true;
}

/// Global y = A x from every rank's local entries.
std::vector<double> multiply(SparseTriplets const& t,
                             std::vector<double> const& x) {
    std::vector<double> local(x.size(), 0.), y(x.size(), 0.);
    for (std::int64_t k = 0; k < t.size(); ++k)
        local[t.rows[k] - 1] += t.values[k] * x[t.cols[k] - 1];
    MPI_Allreduce(local.data(),
                  y.data(),
                  static_cast<int>(x.size()),
                  MPI_DOUBLE,
                  MPI_SUM,
                  MPI_COMM_WORLD);
    return y;
}

double max_abs(std::vector<double> const& v) {
    double m = 0.;
    for (double x : v)
        m = std::max(m, std::fabs(x));
    return m;
}

}  // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, nproc;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);
    if (argc != 2) {
        if (rank == 0)
            std::cerr << "usage: replay_jacobian <prefix>.step<k>" << std::endl;
        MPI_Finalize();
        return 2;
    }
    std::string const prefix = argv[1];
    bool ok_repeat = true;

    int n = 0, nproc_dump = 0;
    SparseTriplets local;
    if (!read_dump(prefix + ".rank0.bin", n, nproc_dump, local)) {
        if (rank == 0)
            std::cerr << "missing " << prefix << ".rank0.bin" << std::endl;
        MPI_Finalize();
        return 1;
    }
    if (rank != 0)
        local = SparseTriplets{};
    for (int r = rank; r < nproc_dump; r += nproc)
        if (r != 0 && !read_dump(prefix + ".rank" + std::to_string(r) + ".bin",
                                 n,
                                 nproc_dump,
                                 local)) {
            std::cerr << "missing " << prefix << ".rank" << r << ".bin"
                      << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

    std::vector<double> x_true(n);
    for (int i = 0; i < n; ++i)
        x_true[i] = 1. + 0.5 * std::sin(1. * i);
    SparseTriplets const original =
        local;  // MumpsSolver keeps its input, but the check must not depend on it
    std::vector<double> const b = multiply(original, x_true);
    std::vector<double> x = b;

    MumpsSolver solver(MPI_COMM_WORLD);
    double const start = MPI_Wtime();
    int const status = solver.solve(n, local, x);
    double const elapsed = MPI_Wtime() - start;
    MumpsStats const& s = solver.stats();
    if (rank == 0)
        std::cout << "replay times: gather " << s.time_gather << ", analysis "
                  << s.time_analysis << ", factor+solve "
                  << s.time_factor_solve << " s" << std::endl;

    // KADATH_REPLAY_REPEAT=k: k more solves on the same instance with perturbed values (a Newton-like
    // sequence), to time and check the kept analysis (KADATH_MUMPS_REUSE_ANALYSIS).
    char const* repeat_env = std::getenv("KADATH_REPLAY_REPEAT");
    int const repeat = repeat_env ? std::atoi(repeat_env) : 0;
    for (int r = 1; r <= repeat && status == 0; ++r) {
        SparseTriplets perturbed = original;
        for (std::int64_t k = 0; k < perturbed.size(); ++k)
            perturbed.values[k] *= 1. + 1e-3 * std::sin(1. * r * (k + 1));
        if (r % 2 == 0 && perturbed.size() > 0) {  // drop one local entry: the kept pattern gets a zero
            perturbed.rows.pop_back();
            perturbed.cols.pop_back();
            perturbed.values.pop_back();
        }
        SparseTriplets const check_matrix = perturbed;
        std::vector<double> const br = multiply(check_matrix, x_true);
        std::vector<double> xr = br;
        double const t0 = MPI_Wtime();
        int const st = solver.solve(n, perturbed, xr);
        double const dt = MPI_Wtime() - t0;
        std::vector<double> res = multiply(check_matrix, xr);
        for (int i = 0; i < n; ++i)
            res[i] -= br[i];
        double const rel = max_abs(res) / max_abs(br);
        if (rank == 0)
            std::cout << "replay repeat " << r << ": status " << st
                      << ", analysis "
                      << (solver.stats().analysis_reused
                              ? std::string("kept")
                              : std::to_string(solver.stats().time_analysis))
                      << ", zeros " << solver.stats().pattern_zeros
                      << ", factor+solve " << solver.stats().time_factor_solve
                      << " s, total " << dt << " s, relative residual " << rel
                      << (st == 0 && rel < 1e-12 ? "  [ok]" : "  [BAD]")
                      << std::endl;
        if (st != 0 || !(rel < 1e-12))
            ok_repeat = false;
    }

    int ok = 0;
    if (rank == 0)
        std::cout << "replay " << prefix << ": n = " << n << ", nnz = " << s.nnz
                  << ", ranks = " << nproc << " (dumped with " << nproc_dump
                  << "), status = " << status << " ("
                  << (status == 0 ? "ok" : mumps_error_message(status))
                  << ")\n  pivots delayed = " << s.delayed_pivots
                  << ", off-diagonal = " << s.off_diagonal_pivots
                  << ", factor entries estimated/actual = "
                  << s.estimated_factor_entries << " / " << s.factor_entries
                  << ", wrapper residual = " << s.relative_residual
                  << ", memory max/total = " << s.peak_mem_mb_max << " / "
                  << s.peak_mem_mb_total
                  << " MB, ICNTL(14) used = " << s.mem_relax_used
                  << ", time = " << elapsed << " s" << std::endl;
    if (status == 0) {
        std::vector<double> r = multiply(original, x);
        std::vector<double> err(n);
        for (int i = 0; i < n; ++i) {
            r[i] -= b[i];
            err[i] = x[i] - x_true[i];
        }
        double const rel_res = max_abs(r) / max_abs(b);
        double const rel_err = max_abs(err) / max_abs(x_true);
        ok = rel_res < 1e-12;
        if (rank == 0)
            std::cout << "  relative residual = " << rel_res
                      << ", relative error = " << rel_err
                      << (ok ? "  [ok]" : "  [INACCURATE]") << std::endl;
    }
    MPI_Finalize();
    return ok && ok_repeat ? 0 : 1;
}

#else
int main() {
    std::cout << "replay_jacobian: SKIP (built without KADATH_USE_MUMPS)"
              << std::endl;
    return 77;
}
#endif
