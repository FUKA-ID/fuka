/*
 * T1: sparse Jacobian bookkeeping used by the MUMPS Newton path (no MUMPS needed).
 *  - newton_block_size reproduces the ScaLAPACK-path block size rule (and never returns 0)
 *  - block_cyclic_columns matches ScaLAPACK numroc/indxl2g on a 1 x nproc grid and partitions 0..n-1,
 *    for the ScaLAPACK-path block size and for the round-robin (1) and odd (3) block sizes
 *  - append_column drops exact zeros only, uses 1-based indices and round-trips to the dense column
 *
 * Serial; returns 0 if all checks pass.
 */
#include <cmath>
#include <iostream>
#include <limits>
#include <random>
#include <set>
#include <vector>

#include "Linear_solvers/sparse_triplets.hpp"
#include "array.hpp"

using namespace Kadath;
using namespace Kadath::linear_solvers;

extern "C" {
int numroc_(int*, int*, int*, int*, int*);
int indxl2g_(int*, int*, int*, int*, int*);
}

namespace {

int failures = 0;

void check(bool ok, std::string const& what) {
    if (!ok) {
        ++failures;
        std::cout << "FAIL: " << what << std::endl;
    }
}

/// Block size exactly as written in the original ScaLAPACK do_newton (may reach 0 for nproc > m).
int legacy_block_size(int m, int nproc) {
    int bsize = 256;
    while (nproc * bsize > m)
        bsize = bsize / 2;
    return bsize;
}

std::vector<int> scalapack_columns(int n, int bsize, int rank, int nproc) {
    int zero = 0;
    int ncol = numroc_(&n, &bsize, &rank, &zero, &nproc);
    std::vector<int> cols;
    for (int j = 1; j <= ncol; ++j)
        cols.push_back(indxl2g_(&j, &bsize, &rank, &zero, &nproc) - 1);
    return cols;
}

void test_block_size() {
    for (int m : {1, 2, 7, 100, 255, 256, 1000, 4096, 123457})
        for (int nproc : {1, 2, 3, 4, 8, 16, 40, 128, 640}) {
            int const b = newton_block_size(m, nproc);
            check(b >= 1,
                  "block size >= 1 for m=" + std::to_string(m) +
                      " nproc=" + std::to_string(nproc));
            if (legacy_block_size(m, nproc) >= 1)
                check(b == legacy_block_size(m, nproc),
                      "block size matches legacy rule for m=" +
                          std::to_string(m) +
                          " nproc=" + std::to_string(nproc));
        }
}

void test_column_distribution() {
    for (int n : {1, 5, 64, 257, 1000, 5003})
        for (int nproc : {1, 2, 3, 4, 7, 16})
            for (int bsize : {newton_block_size(n, nproc), 1, 3}) {
                std::vector<int> owner(n, -1);
                for (int rank = 0; rank < nproc; ++rank) {
                    auto const cols =
                        block_cyclic_columns(n, bsize, rank, nproc);
                    check(cols == scalapack_columns(n, bsize, rank, nproc),
                          "columns match ScaLAPACK for n=" + std::to_string(n) +
                              " nproc=" + std::to_string(nproc) +
                              " rank=" + std::to_string(rank));
                    for (int c : cols) {
                        check(c >= 0 && c < n, "column in range");
                        if (c >= 0 && c < n) {
                            check(owner[c] == -1, "column owned once");
                            owner[c] = rank;
                        }
                    }
                }
                for (int c = 0; c < n; ++c)
                    check(owner[c] != -1,
                          "column " + std::to_string(c) +
                              " owned (n=" + std::to_string(n) + ")");
            }
}

void test_append_column() {
    std::mt19937 gen(1234);
    std::uniform_real_distribution<double> value(-1., 1.);
    std::bernoulli_distribution keep(0.2);

    int const m = 333;
    int const ncols = 17;
    std::vector<std::vector<double>> dense(ncols, std::vector<double>(m, 0.));
    SparseTriplets triplets;
    std::int64_t expected_nnz = 0;

    for (int j = 0; j < ncols; ++j) {
        Array<double> column(m);
        column = 0.;
        for (int i = 0; i < m; ++i)
            if (keep(gen))
                column.set(i) = value(gen);
        column.set(0) = -0.;  // signed zero is a zero
        column.set(1) = std::numeric_limits<
            double>::denorm_min();  // tiny but nonzero: kept
        column.set(m - 1) = 1e-300;
        for (int i = 0; i < m; ++i) {
            dense[j][i] = column(i);
            expected_nnz += (column(i) != 0.);
        }
        int const global_col = 3 * j + 1;
        append_column(column, global_col, triplets);
    }

    check(triplets.size() == expected_nnz, "nnz equals number of nonzeros");
    check(triplets.rows.size() == triplets.values.size() &&
              triplets.cols.size() == triplets.values.size(),
          "triplet arrays have equal length");

    std::vector<std::vector<double>> rebuilt(ncols, std::vector<double>(m, 0.));
    std::set<std::pair<int, int>> seen;
    for (std::int64_t k = 0; k < triplets.size(); ++k) {
        int const i = triplets.rows[k] - 1;
        int const gc = triplets.cols[k] - 1;
        check(i >= 0 && i < m, "row index 1-based and in range");
        check(gc % 3 == 1, "column index 1-based global column");
        check(triplets.values[k] != 0., "no stored zero");
        check(seen.insert({i, gc}).second, "no duplicate entry");
        if (i >= 0 && i < m && gc % 3 == 1)
            rebuilt[gc / 3][i] = triplets.values[k];
    }
    for (int j = 0; j < ncols; ++j)
        for (int i = 0; i < m; ++i)
            check(rebuilt[j][i] == dense[j][i] ||
                      (rebuilt[j][i] == 0. && dense[j][i] == 0.),
                  "round trip column " + std::to_string(j) + " row " +
                      std::to_string(i));
}

}  // namespace

int main() {
    test_block_size();
    test_column_distribution();
    test_append_column();
    std::cout << (failures == 0 ? "test_sparse_triplets: PASS"
                                : "test_sparse_triplets: FAIL")
              << " (" << failures << " failures)" << std::endl;
    return failures == 0 ? 0 : 1;
}
