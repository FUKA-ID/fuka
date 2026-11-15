/*
 * T1: sparse Jacobian bookkeeping used by the MUMPS Newton path (no MUMPS needed).
 *  - newton_block_size reproduces the ScaLAPACK-path block size rule (and never returns 0)
 *  - block_cyclic_columns matches ScaLAPACK numroc/indxl2g on a 1 x nproc grid and partitions 0..n-1,
 *    for the ScaLAPACK-path block size and for the round-robin (1) and odd (3) block sizes
 *  - append_column drops exact zeros only, uses 1-based indices and round-trips to the dense column
 *  - drop_negligible removes only entries small relative to both their row and column maximum
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

void test_drop_negligible() {
    // 3 x 3, columns 0..2. Row 2 is tiny overall (scale 1e-20): its entries are small in their columns
    // but not in their row, so they must stay. a(0,1) = 1e-15 is negligible in row 0 and column 1.
    SparseTriplets t;
    auto add = [&t](int i, int j, double v) {
        t.rows.push_back(i + 1);
        t.cols.push_back(j + 1);
        t.values.push_back(v);
    };
    add(0, 0, 1.);
    add(0, 1, -1e-15);
    add(1, 1, 2.);
    add(1, 2, 3.);
    add(2, 0, 1e-20);
    add(2, 2, -2e-20);
    std::vector<double> row_max, col_max;
    local_row_column_max(t, 3, row_max, col_max);
    check(row_max[0] == 1. && row_max[2] == 2e-20 && col_max[2] == 3.,
          "row and column maxima");
    std::int64_t const removed = drop_negligible(t, row_max, col_max, 1e-13);
    check(removed == 1 && t.size() == 5, "exactly one negligible entry removed");
    bool kept_small_row = false, removed_target = true;
    for (std::int64_t k = 0; k < t.size(); ++k) {
        if (t.rows[k] == 3 && t.cols[k] == 1 && t.values[k] == 1e-20)
            kept_small_row = true;
        if (t.rows[k] == 1 && t.cols[k] == 2)
            removed_target = false;
    }
    check(kept_small_row, "entries of a small-scale row are kept");
    check(removed_target, "entry negligible in row and column removed");
    check(drop_negligible(t, row_max, col_max, 0.) == 0, "tolerance 0 keeps everything");
}

void test_merge_with_pattern() {
    // pattern (col, row): (1,1) (1,3) (2,2) (3,1)
    std::vector<int> const prow = {1, 3, 2, 1}, pcol = {1, 1, 2, 3};
    SparseTriplets t;
    t.rows = {1, 2};
    t.cols = {1, 2};
    t.values = {5., 7.};
    check(sorted_by_column(t), "append order is sorted");
    std::int64_t zeros = -1;
    SparseTriplets m;
    check(merge_with_pattern(t, prow, pcol, m, zeros) == 0 && zeros == 2,
          "subset of the pattern: nothing outside, two zeros added");
    check(m.rows == prow && m.cols == pcol, "union is the pattern");
    check(m.values[0] == 5. && m.values[1] == 0. && m.values[2] == 7. &&
              m.values[3] == 0.,
          "values kept, missing positions zero");
    check(t.size() == 2, "input unchanged");

    SparseTriplets u;
    u.rows = {2, 3, 2};
    u.cols = {1, 1, 3};
    u.values = {1., 2., 3.};
    check(merge_with_pattern(u, prow, pcol, m, zeros) == 2 && zeros == 3,
          "two new entries, three zeros");
    check(m.size() == 6 && sorted_by_column(m), "union sorted, size 6");

    SparseTriplets w;
    w.rows = {2, 1};
    w.cols = {1, 1};
    w.values = {1., 1.};
    check(!sorted_by_column(w), "unsorted rows detected");
}

}  // namespace

int main() {
    test_block_size();
    test_column_distribution();
    test_append_column();
    test_drop_negligible();
    test_merge_with_pattern();
    std::cout << (failures == 0 ? "test_sparse_triplets: PASS"
                                : "test_sparse_triplets: FAIL")
              << " (" << failures << " failures)" << std::endl;
    return failures == 0 ? 0 : 1;
}
