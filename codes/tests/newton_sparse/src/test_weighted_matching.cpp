/*
 * T7: maximum-product matching (include/Linear_solvers/weighted_matching.hpp).
 *  - random sparse matrices (n <= 8): the matching is perfect and its product of |a_i,p(i)| / colmax equals
 *    the brute-force optimum over all permutations
 *  - a permuted diagonal-dominant matrix: the matching recovers the permutation
 *  - a structurally singular matrix (an empty row): empty result
 *  - a larger random matrix (n = 2000, a forced diagonal plus 10 random entries per column): perfect matching
 * Serial; returns 0 if all checks pass.
 */
#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "Linear_solvers/weighted_matching.hpp"

using namespace Kadath::linear_solvers;

namespace {

int failures = 0;

void check(bool ok, std::string const& what) {
    if (!ok)
        ++failures;
    std::cout << (ok ? "ok:   " : "FAIL: ") << what << std::endl;
}

struct Csc {
    int n = 0;
    std::vector<std::int64_t> ptr;
    std::vector<int> row;
    std::vector<double> val;
};

Csc from_dense(std::vector<std::vector<double>> const& a) {
    Csc m;
    m.n = static_cast<int>(a.size());
    m.ptr.assign(1, 0);
    for (int j = 0; j < m.n; ++j) {
        for (int i = 0; i < m.n; ++i)
            if (a[i][j] != 0.) {
                m.row.push_back(i);
                m.val.push_back(a[i][j]);
            }
        m.ptr.push_back(static_cast<std::int64_t>(m.row.size()));
    }
    return m;
}

/// sum over columns of log(|a_{r(j), j}| / colmax_j); -inf if some matched entry is zero
double log_product(std::vector<std::vector<double>> const& a, std::vector<int> const& row_of_col) {
    int const n = static_cast<int>(a.size());
    double s = 0.;
    for (int j = 0; j < n; ++j) {
        double colmax = 0.;
        for (int i = 0; i < n; ++i)
            colmax = std::max(colmax, std::fabs(a[i][j]));
        double const x = std::fabs(a[row_of_col[j]][j]);
        if (x == 0.)
            return -INFINITY;
        s += std::log(x / colmax);
    }
    return s;
}

bool is_permutation(std::vector<int> const& p, int n) {
    if (static_cast<int>(p.size()) != n)
        return false;
    std::vector<char> seen(n, 0);
    for (int x : p) {
        if (x < 0 || x >= n || seen[x])
            return false;
        seen[x] = 1;
    }
    return true;
}

}  // namespace

int main() {
    std::mt19937 gen(12345);
    std::uniform_real_distribution<double> mag(-6., 2.);
    std::uniform_real_distribution<double> unit(0., 1.);

    // Random small matrices against brute force.
    int optimal = 0, trials = 0, singular_skipped = 0;
    for (int trial = 0; trial < 300; ++trial) {
        int const n = 2 + trial % 7;
        std::vector<std::vector<double>> a(n, std::vector<double>(n, 0.));
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
                if (unit(gen) < 0.5)
                    a[i][j] = (unit(gen) < 0.5 ? -1. : 1.) * std::pow(10., mag(gen));
        std::vector<int> perm(n);
        std::iota(perm.begin(), perm.end(), 0);
        double best = -INFINITY;
        do {
            best = std::max(best, log_product(a, perm));  // perm = row of column j
        } while (std::next_permutation(perm.begin(), perm.end()));
        Csc m = from_dense(a);
        std::vector<int> r = max_product_matching(n, m.ptr, m.row, m.val);
        if (best == -INFINITY) {
            ++singular_skipped;
            if (!r.empty())
                check(false, "structurally singular random matrix gives an empty result");
            continue;
        }
        ++trials;
        if (is_permutation(r, n) && std::fabs(log_product(a, r) - best) <= 1e-10 * (1. + std::fabs(best)))
            ++optimal;
    }
    check(optimal == trials, "random n <= 8: optimal in " + std::to_string(optimal) + " of " +
                                 std::to_string(trials) + " nonsingular cases (" +
                                 std::to_string(singular_skipped) + " singular)");

    // Permuted diagonally dominant matrix.
    {
        int const n = 50;
        std::vector<int> p(n);
        std::iota(p.begin(), p.end(), 0);
        std::shuffle(p.begin(), p.end(), gen);
        std::vector<std::vector<double>> a(n, std::vector<double>(n, 0.));
        for (int j = 0; j < n; ++j) {
            a[p[j]][j] = 100.;
            for (int k = 0; k < 4; ++k)
                a[gen() % n][j] += 1e-2 * unit(gen);
        }
        Csc m = from_dense(a);
        std::vector<int> r = max_product_matching(n, m.ptr, m.row, m.val);
        check(r == p, "permuted diagonally dominant matrix: permutation recovered");
    }

    // Structurally singular: row 2 empty.
    {
        std::vector<std::vector<double>> a = {{1, 2, 0}, {3, 0, 4}, {0, 0, 0}};
        Csc m = from_dense(a);
        check(max_product_matching(3, m.ptr, m.row, m.val).empty(), "empty row: structurally singular");
        std::vector<int> stuck;
        check(max_product_matching(3, m.ptr, m.row, m.val, &stuck).empty() && !stuck.empty(),
              "empty row: failed searches report the columns they reached");
    }

    // Pattern missing entries: a sampled pattern of a nonsingular matrix without a perfect matching; adding
    // the full columns reported as stuck completes it.
    {
        std::vector<std::vector<double>> full = {{5, 4, 0, 1}, {3, 0, 0, 0}, {0, 2, 6, 0}, {0, 0, 1, 7}};
        std::vector<std::vector<double>> sampled = {{5, 4, 0, 0}, {0, 0, 0, 0}, {0, 2, 6, 0}, {0, 0, 1, 7}};
        sampled[1][0] = 0.;  // row 1 only has a (small) entry in column 0, dropped by the sampling
        Csc m = from_dense(sampled);
        std::vector<int> stuck;
        bool const failed = max_product_matching(4, m.ptr, m.row, m.val, &stuck).empty();
        for (int j : stuck)
            for (int i = 0; i < 4; ++i)
                sampled[i][j] = full[i][j];
        Csc m2 = from_dense(sampled);
        std::vector<int> r = max_product_matching(4, m2.ptr, m2.row, m2.val);
        check(failed && is_permutation(r, 4) && r[0] == 1, "stuck columns expanded: perfect matching found");
    }

    // Larger random matrix in CSC directly.
    {
        int const n = 2000;
        std::vector<int> p(n);
        std::iota(p.begin(), p.end(), 0);
        std::shuffle(p.begin(), p.end(), gen);
        std::vector<std::int64_t> ptr(1, 0);
        std::vector<int> row;
        std::vector<double> val;
        for (int j = 0; j < n; ++j) {
            std::vector<std::pair<int, double>> col = {{p[j], std::pow(10., mag(gen))}};
            for (int k = 0; k < 10; ++k)
                col.push_back({static_cast<int>(gen() % n), std::pow(10., mag(gen))});
            std::sort(col.begin(), col.end());
            col.erase(std::unique(col.begin(), col.end(),
                                  [](auto const& x, auto const& y) { return x.first == y.first; }),
                      col.end());
            for (auto const& [i, x] : col) {
                row.push_back(i);
                val.push_back(x);
            }
            ptr.push_back(static_cast<std::int64_t>(row.size()));
        }
        std::vector<int> r = max_product_matching(n, ptr, row, val);
        bool entries_exist = is_permutation(r, n);
        for (int j = 0; entries_exist && j < n; ++j)
            entries_exist = std::find(row.begin() + ptr[j], row.begin() + ptr[j + 1], r[j]) != row.begin() + ptr[j + 1];
        check(entries_exist, "n = 2000 random: perfect matching on existing entries");
    }

    std::cout << "test_weighted_matching: " << (failures == 0 ? "PASS" : "FAIL") << " (" << failures
              << " failures)" << std::endl;
    return failures == 0 ? 0 : 1;
}
