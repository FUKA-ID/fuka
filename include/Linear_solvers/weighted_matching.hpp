#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <queue>
#include <utility>
#include <vector>

namespace Kadath {
namespace linear_solvers {

/// Maximum-product bipartite matching of a sparse square matrix (the criterion of MUMPS ICNTL(6)=5 / MC64 job
/// 5): a column permutation that puts large entries on the diagonal, so the matrix can be factored with little
/// pivoting.
///
/// Input in compressed sparse column form (0-based): column j holds rows row[ptr[j]..ptr[j+1]) with values
/// val[...]. The cost of entry (i, j) is log(max_k |a_kj|) - log|a_ij| >= 0, and the matching minimizes the
/// sum of the costs, i.e. maximizes the product of the |diagonal| relative to the column maxima. Solved by
/// successive shortest augmenting paths (Dijkstra with row and column potentials), after a greedy start
/// that matches every column to its largest entry when that row is free.
///
/// Returns row_of_col (row matched to each column), or an empty vector if the matrix (as given) has no
/// perfect matching, i.e. is structurally singular. In that case, if @p stuck is given, it receives the
/// columns reached by the failed augmenting-path searches (sorted, unique): their entries outside the given
/// pattern are the only ones that can complete the matching.
inline std::vector<int> max_product_matching(
    int n,
    std::vector<std::int64_t> const& ptr,
    std::vector<int> const& row,
    std::vector<double> const& val,
    std::vector<int>* stuck = nullptr) {
    constexpr double inf = std::numeric_limits<double>::infinity();
    std::vector<double> cost(row.size(), inf);
    for (int j = 0; j < n; ++j) {
        double colmax = 0.;
        for (std::int64_t k = ptr[j]; k < ptr[j + 1]; ++k)
            colmax = std::max(colmax, std::fabs(val[k]));
        if (colmax == 0.)
            continue;
        double const lmax = std::log(colmax);
        for (std::int64_t k = ptr[j]; k < ptr[j + 1]; ++k)
            if (val[k] != 0.)
                cost[k] = lmax - std::log(std::fabs(val[k]));
    }

    std::vector<int> row_of_col(n, -1), col_of_row(n, -1);
    std::vector<double> u(n, 0.),
        v(n, 0.);  // row and column potentials; reduced cost c - u - v >= 0

    // Greedy start on zero reduced cost: the largest entry of each column (cost 0), if its row is free.
    for (int j = 0; j < n; ++j)
        for (std::int64_t k = ptr[j]; k < ptr[j + 1]; ++k)
            if (cost[k] == 0. && col_of_row[row[k]] < 0) {
                row_of_col[j] = row[k];
                col_of_row[row[k]] = j;
                break;
            }

    std::vector<double> dist(n, inf);
    std::vector<int> pred(n, -1);  // column from which a row was reached
    std::vector<char> done(n, 0);  // row distance final
    // Rows reached by a failed search: they and their matched columns cannot reach a free row (otherwise
    // the search would have found one), and later augmenting paths never enter that closed set, so later
    // searches skip them. This keeps the cost of many failed searches linear instead of quadratic.
    std::vector<char> dead(n, 0);
    std::vector<int> touched, finalized;
    using item = std::pair<double, int>;
    for (int s = 0; s < n; ++s) {
        if (row_of_col[s] >= 0)
            continue;
        std::priority_queue<item, std::vector<item>, std::greater<item>> heap;
        touched.clear();
        finalized.clear();
        auto relax =
            [&](int j,
                double dj) {  // rows of column j at distance dj + reduced cost
                for (std::int64_t k = ptr[j]; k < ptr[j + 1]; ++k) {
                    if (cost[k] == inf)
                        continue;
                    int const i = row[k];
                    if (done[i] || dead[i])
                        continue;
                    double const d = dj + cost[k] - u[i] - v[j];
                    if (d < dist[i]) {
                        if (dist[i] == inf)
                            touched.push_back(i);
                        dist[i] = d;
                        pred[i] = j;
                        heap.push({d, i});
                    }
                }
            };
        relax(s, 0.);
        int free_row = -1;
        double D = inf;
        while (!heap.empty()) {
            auto [d, i] = heap.top();
            heap.pop();
            if (done[i] || d > dist[i])
                continue;
            if (col_of_row[i] < 0) {
                free_row = i;
                D = d;
                break;
            }
            done[i] = 1;
            finalized.push_back(i);
            relax(col_of_row[i], d);  // matched edge has reduced cost 0
        }
        if (free_row <
            0) {  // structurally singular (on the given pattern): leave s unmatched, go on
            if (!stuck)
                return {};
            stuck->push_back(s);
            for (int i : finalized) {
                stuck->push_back(col_of_row[i]);
                dead[i] = 1;
            }
            for (int i : touched) {
                dist[i] = inf;
                done[i] = 0;
                pred[i] = -1;
            }
            continue;
        }
        // Potentials: keep reduced costs >= 0 and make the tree and the path tight.
        v[s] += D;
        for (int i : finalized) {
            double const delta = D - dist[i];
            u[i] -= delta;
            v[col_of_row[i]] += delta;
        }
        // Augment along the path that ends in free_row.
        for (int i = free_row;;) {
            int const j = pred[i];
            int const previous = row_of_col[j];
            row_of_col[j] = i;
            col_of_row[i] = j;
            if (j == s)
                break;
            i = previous;
        }
        for (int i : touched) {
            dist[i] = inf;
            done[i] = 0;
            pred[i] = -1;
        }
    }
    if (stuck && !stuck->empty()) {
        std::sort(stuck->begin(), stuck->end());
        stuck->erase(std::unique(stuck->begin(), stuck->end()), stuck->end());
        return {};
    }
    return row_of_col;
}

}  // namespace linear_solvers
}  // namespace Kadath
