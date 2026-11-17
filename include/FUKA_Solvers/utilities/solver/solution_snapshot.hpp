/**
 * @file solution_snapshot.hpp
 * @brief Keep a solver's solution and grid in memory so that it can be
 * restored, e.g. when a refined solution turns out worse.
 *
 * @copyright Copyright (c) 2026, GNU General Public Licensev3
 * This file is part of the KADATH library and published under
 * https://arxiv.org/abs/2103.09911
 */
#pragma once
#include <memory>
#include <optional>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <utility>

/** \addtogroup Solver_utils
 * \ingroup FUKA
 * @{*/
namespace Kadath::FUKA_Solvers {

namespace snapshot_detail {
/// Storage of one snapshot member, chosen from the member's ownership.
template <class member_ptr_t>
struct member_storage;

/// Owned member: moved into the snapshot and back.
template <class class_t, class T>
struct member_storage<std::unique_ptr<T> class_t::*> {
    using type = std::unique_ptr<T>;

    template <class solver_t>
    static void take(type& store,
                     solver_t& s,
                     std::unique_ptr<T> class_t::* m) {
        store = std::move(s.*m);
    }

    template <class solver_t>
    static void restore(type& store,
                        solver_t& s,
                        std::unique_ptr<T> class_t::* m) {
        s.*m = std::move(store);
    }
};

/// Observed member (e.g. a config owned by the driver): its value is copied,
/// and assigned back so the observed object keeps its identity.
template <class class_t, class T>
struct member_storage<T * class_t::*> {
    using type = std::optional<T>;

    template <class solver_t>
    static void take(type& store, solver_t& s, T* class_t::* m) {
        store.emplace(*(s.*m));
    }

    template <class solver_t>
    static void restore(type& store, solver_t& s, T* class_t::* m) {
        *(s.*m) = std::move(*store);
        store.reset();
    }
};

template <class member_ptr_t>
using storage_t = typename member_storage<member_ptr_t>::type;

template <class tuple_t>
struct storage_tuple;

template <class... member_ptr_ts>
struct storage_tuple<std::tuple<member_ptr_ts...>> {
    using type = std::tuple<storage_t<member_ptr_ts>...>;
};
}  // namespace snapshot_detail

/**
 * @brief Snapshot of the members a solver lists in `snapshot_members()`.
 *
 * `solver_t::snapshot_members()` returns a tuple of pointers to data members:
 * `std::unique_ptr` members are moved (space, solution fields), raw-pointer
 * observers are copied by value (config). `take` requires a released system
 * (i.e. after do_newton) so that no Jacobian or factorization is kept;
 * `restore` moves everything back and rebuilds the solver's support
 * containers.
 *
 * @tparam solver_t solver providing snapshot_members() and
 * rebuild_support_containers()
 */
template <class solver_t>
class Solution_snapshot {
    using members_t = decltype(solver_t::snapshot_members());
    using storage_t = typename snapshot_detail::storage_tuple<members_t>::type;
    static constexpr std::size_t nmembers = std::tuple_size_v<members_t>;

    storage_t members;
    bool taken = false;

    template <std::size_t... I>
    void take_impl(solver_t& s, std::index_sequence<I...>) {
        constexpr auto ptrs = solver_t::snapshot_members();
        (snapshot_detail::member_storage<
             std::tuple_element_t<I, members_t>>::take(std::get<I>(members),
                                                       s,
                                                       std::get<I>(ptrs)),
         ...);
    }

    template <std::size_t... I>
    void restore_impl(solver_t& s, std::index_sequence<I...>) {
        constexpr auto ptrs = solver_t::snapshot_members();
        (snapshot_detail::member_storage<
             std::tuple_element_t<I, members_t>>::restore(std::get<I>(members),
                                                          s,
                                                          std::get<I>(ptrs)),
         ...);
    }

   public:
    /// Move the solver's solution and grid into the snapshot.
    /// @throws std::logic_error if the solver still holds a system or a
    /// snapshot was already taken
    void take(solver_t& s) {
        if (s.get_syst())
            throw std::logic_error(
                "Solution_snapshot::take: release the system first");
        if (taken)
            throw std::logic_error("Solution_snapshot::take: already taken");
        take_impl(s, std::make_index_sequence<nmembers>{});
        taken = true;
    }

    /// Move the saved solution and grid back and rebuild the support
    /// containers; the snapshot is empty afterwards.
    void restore(solver_t& s) {
        if (!taken)
            throw std::logic_error("Solution_snapshot::restore: empty");
        s.get_syst().reset(nullptr);
        restore_impl(s, std::make_index_sequence<nmembers>{});
        taken = false;
        s.rebuild_support_containers();
    }

    /// Drop the saved solution (e.g. once a refinement is accepted).
    void clear() {
        members = storage_t{};
        taken = false;
    }

    [[nodiscard]] bool empty() const { return !taken; }
};

/** @}*/
}  // namespace Kadath::FUKA_Solvers
