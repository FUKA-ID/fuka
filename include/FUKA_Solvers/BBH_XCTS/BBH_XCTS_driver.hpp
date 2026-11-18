/**
 * @file BBH_XCTS_driver.hpp
 * @brief Drivers of the BBH solver: a single binary through its stages and
 * resolution steps, and sequences.
 *
 * @copyright Copyright (c) 2026, GNU General Public Licensev3
 * This file is part of the KADATH library and published under
 * https://arxiv.org/abs/2103.09911
 */
#pragma once
#include <filesystem>
#include <string>

#include "BBH_XCTS.hpp"
#include "FUKA_Solvers/utilities/binary/binary_driver.hpp"
#include "utilities/BBH_XCTS_setup.hpp"

/**
 * \addtogroup BBH_XCTS
 * \ingroup FUKA
 * @{*/
namespace Kadath::FUKA_Solvers {

namespace BBH_XCTS_detail {
/**
 * @brief Enabled stages at the current resolution: TOTAL_BC (first at the
 * fixed orbital frequency if FIXED_GOMEGA, `TOTAL_BC_FIXED_OMEGA`, then in
 * quasi-equilibrium) and ECC_RED.
 */
inline void solve_stages(BBH_XCTS& solver,
                         kadath_config_boost<BIN_INFO>& bconfig) {
    auto& stages = bconfig.return_stages();
    auto solve_as = [&](STAGES const stage, std::string const& name) {
        solver.set_solver_stage() = stage;
        solver.set_stagename() = name;
        solver.solve();
    };
    if (stages[STAGES::TOTAL_BC]) {
        if (bconfig.control(CONTROLS::FIXED_GOMEGA)) {
            solve_as(STAGES::TOTAL_BC, "TOTAL_BC_FIXED_OMEGA");
            bconfig.control(CONTROLS::FIXED_GOMEGA) = false;
        }
        solve_as(STAGES::TOTAL_BC, "TOTAL_BC");
    }
    if (stages[STAGES::ECC_RED])
        solve_as(STAGES::ECC_RED, "ECC_RED");
}
}  // namespace BBH_XCTS_detail

/**
 * @brief Solve one BBH from its config (initial guess or previous
 * solution): every enabled stage at the initial resolution, then only the
 * last enabled stage after each resolution step.
 *
 * @param bconfig binary config; updated to the last solution
 * @param resolution resolution sequence (BIN_RES)
 * @param bh1_seq, bh2_seq fixing values of the holes
 * @param outputdir output directory
 */
template <class config_t>
int BBH_XCTS_solution_driver(config_t& bconfig,
                             Parameter_sequence<BIN_PARAMS> const& resolution,
                             bco_sequence<BCO_PARAMS, NODES> const& bh1_seq,
                             bco_sequence<BCO_PARAMS, NODES> const& bh2_seq,
                             std::string outputdir) {
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    require_solution_file(bconfig, rank);
    if (outputdir != "")
        std::filesystem::create_directory(outputdir);

    auto& stages = bconfig.return_stages();
    auto const [last_stage, last_stage_idx] = get_last_enabled(MSTAGE, stages);

    BBH_XCTS solver(&bconfig, bh1_seq, bh2_seq, resolution, outputdir, rank);
    auto delete_shift = [&]() {
        if (bconfig.control(CONTROLS::DELETE_SHIFT))
            solver.get_shift()->annule_hard();
    };

    delete_shift();
    BBH_XCTS_detail::solve_stages(solver, bconfig);
    while (solver.increment_resolution()) {
        stages.fill(false);
        stages[last_stage_idx] = true;
        solver.regrid();
        delete_shift();
        BBH_XCTS_detail::solve_stages(solver, bconfig);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    return EXIT_SUCCESS;
}

/**
 * @brief Solve a sequence of BBHs (see binary_sequence).
 */
template <class Seq_t, class config_t>
int BBH_XCTS_sequence(config_t& seqconfig,
                      Seq_t const& seq,
                      Parameter_sequence<BIN_PARAMS> const& resolution,
                      bco_sequence<BCO_PARAMS, NODES> const& bh1_seq,
                      bco_sequence<BCO_PARAMS, NODES> const& bh2_seq,
                      std::string outputdir) {
    return binary_sequence(
        seqconfig,
        seq,
        resolution,
        outputdir,
        {"bh", "bh"},
        {},
        [](config_t& c) { BBH_XCTS_setup_bin_config(c); },
        [](config_t& c) { BBH_XCTS_setup_space(c); },
        [&](config_t& c) {
            return BBH_XCTS_solution_driver(c,
                                            resolution,
                                            bh1_seq,
                                            bh2_seq,
                                            outputdir);
        });
}

/** @}*/
}  // namespace Kadath::FUKA_Solvers
