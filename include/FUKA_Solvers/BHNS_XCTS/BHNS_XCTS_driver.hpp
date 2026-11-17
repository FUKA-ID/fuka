/**
 * @file BHNS_XCTS_driver.hpp
 * @brief Drivers of the BHNS solver: a single binary through its stages and
 * resolution steps, and sequences.
 *
 * @copyright Copyright (c) 2026, GNU General Public Licensev3
 * This file is part of the KADATH library and published under
 * https://arxiv.org/abs/2103.09911
 */
#pragma once
#include <filesystem>
#include <string>

#include "BHNS_XCTS.hpp"
#include "EOS/FUKA_EOS_Utilities.hh"
#include "FUKA_Solvers/utilities/binary/binary_driver.hpp"
#include "utilities/BHNS_XCTS_setup.hpp"

/**
 * \addtogroup BHNS_XCTS
 * \ingroup FUKA
 * @{*/
namespace Kadath::FUKA_Solvers {

namespace BHNS_XCTS_detail {
/**
 * @brief Enabled stages at the current resolution: TOTAL_BC (with a first
 * hydro-rescaling solution at fixed orbital frequency for a new setup) and
 * ECC_RED.
 */
template <class solver_t, class config_t>
void solve_stages(solver_t& solver, config_t& bconfig) {
    auto& stages = bconfig.return_stages();
    std::string const corot{bconfig.control(CONTROLS::COROT_BIN) ? "_COROT"
                                                                 : ""};
    auto solve_as = [&](STAGES const stage, std::string const& name) {
        solver.set_solver_stage() = stage;
        solver.set_stagename() = name + corot;
        solver.solve();
    };

    if (stages[STAGES::TOTAL_BC]) {
        if (bconfig.control(CONTROLS::FIXED_GOMEGA)) {
            solve_as(STAGES::TOTAL_BC, "TOTAL_BC_FIXED_OMEGA");
            // a new setup continues in hydrostatic equilibrium
            if (bconfig.control(CONTROLS::SEQUENCES)) {
                bconfig.control(CONTROLS::SEQUENCES) = false;
                bconfig.control(CONTROLS::FIXED_GOMEGA) = false;
                solve_as(STAGES::TOTAL_BC, "TOTAL_BC");
            }
        } else {
            solve_as(STAGES::TOTAL_BC, "TOTAL_BC");
        }
    }
    if (stages[STAGES::ECC_RED])
        solve_as(STAGES::ECC_RED, "ECC_RED");
}
}  // namespace BHNS_XCTS_detail

/// Solve one BHNS binary for the EOS type eos_t (EOS dispatch functor).
template <class eos_t>
struct launch_BHNS_XCTS_solver {
    template <class config_t>
    int operator()(const int rank,
                   config_t& bconfig,
                   Parameter_sequence<BIN_PARAMS> const& resolution,
                   bco_sequence<BCO_PARAMS, NODES> const& ns_seq,
                   bco_sequence<BCO_PARAMS, NODES> const& bh_seq,
                   std::string outputdir) {
        require_solution_file(bconfig, rank);
        if (outputdir != "")
            std::filesystem::create_directory(outputdir);

        // only the hydrostatic equilibrium is solved below the final
        // resolution; the other stages rescale the matter
        auto& stages = bconfig.return_stages();
        auto const final_stages{stages};
        if (resolution.final() > resolution.init()) {
            stages.fill(false);
            stages[STAGES::TOTAL_BC] = true;
        }

        BHNS_XCTS<eos_t> solver(&bconfig,
                                ns_seq,
                                bh_seq,
                                resolution,
                                outputdir,
                                rank);
        auto delete_shift = [&]() {
            if (bconfig.control(CONTROLS::DELETE_SHIFT))
                solver.get_shift()->annule_hard();
        };

        delete_shift();
        BHNS_XCTS_detail::solve_stages(solver, bconfig);
        while (solver.increment_resolution()) {
            solver.regrid();
            delete_shift();
            if (bconfig(resolution.get_indices()) >= resolution.final())
                stages = final_stages;
            BHNS_XCTS_detail::solve_stages(solver, bconfig);
        }
        return EXIT_SUCCESS;
    }
};

/**
 * @brief Solve one BHNS binary from its config (initial guess or previous
 * solution).
 *
 * @param bconfig binary config; updated to the last solution
 * @param resolution resolution sequence (BIN_RES)
 * @param ns_seq, bh_seq fixing values of the star and the hole
 * @param outputdir output directory
 */
template <class config_t>
int BHNS_XCTS_solution_driver(config_t& bconfig,
                              Parameter_sequence<BIN_PARAMS> const& resolution,
                              bco_sequence<BCO_PARAMS, NODES> const& ns_seq,
                              bco_sequence<BCO_PARAMS, NODES> const& bh_seq,
                              std::string outputdir) {
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    using namespace Kadath::FUKA_EOS;
    const std::string eos_type =
        bconfig.template eos<std::string>(EOS_PARAMS::EOSTYPE, NODES::BCO1);
    return EOS_Function_Dispatcher::dispatch<launch_BHNS_XCTS_solver>(
        bconfig,
        eos_type,
        rank,
        bconfig,
        resolution,
        ns_seq,
        bh_seq,
        outputdir);
}

/**
 * @brief Solve a sequence of BHNS binaries (see binary_sequence).
 */
template <class Seq_t, class config_t>
int BHNS_XCTS_sequence(config_t& seqconfig,
                       Seq_t const& seq,
                       Parameter_sequence<BIN_PARAMS> const& resolution,
                       bco_sequence<BCO_PARAMS, NODES> const& ns_seq,
                       bco_sequence<BCO_PARAMS, NODES> const& bh_seq,
                       std::string outputdir) {
    return binary_sequence(
        seqconfig,
        seq,
        resolution,
        outputdir,
        {"ns", "bh"},
        {NODES::BCO1},
        [](config_t& c) { BHNS_XCTS_setup_bin_config(c); },
        [](config_t& c) { BHNS_XCTS_setup_space(c); },
        [&](config_t& c) {
            return BHNS_XCTS_solution_driver(c,
                                             resolution,
                                             ns_seq,
                                             bh_seq,
                                             outputdir);
        });
}

/** @}*/
}  // namespace Kadath::FUKA_Solvers
