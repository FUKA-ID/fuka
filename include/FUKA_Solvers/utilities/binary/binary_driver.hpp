/**
 * @file binary_driver.hpp
 * @brief Sequence loop shared by the binary drivers.
 *
 * @copyright Copyright (c) 2026, GNU General Public Licensev3
 * This file is part of the KADATH library and published under
 * https://arxiv.org/abs/2103.09911
 */
#pragma once
#include <array>
#include <initializer_list>
#include <string>

#include "Configurator/config_binary.hpp"
#include "Solvers/sequences/parameter_sequence.hpp"
#include "Solvers/sequences/sequence_utilities.hpp"

/** \addtogroup Solver_utils
 * \ingroup FUKA
 * @{*/
namespace Kadath::FUKA_Solvers {

/**
 * @brief Solve a sequence of binaries (one parameter varied, or a single
 * binary if the sequence is unset). Each element starts from a new setup
 * when SEQUENCES is on (ADOT is kept if given).
 *
 * @param seqconfig sequence config
 * @param seq parameter sequence
 * @param resolution resolution sequence
 * @param outputdir output directory
 * @param types companion types, e.g. {"ns", "bh"}
 * @param ns_nodes companions whose EOS settings are copied
 * @param setup_bin_config new-setup binary parameters, f(config&)
 * @param setup_space new-setup initial guess, f(config&)
 * @param solve solution driver of one binary, int f(config&)
 * @return exit status of the last solution
 */
template <class config_t,
          class Seq_t,
          class setup_config_f,
          class setup_space_f,
          class solve_f>
int binary_sequence(config_t& seqconfig,
                    Seq_t const& seq,
                    Parameter_sequence<BIN_PARAMS> const& resolution,
                    std::string const& outputdir,
                    std::array<std::string, 2> const& types,
                    std::initializer_list<NODES> const ns_nodes,
                    setup_config_f&& setup_bin_config,
                    setup_space_f&& setup_space,
                    solve_f&& solve) {
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    config_t base_config =
        binary_generate_sequence_config(seqconfig, types, outputdir);
    for (NODES const bco : ns_nodes)
        update_eos_parameters(seqconfig, base_config, bco);
    if (rank == 0)
        base_config.write_config();
    base_config.set(resolution.get_indices()) = resolution.init();

    int exit_status = EXIT_SUCCESS;
    auto single = [&](auto const val) {
        config_t bconfig = base_config;
        if (seq.is_set())
            bconfig.set(seq.get_indices()) = val;

        if (bconfig.control(CONTROLS::SEQUENCES)) {
            // a manually set ADOT survives the new setup
            auto const adot = bconfig.set(BIN_PARAMS::ADOT);
            setup_bin_config(bconfig);
            bconfig.set(BIN_PARAMS::ADOT) = adot;
            setup_space(bconfig);
        }
        exit_status = solve(bconfig);
    };

    if (seq.is_set()) {
        auto const& dx = seq.step_size();
        for (auto val = seq.init(); seq.loop_condition(val); val += dx)
            single(val);
    } else {
        single(0);
    }
    return exit_status;
}

/** @}*/
}  // namespace Kadath::FUKA_Solvers
