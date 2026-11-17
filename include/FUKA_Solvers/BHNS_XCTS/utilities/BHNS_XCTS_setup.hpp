/**
 * @file BHNS_XCTS_setup.hpp
 * @brief New BHNS setup: binary parameters and the initial guess superposed
 * from an isolated (boosted) neutron star (NS_XCTS) and black hole (BH_XCTS).
 *
 * @copyright Copyright (c) 2026, GNU General Public Licensev3
 * This file is part of the KADATH library and published under
 * https://arxiv.org/abs/2103.09911
 */
#pragma once
#include <array>
#include <string>

#include "Configurator/config_binary.hpp"

/**
 * \addtogroup BHNS_XCTS
 * \ingroup FUKA
 * @{*/
namespace Kadath::FUKA_Solvers {

/// Binary parameters of a new BHNS setup (REXT, q, COM, orbital frequency).
template <class config_t>
void BHNS_XCTS_setup_bin_config(config_t& bconfig);

/// Solve the isolated companions and write the superposed initial guess;
/// every rank reads the new config afterwards.
template <class config_t>
void BHNS_XCTS_setup_space(config_t& bconfig);

/// Superpose the isolated star and hole of the given solution files (rank 0).
template <class config_t>
void BHNS_XCTS_superimposed_import(config_t& bconfig,
                                   std::array<std::string, 2> co_filenames);

/** @}*/
}  // namespace Kadath::FUKA_Solvers

#include "BHNS_XCTS_setup.cpp"
