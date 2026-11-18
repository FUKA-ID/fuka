/**
 * @file BBH_XCTS_setup.hpp
 * @brief New BBH setup: binary parameters and the initial guess superposed
 * from two isolated (boosted) BH_XCTS holes.
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
 * \addtogroup BBH_XCTS
 * \ingroup FUKA
 * @{*/
namespace Kadath::FUKA_Solvers {

/// Binary parameters of a new BBH setup (REXT, q, COM, orbital frequency).
template <class config_t>
void BBH_XCTS_setup_bin_config(config_t& bconfig);

/// Solve the isolated holes and write the superposed initial guess; every
/// rank reads the new config afterwards.
template <class config_t>
void BBH_XCTS_setup_space(config_t& bconfig);

/// Superpose the isolated holes of the given solution files (rank 0).
inline void BBH_XCTS_superimposed_import(
    kadath_config_boost<BIN_INFO>& bconfig,
    std::array<std::string, 2> const& bh_filenames);

/** @}*/
}  // namespace Kadath::FUKA_Solvers

#include "BBH_XCTS_setup.cpp"
