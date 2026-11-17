/**
 * @file BH_XCTS_driver.hpp
 * @brief Drivers of the isolated BH solver: a single solution with resolution
 * steps, sequences, and the boosted holes of binary initial guesses.
 *
 * @copyright Copyright (c) 2026, GNU General Public Licensev3
 * This file is part of the KADATH library and published under
 * https://arxiv.org/abs/2103.09911
 */
#pragma once
#include "BH_XCTS.hpp"
#include "Solvers/sequences/parameter_sequence.hpp"
#include "Solvers/sequences/sequence_utilities.hpp"
#include "utilities/BH_XCTS_setup.hpp"
#include "utilities/boost/Boosted_BH_XCTS.hpp"

/**
 * \addtogroup BH_XCTS
 * \ingroup FUKA
 * @{*/
namespace Kadath::FUKA_Solvers {

using BH_XCTS_config_t = BH_XCTS_base::base_config_t;

/**
 * @brief Solve the BH of a config: stationary stages at the initial
 * resolution (with a spin-up for |chi| > 0.5 when the setup is new), the
 * linear boost if enabled, then the resolution steps up to the final one.
 *
 * @param bconfig config of the initial setup or of a previous solution;
 * updated to the final solution
 * @param resolution resolution sequence (BCO_RES)
 * @param outputdir output directory
 * @return int EXIT_SUCCESS
 */
inline int BH_XCTS_driver(BH_XCTS_config_t& bconfig,
                          Parameter_sequence<BCO_PARAMS>& resolution,
                          std::string outputdir);

/**
 * @brief Solve a sequence of BHs (one parameter varied); each element
 * starts from a new setup when SEQUENCES is on.
 *
 * @param seqconfig sequence config
 * @param seq parameter sequence (may be unset: a single solution)
 * @param resolution resolution sequence
 * @param outputdir output directory
 * @return config of the last solution
 */
template <class Seq_t>
BH_XCTS_config_t BH_XCTS_sequence(
    BH_XCTS_config_t& seqconfig,
    Seq_t const& seq,
    Parameter_sequence<BCO_PARAMS> const& resolution,
    std::string outputdir);

/**
 * @brief Boosted BH for a binary initial guess: stationary solution at the
 * initial resolution, boost, regrid to the final resolution (BCO_RES of the
 * input config), boost again.
 *
 * @param bconfig BH config; updated to the boosted solution
 * @param resolution resolution sequence (its initial order is used first)
 * @param outputdir output directory
 * @param binconfig binary config (DIST, GOMEGA)
 * @return int EXIT_SUCCESS
 */
inline int BH_XCTS_binary_boost_driver(
    BH_XCTS_config_t& bconfig,
    Parameter_sequence<BCO_PARAMS> const& resolution,
    std::string outputdir,
    kadath_config_boost<BIN_INFO>& binconfig);

/** @}*/
}  // namespace Kadath::FUKA_Solvers

#include "BH_XCTS_driver.cpp"
