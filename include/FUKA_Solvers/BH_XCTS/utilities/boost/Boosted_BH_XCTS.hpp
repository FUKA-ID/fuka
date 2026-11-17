/**
 * @file Boosted_BH_XCTS.hpp
 * @brief Isolated BH boosted by the orbital rotation of a binary, used as
 * the initial guess of binary initial data.
 *
 * @copyright Copyright (c) 2026, GNU General Public Licensev3
 * This file is part of the KADATH library and published under
 * https://arxiv.org/abs/2103.09911
 */
#pragma once
#include "Configurator/config_binary.hpp"
#include "FUKA_Solvers/BH_XCTS/BH_XCTS.hpp"

/**
 * \addtogroup BH_XCTS
 * \ingroup FUKA
 * @{*/
namespace Kadath::FUKA_Solvers {

/**
 * @brief BIN_BOOST stage: von Neumann lapse condition with the horizon shift
 * condition in the frame co-rotating with the binary (orbital frequency
 * GOMEGA of the binary config).
 */
struct Boosted_BH_XCTS : public BH_XCTS_base {
    using binary_config_t = kadath_config_boost<BIN_INFO>;

    /**
     * @param config_ config of the stationary solution to boost (owned by the
     * caller)
     * @param res_ resolution sequence
     * @param outputdir_ output directory
     * @param binary_config_ binary config providing DIST and GOMEGA (owned by
     * the caller)
     * @param rank_ MPI rank
     */
    Boosted_BH_XCTS(base_config_t* config_,
                    res_seq_t const& res_,
                    std::string const& outputdir_,
                    binary_config_t* binary_config_,
                    int const rank_);

    void setup_syst() override;

   private:
    binary_config_t* binary_config;
};

/** @}*/
}  // namespace Kadath::FUKA_Solvers

#include "Boosted_BH_XCTS_imp.cpp"
