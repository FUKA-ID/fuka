/**
 * @file BBH_XCTS.hpp
 * @brief Binary black holes in the XCTS formulation (conformally flat,
 * maximal slicing), both holes excised with apparent-horizon boundary
 * conditions.
 *
 * Equations: arXiv:2103.09911 (FUKA).
 *
 * @copyright Copyright (c) 2026, GNU General Public Licensev3
 * This file is part of the KADATH library and published under
 * https://arxiv.org/abs/2103.09911
 */
#pragma once
#include <array>
#include <string>

#include "FUKA_Solvers/FUKA_XCTS_Binary_base.hpp"
#include "FUKA_Solvers/utilities/binary/binary_syst.hpp"
#include "kadath_bin_bh.hpp"

/**
 * \addtogroup BBH_XCTS
 * \ingroup FUKA
 * @{*/
namespace Kadath::FUKA_Solvers {
using namespace ::Kadath::FUKA_Config;

/**
 * @brief BBH solver: TOTAL_BC (quasi-equilibrium, first at a fixed orbital
 * frequency for a new setup) and ECC_RED (fixed orbital frequency and radial
 * velocity), both with vanishing ADM linear momentum.
 */
struct BBH_XCTS : public FUKA_XCTS_Binary_base<Kadath::Space_bin_bh> {
    using binary_base_t = FUKA_XCTS_Binary_base<Kadath::Space_bin_bh>;

    /**
     * @param config_ config of the solution to load (owned by the caller)
     * @param bh1_seq_, bh2_seq_ fixing values of the holes
     * @param res_ resolution sequence (BIN_RES)
     * @param outputdir_ output directory
     * @param rank_ MPI rank
     */
    BBH_XCTS(base_config_t* config_,
             bco_seq_t const& bh1_seq_,
             bco_seq_t const& bh2_seq_,
             bin_res_seq_t const& res_,
             std::string outputdir_,
             int const rank_);

   protected:
    std::array<int, 2> nucleus_domains() const override {
        return {space->BH1, space->BH2};
    }

    /// Domains inside the excision surfaces.
    std::array<int, 4> excluded_doms() const {
        return {space->BH1, space->BH1 + 1, space->BH2, space->BH2 + 1};
    }

    void update_config_quantities() override;

   public:
    bool increment_resolution() override;
    void regrid() override;
    void write_regridded_solution(base_config_t& target) const override;
    void syst_init() override;
    void setup_syst() override;

    void print_diagnostics(const int ite, const double conv) const override;
    std::string converged_filename(const std::string stage) const override;
};

/** @}*/
}  // namespace Kadath::FUKA_Solvers

#include "BBH_XCTS_imp.cpp"

#include "stages/BBH_XCTS_stages.cpp"

#include "utilities/BBH_XCTS_diagnostics.cpp"
#include "utilities/BBH_XCTS_regrid.cpp"
