/**
 * @file BHNS_XCTS.hpp
 * @brief Black hole - neutron star binaries in the XCTS formulation
 * (conformally flat, maximal slicing), the hole excised with apparent-horizon
 * boundary conditions.
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
#include "kadath_bhns.hpp"

/**
 * \addtogroup BHNS_XCTS
 * \ingroup FUKA
 * @{*/
namespace Kadath::FUKA_Solvers {
using namespace ::Kadath::FUKA_Config;

/**
 * @brief Space, solution, I/O, regrid, diagnostics and the equations shared
 * by the BHNS stages. The neutron star is companion 1, the black hole
 * companion 2.
 */
struct BHNS_XCTS_base : public FUKA_XCTS_Binary_base<Kadath::Space_bhns> {
    using binary_base_t = FUKA_XCTS_Binary_base<Kadath::Space_bhns>;

   protected:
    /// Residual scaling factor of the enthalpy (hydro-rescaling stages).
    internal_variable(double, H_scale);
    /// Central log enthalpy (hydrostatic stage variable).
    internal_variable(double, loghc);

   protected:
    BHNS_XCTS_base(base_config_t* config_,
                   bco_seq_t const& ns_seq_,
                   bco_seq_t const& bh_seq_,
                   bin_res_seq_t const& res_,
                   std::string outputdir_,
                   int const rank_);

    std::array<int, 2> nucleus_domains() const override {
        return {space->NS, space->BH};
    }

    /// Domains inside the excision surface.
    std::array<int, 2> excluded_doms() const {
        return {space->BH, space->BH + 1};
    }

    void update_config_quantities() override;

    /// Velocity potential, stellar spin and horizon shift conditions.
    void add_spin_equations();
    /**
     * @brief Apparent-horizon conditions (von Neumann lapse), horizon area
     * fixed by the irreducible mass, and the excised domains.
     */
    void add_horizon_equations();

   public:
    bool increment_resolution() override;
    void regrid() override;
    void write_regridded_solution(base_config_t& target) const override;
    void syst_init() override;

    void print_diagnostics(const int ite, const double conv) const override;
    std::string converged_filename(const std::string stage) const override;
};

/**
 * @brief BHNS stages, dispatched on the stage and FIXED_GOMEGA:
 * hydro rescaling (TOTAL_BC with FIXED_GOMEGA, ECC_RED) and hydrostatic
 * equilibrium (TOTAL_BC).
 */
template <class eos_t>
struct BHNS_XCTS : public BHNS_XCTS_base {
    BHNS_XCTS(base_config_t* config_,
              bco_seq_t const& ns_seq_,
              bco_seq_t const& bh_seq_,
              bin_res_seq_t const& res_,
              std::string outputdir_,
              int const rank_)
        : BHNS_XCTS_base(config_, ns_seq_, bh_seq_, res_, outputdir_, rank_) {
        reload();
        initialize_EOS(*this, NODES::BCO1);
    }

    void setup_syst() override {
        if (solver_stage == STAGES::ECC_RED ||
            bconfig->control(CONTROLS::FIXED_GOMEGA))
            setup_hydro_rescaling_syst();
        else
            setup_hydrostatic_equilibrium_syst();
    }

   private:
    void setup_hydro_rescaling_syst();
    void setup_hydrostatic_equilibrium_syst();
};

/** @}*/
}  // namespace Kadath::FUKA_Solvers

#include "BHNS_XCTS_imp.cpp"

#include "stages/BHNS_XCTS_hydro_rescaling.hpp"
#include "stages/BHNS_XCTS_hydrostatic_equilibrium.hpp"

#include "utilities/BHNS_XCTS_diagnostics.cpp"
#include "utilities/BHNS_XCTS_regrid.cpp"
#include "utilities/BHNS_XCTS_syst_init.cpp"
