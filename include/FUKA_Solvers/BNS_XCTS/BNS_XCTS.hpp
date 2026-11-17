/**
 * @file BNS_XCTS.hpp
 * @author Samuel Tootle (sdtootle@gmail.com)
 * @brief This is the base class for computing Black Hole - Neutron Star
 * binaries using the XCTS formulation The rewrite was both motivated to
 * drastically clean up the code for future maintenance as well as prepare for
 * merger with Kadath master branch
 * @date 2025-11
 *
 * @copyright Copyright (c) 2025, GNU General Public Licensev3 This file
 * is part of the KADATH library and published under
 * https://arxiv.org/abs/2103.09911
 */
#pragma once
#include "FUKA_Solvers/FUKA_XCTS_Binary_base.hpp"
#include "FUKA_Solvers/utilities/binary/binary_syst.hpp"
#include "kadath_bin_ns.hpp"

/**
 * \addtogroup BNS_XCTS
 * \ingroup FUKA
 * @{*/
namespace Kadath::FUKA_Solvers {
using namespace ::Kadath::FUKA_Config;

struct BNS_XCTS_base : public FUKA_XCTS_Binary_base<Kadath::Space_bin_ns> {
    using binary_base_t = FUKA_XCTS_Binary_base<Kadath::Space_bin_ns>;

   protected:
    // Residual scaling factors of the enthalpy (hydro-rescaling stages)
    internal_variable(double, H_scale1);
    internal_variable(double, H_scale2);

   protected:
    BNS_XCTS_base(base_config_t* config_,
                  bco_seq_t const& ns1_seq_,
                  bco_seq_t const& ns2_seq_,
                  bin_res_seq_t const& res_,
                  std::string outputdir_,
                  int const rank_);

    std::array<int, 2> nucleus_domains() const override {
        return {space->NS1, space->NS2};
    }

    std::unique_ptr<base_space_t> read_space(FILE* ff) const override;

    void update_config_quantities() override {}

   public:
    bool increment_resolution() override;
    void regrid() override;
    void write_regridded_solution(base_config_t& target) const override;
    void syst_init() override;

    void print_diagnostics(const int ite, const double conv) const override;
    std::string converged_filename(const std::string stage) const override;
};

template <class eos_t>
struct BNS_XCTS : public BNS_XCTS_base {
    BNS_XCTS(BNS_XCTS_base::base_config_t* config_,
             bco_seq_t const& ns1_seq_,
             bco_seq_t const& ns2_seq_,
             bin_res_seq_t const& res_,
             std::string outputdir_,
             int const rank_)
        : BNS_XCTS_base(config_, ns1_seq_, ns2_seq_, res_, outputdir_, rank_) {
        reload();

        initialize_EOS(*this, NODES::BCO1);
    }

   private:
    void setup_hydro_rescaling_syst();
    void setup_hydrostatic_equilibrium_stage();

   public:
    void setup_syst() override {
        if (solver_stage == STAGES::PRE || solver_stage == STAGES::TOTAL_BC ||
            solver_stage == STAGES::ECC_RED) {
            setup_hydro_rescaling_syst();
        } else if (solver_stage == STAGES::TOTAL) {
            setup_hydrostatic_equilibrium_stage();
        }
    }
};

/** @}*/
};  // namespace Kadath::FUKA_Solvers

#include "BNS_XCTS_imp.cpp"
#include "stages/BNS_XCTS_hydro_rescaling.hpp"
#include "stages/BNS_XCTS_hydrostatic_equilibrium.hpp"
#include "utilities/BNS_XCTS_diagnostics.cpp"
#include "utilities/BNS_XCTS_regrid.cpp"
#include "utilities/BNS_XCTS_syst_init.cpp"