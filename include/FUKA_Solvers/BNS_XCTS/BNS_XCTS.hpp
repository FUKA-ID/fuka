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
#include "FUKA_Solvers/FUKA_Solver_base.hpp"
#include "Solvers/sequences/bco_sequence.hpp"
#include "kadath_bin_ns.hpp"

/**
 * \addtogroup BNS_XCTS
 * \ingroup FUKA
 * @{*/
namespace Kadath::FUKA_Solvers {
using namespace ::Kadath::FUKA_Config;

using seq_t = bco_sequence<BCO_PARAMS, NODES>;
using res_seq_t = Parameter_sequence<BIN_PARAMS>;

struct BNS_XCTS_base : public FUKA_Solver_base {
    using base_space_t = Kadath::Space_bin_ns;
    using base_config_t = kadath_config_boost<BIN_INFO>;

    ptr_data_member(base_space_t, space, unique);
    ptr_data_member(base_config_t, bconfig, unique);

    using cfgen_t = CoordFields<base_space_t>;
    using cfary_t = std::array<std::optional<Vector>, NUM_VECTORS>;

   protected:
    void populate_domain_support_containers() {
        // Set space specific quantities
        xc1 = bco_utils::get_center(*space, space->NS1);
        xc2 = bco_utils::get_center(*space, space->NS2);
        xo = bco_utils::get_center(*space, ndom - 1);
    }

    // EOS Parameters - Perhaps this should be a container?
    internal_variable(double, h_cut);
    internal_variable(std::string, eos_file);
    internal_variable(std::string, eos_type);

    // Support containers
    ptr_data_member(Base_tensor, basis, unique);
    ptr_data_member(Metric_flat, fmet, unique);
    ptr_data_member(System_of_eqs, syst, unique);
    ptr_data_member(cfgen_t, cfields, unique);
    ptr_data_member(cfary_t, coord_vectors, unique);

    // Sequence containers
    ptr_data_member(seq_t const, ns1_seq, unique);
    ptr_data_member(seq_t const, ns2_seq, unique);
    ptr_data_member(res_seq_t, resolution, unique);

    // Variable fields - i.e. Solution
    ptr_data_member(Scalar, conformal_factor, unique);
    ptr_data_member(Scalar, lapse, unique);
    ptr_data_member(Vector, shift, unique);
    ptr_data_member(Scalar, logh, unique);
    ptr_data_member(Scalar, phi, unique);

    ptr_data_member(Scalar, logh_const, unique);

    // Domain centers
    internal_variable(double, xc1);
    internal_variable(double, xc2);
    internal_variable(double, xo);
    internal_variable(double, H_scale1);
    internal_variable(double, H_scale2);

    virtual ~BNS_XCTS_base() { bconfig.release(); };

    BNS_XCTS_base();

    BNS_XCTS_base(BNS_XCTS_base::base_config_t* config_,
                  seq_t const& ns1_seq_,
                  seq_t const& ns2_seq_,
                  res_seq_t const& res_,
                  std::string outputdir_,
                  int const rank_);

    bool increment_resolution() override;

    void regrid() override;
    void syst_init() override;

    void reset_all_ptrs() override { reset_base_ptrs(); }

   protected:
    // Initialization Methods
    void initialize_support_containers();

    // I/O Methods
    void save_solution_to_file() const override;
    void load_solution_from_file() override;
    void update_config_quantities() override {};
    void reset_base_ptrs();

   public:
    int do_newton() override;
    void checkpoint(bool termination_chkpt = false) const;
    void print_diagnostics(const int ite, const double conv) override;
    std::string converged_filename(const std::string stage) const override;
};

template <class eos_t>
struct BNS_XCTS : public BNS_XCTS_base {
    BNS_XCTS(BNS_XCTS_base::base_config_t* config_,
             seq_t const& ns1_seq_,
             seq_t const& ns2_seq_,
             res_seq_t const& res_,
             std::string outputdir_,
             int const rank_)
        : BNS_XCTS_base(config_, ns1_seq_, ns2_seq_, res_, outputdir_, rank_) {
        load_solution_from_file();
        initialize_EOS(*this, NODES::BCO1);
        initialize_support_containers();

        populate_domain_support_containers();
    }

    void setup_hydro_rescaling_syst();
    void setup_hydrostatic_equilibrium_stage();
};

/** @}*/
};  // namespace Kadath::FUKA_Solvers

#include "BNS_XCTS_imp.cpp"
#include "stages/BNS_XCTS_hydro_rescaling.hpp"
#include "stages/BNS_XCTS_hydrostatic_equilibrium.hpp"
#include "utilities/BNS_XCTS_diagnostics.cpp"
#include "utilities/BNS_XCTS_do_newton.cpp"
#include "utilities/BNS_XCTS_regrid.cpp"
#include "utilities/BNS_XCTS_syst_init.cpp"