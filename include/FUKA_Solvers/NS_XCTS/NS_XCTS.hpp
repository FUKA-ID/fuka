/**
 * @file NS_XCTS.hpp
 * @author Samuel Tootle (sdtootle@gmail.com)
 * @brief This is a rewrite of the isolated neutron star solvers using the XCTS
 * formulation The rewrite was both motivated to drastically clean up the code
 * for future maintenance as well as prepare for merger with Kadath master
 * branch
 * @date 2024-02-21
 *
 * @copyright Copyright (c) 2024, GNU General Public Licensev3
 * This file is part of the KADATH library and published under
 * https://arxiv.org/abs/2103.09911
 */
#pragma once
#include "codes_utilities.hpp"
#include "kadath.hpp"
#include "kadath_adapted.hpp"

#include "Configurator/config_bco.hpp"
#include "EOS/EOS.hh"
#include "FUKA_Solvers/FUKA_XCTS_Solver_base.hpp"
#include "bco_utilities.hpp"
#include "coord_fields.hpp"

#include "Solvers/sequences/ns_sequence.hpp"
#include "Solvers/sequences/parameter_sequence.hpp"

/**
 * \addtogroup NS_XCTS
 * \ingroup FUKA
 * @{*/
namespace Kadath::FUKA_Solvers {
using namespace ::Kadath::FUKA_Config;

struct NS_XCTS_BASE
    : public FUKA_XCTS_Solver_base<Space_spheric_adapted,
                                   Kadath::FUKA_Config::kadath_config_boost<
                                       Kadath::FUKA_Config::BCO_NS_INFO>> {
    using solver_base_t =
        FUKA_XCTS_Solver_base<Space_spheric_adapted,
                              Kadath::FUKA_Config::kadath_config_boost<
                                  Kadath::FUKA_Config::BCO_NS_INFO>>;

   protected:
    // EOS Parameters - Perhaps this should be a container?
    internal_variable(double, h_cut);
    internal_variable(std::string, eos_file);
    internal_variable(std::string, eos_type);

    // Sequence containers
    ptr_data_member(ns_sequence const, seq, unique);
    ptr_data_member(Parameter_sequence<BCO_PARAMS>, resolution, unique);

    // Variable fields - i.e. Solution
    ptr_data_member(Scalar, conformal_factor, unique);
    ptr_data_member(Scalar, lapse, unique);
    ptr_data_member(Vector, shift, unique);
    ptr_data_member(Scalar, logh, unique);
    ptr_data_member(Scalar, diff_omega, unique);

    NS_XCTS_BASE(base_config_t* config_,
                 ns_sequence const& seq_,
                 Parameter_sequence<BCO_PARAMS> const& res_,
                 std::string outputdir_,
                 int const rank_ = 0);

    void save_solution_to_file() const override;
    void load_solution_from_file() override;
    void initialize_support_containers() override;
    bool is_compatible_solution(base_config_t& old_config) override;

   public:
    /// Members a revert needs (Solution_snapshot).
    static constexpr auto snapshot_members() {
        return std::make_tuple(&NS_XCTS_BASE::space,
                               &NS_XCTS_BASE::bconfig,
                               &NS_XCTS_BASE::conformal_factor,
                               &NS_XCTS_BASE::lapse,
                               &NS_XCTS_BASE::shift,
                               &NS_XCTS_BASE::logh,
                               &NS_XCTS_BASE::diff_omega);
    }

    void reset_all_ptrs() override;
    bool increment_resolution() override;
    bool increment_seq();
    void regrid() override;
    void write_regridded_solution(base_config_t& target) const override;
};

template <class eos_t>
struct NS_XCTS_NOROT : NS_XCTS_BASE {
   private:
    void print_diagnostics(const int ite, const double conv) const override;
    void update_config_quantities() override;

   public:
    void syst_init() override;
    void setup_syst() override;
    std::string converged_filename(const std::string stage) const override;

    NS_XCTS_NOROT(base_config_t* config_,
                  ns_sequence const& seq_,
                  Parameter_sequence<BCO_PARAMS> const& res_,
                  std::string outputdir_,
                  int const rank_ = 0);
};

template <class eos_t>
struct NS_XCTS_UNIFORM_ROT : NS_XCTS_BASE {
   private:
    void print_diagnostics(const int ite, const double conv) const override;
    void update_config_quantities() override;
    void initialize_spinup();
    ptr_data_member(Parameter_sequence<BCO_PARAMS>, spinup, unique);

   public:
    void syst_init() override;
    void setup_syst() override;
    bool increment_spin();
    std::string converged_filename(const std::string stage) const override;

    NS_XCTS_UNIFORM_ROT(base_config_t* config_,
                        ns_sequence const& seq_,
                        Parameter_sequence<BCO_PARAMS> const& res_,
                        std::string outputdir_,
                        int const rank_ = 0);
};

template <class eos_t>
struct NS_XCTS_DIFF_ROT : NS_XCTS_BASE {
   private:
    void print_diagnostics(const int ite, const double conv) const override;
    void update_config_quantities() override;
    void initialize_spinup();
    void initialize_diffrot_params();

    /// Variables needed for DIFFROT solver
    using diffparams_t = std::array<double, DIFFROT_PARAMS::NUM_DIFFROT_PARAMS>;
    internal_variable(diffparams_t, diffrot_params);
    ptr_data_member(Parameter_sequence<DIFFROT_PARAMS>, spinup, unique);
    ptr_data_member(Kadath::Scalar, ones, unique);
    ptr_data_member(Kadath::Index, pos_origin, unique);
    ptr_data_member(Kadath::Index, pos_eq, unique);
    ptr_data_member(Kadath::Index, pos_pole, unique);
    internal_variable(double, diffA);
    internal_variable(double, axis_ratio);
    internal_variable(std::string, law);

   protected:
    void initialize_support_containers() override;

   public:
    void syst_init() override;
    void setup_syst() override;
    bool increment_spin();
    std::string converged_filename(const std::string stage) const override;

    NS_XCTS_DIFF_ROT(base_config_t* config_,
                     ns_sequence const& seq_,
                     Parameter_sequence<BCO_PARAMS> const& res_,
                     std::string outputdir_,
                     int const rank_ = 0);

    void reset_all_ptrs() override;
};

/** @}*/
};  // namespace Kadath::FUKA_Solvers

#include "NS_XCTS_BASE.cpp"
#include "NS_XCTS_DIFF_ROT.cpp"
#include "NS_XCTS_NOROT.cpp"
#include "NS_XCTS_UNIFORM_ROT.cpp"
