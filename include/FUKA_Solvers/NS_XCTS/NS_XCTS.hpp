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

struct NS_XCTS_BASE {
    using base_space_t = Space_spheric_adapted;
    using base_config_t = Kadath::FUKA_Config::kadath_config_boost<
        Kadath::FUKA_Config::BCO_NS_INFO>;

    ptr_data_member(base_space_t, space, unique);
    ptr_data_member(base_config_t, bconfig, unique);

    using cfgen_t = CoordFields<base_space_t>;
    using cfary_t = std::array<std::optional<Vector>, NUM_VECTORS>;

   protected:
    internal_variable(int, rank)
    internal_variable(int, verbosity)
    internal_variable(int, ndom)
    internal_variable(std::string, outputdir)
    internal_variable(int, last_stage_idx)
    internal_variable(std::string, stagename)
    internal_variable(STAGES, solver_stage);

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
    ptr_data_member(ns_sequence const, seq, unique);
    ptr_data_member(Parameter_sequence<BCO_PARAMS>, resolution, unique);

    // Variable fields - i.e. Solution
    ptr_data_member(Scalar, conformal_factor, unique);
    ptr_data_member(Scalar, lapse, unique);
    ptr_data_member(Vector, shift, unique);
    ptr_data_member(Scalar, logh, unique);
    ptr_data_member(Scalar, diff_omega, unique);

    virtual ~NS_XCTS_BASE() { bconfig.release(); };

    NS_XCTS_BASE();
    NS_XCTS_BASE(base_config_t* config_,
                 ns_sequence const& seq_,
                 Parameter_sequence<BCO_PARAMS> const& res_,
                 std::string outputdir_,
                 int const rank_ = 0);
    virtual void save_to_file() const;
    bool increment_resolution();
    bool increment_seq();
    int do_newton();
    void regrid();

    virtual std::string converged_filename(const std::string stage) const = 0;

   protected:
    virtual void load_solution_from_file();
    void initialize_support_containers();
    virtual void reset_all_ptrs();
    virtual void update_config_quantities() = 0;
    virtual void print_diagnostics(const int ite, const double conv) const = 0;

   public:
    /**
   * @brief Consistent interface for writing a checkpoint
   *
   * @param termination_chkpt Toggle writing to stdout for termination
   */
    void checkpoint(bool termination_chkpt = false) const {
        // Backup activated stages
        auto const final_stages{bconfig->return_stages()};

        // Clear stages and only set the current stage as active
        auto& stages = bconfig->return_stages();
        stages.fill(false);
        stages[this->solver_stage] = true;

        // Save to file
        save_to_file();

        // Reset to original stages
        stages = final_stages;

        if (termination_chkpt) {
            std::stringstream ss;
            ss << "***Writing early termination chkpt "
               << bconfig->config_filename() << "***\n";
            throw std::runtime_error(ss.str().c_str());
        }
    }

    /**
   * @brief Branch for handling finding a previous solution
   *
   * @return bool
   */
    template <class... idx_t>
    bool solution_exists(idx_t... bco) {
        bool exists = false;
        std::string prev_name{converged_filename(stagename)};
        std::string prev_abs{bconfig->config_outputdir() + prev_name};

        const std::string home_kadath{std::getenv("HOME_KADATH")};
        std::string central_abs{home_kadath + "/COs/" + prev_name};

        auto check_and_update_config = [&](auto p) {
            exists = true;
            base_config_t old_solution(p + ".info");
            if (bconfig->set(BCO_PARAMS::NSHELLS, bco...) !=
                old_solution.set(BCO_PARAMS::NSHELLS, bco...))
                return false;

            // make sure we copy stages, controls, and settings over
            for (auto idx = 0; idx < STAGES::NUM_STAGES; ++idx)
                old_solution.set_stage(idx) = bconfig->set_stage(idx);
            for (auto idx : CONTROLS_ARY)
                old_solution.control(idx) = bconfig->control(idx);
            for (auto idx = 0; idx < SEQ_SETTINGS::NUM_SEQ_SETTINGS; ++idx)
                old_solution.seq_setting(idx) = bconfig->seq_setting(idx);

            auto& stages = bconfig->return_stages();
            auto [last_stage_name, last_stage_idx] =
                get_last_enabled(MSTAGE, stages);
            if (solver_stage != last_stage_idx)
                // Deactivate current stage since we found solution
                old_solution.set_stage(solver_stage) = false;
            *bconfig = old_solution;
            return exists;
        };

        /// Check based on current output directory
        if (fs::exists(prev_abs + ".info") && fs::exists(prev_abs + ".dat")) {
            exists = check_and_update_config(prev_abs);
        }
        /// Check based on global output directory
        else if (bconfig->control(SAVE_COS) &&
                 fs::exists(central_abs + ".info") &&
                 fs::exists(central_abs + ".dat")) {
            exists = check_and_update_config(central_abs);
        }

        return exists;
    }

    void reload() {
        reset_all_ptrs();
        load_solution_from_file();
        initialize_support_containers();
    }
};

template <class eos_t>
struct NS_XCTS_NOROT : NS_XCTS_BASE {
   private:
    void print_diagnostics(const int ite, const double conv) const override;
    void update_config_quantities() override;

   public:
    void syst_init();
    void setup_syst();
    std::string converged_filename(const std::string stage) const override;

    NS_XCTS_NOROT() = default;
    NS_XCTS_NOROT(std::string filename);
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
    void syst_init();
    void setup_syst();
    bool increment_spin();
    std::string converged_filename(const std::string stage) const;

    NS_XCTS_UNIFORM_ROT() = default;
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

   public:
    void syst_init();
    void setup_syst();
    bool increment_spin();
    std::string converged_filename(const std::string stage) const;

    NS_XCTS_DIFF_ROT() = default;
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
