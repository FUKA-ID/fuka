/**
 * @file BH_XCTS.hpp
 * @brief Isolated black hole in the XCTS formulation (conformally flat,
 * maximal slicing), excised with apparent-horizon boundary conditions.
 *
 * Equations: arXiv:2103.09911 (FUKA) and arXiv:0805.4192.
 *
 * @copyright Copyright (c) 2026, GNU General Public Licensev3
 * This file is part of the KADATH library and published under
 * https://arxiv.org/abs/2103.09911
 */
#pragma once
#include <array>
#include <string>
#include <tuple>

#include "FUKA_Solvers/FUKA_XCTS_Solver_base.hpp"
#include "Solvers/sequences/parameter_sequence.hpp"
#include "kadath_adapted_bh.hpp"

/**
 * \addtogroup BH_XCTS
 * \ingroup FUKA
 * @{*/
namespace Kadath::FUKA_Solvers {

/**
 * @brief Space, solution fields (conformal factor, lapse, shift), I/O, regrid
 * and diagnostics shared by the BH stages.
 */
struct BH_XCTS_base
    : public FUKA_XCTS_Solver_base<Space_adapted_bh,
                                   kadath_config_boost<BCO_BH_INFO>> {
    using solver_base_t =
        FUKA_XCTS_Solver_base<Space_adapted_bh,
                              kadath_config_boost<BCO_BH_INFO>>;
    using res_seq_t = Parameter_sequence<BCO_PARAMS>;

   protected:
    ptr_data_member(res_seq_t, resolution, unique);

    // Solution
    ptr_data_member(Scalar, conformal_factor, unique);
    ptr_data_member(Scalar, lapse, unique);
    ptr_data_member(Vector, shift, unique);

    /// Domains inside the excision surface, where the fields are zero.
    static constexpr std::array<int, 2> excluded_doms{0, 1};

    /**
     * @param config_ config of the solution to load (owned by the caller)
     * @param res_ resolution sequence
     * @param outputdir_ output directory
     * @param rank_ MPI rank
     */
    BH_XCTS_base(base_config_t* config_,
                 res_seq_t const& res_,
                 std::string const& outputdir_,
                 int const rank_);

    void initialize_support_containers() override;
    void save_solution_to_file() const override;
    void load_solution_from_file() override;
    void update_config_quantities() override;
    void print_diagnostics(const int ite, const double conv) const override;

    /// Definitions of the spin integrand and of the constraint equations.
    void add_constraint_definitions();
    /**
     * @brief Constraint equations, boundary conditions at infinity and on the
     * excision surface, horizon mass and spin fixing.
     *
     * @param lapse_bc lapse condition on the excision surface
     * @param shift_inf_bc shift condition at infinity
     * @param shift_bh_bc shift condition on the excision surface
     */
    void add_equations(std::string const& lapse_bc,
                       std::string const& shift_inf_bc,
                       std::string const& shift_bh_bc);

   public:
    /// Members a revert needs (Solution_snapshot).
    static constexpr auto snapshot_members() {
        return std::make_tuple(&BH_XCTS_base::space,
                               &BH_XCTS_base::bconfig,
                               &BH_XCTS_base::conformal_factor,
                               &BH_XCTS_base::lapse,
                               &BH_XCTS_base::shift);
    }

    std::string converged_filename(const std::string stage) const override;
    void syst_init() override;
    void reset_all_ptrs() override;
    bool increment_resolution() override;
    void regrid() override;
    void write_regridded_solution(base_config_t& target) const override;
};

/**
 * @brief Stationary BH stages: TOTAL (fixed lapse on the horizon), TOTAL_BC
 * (von Neumann lapse condition) and LINBOOST (TOTAL_BC with a linear boost,
 * BVELX/BVELY, at infinity).
 */
struct BH_XCTS : public BH_XCTS_base {
    BH_XCTS(base_config_t* config_,
            res_seq_t const& res_,
            std::string const& outputdir_,
            int const rank_);

    /// Select the stage solved by the next setup_syst() / solve().
    void set_stage(STAGES const stage);

    void setup_syst() override;
};

/** @}*/
}  // namespace Kadath::FUKA_Solvers

#include "BH_XCTS_imp.cpp"

#include "stages/BH_XCTS_stages.cpp"

#include "utilities/BH_XCTS_regrid.cpp"
