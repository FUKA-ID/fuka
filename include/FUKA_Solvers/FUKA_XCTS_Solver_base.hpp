/**
 * @file FUKA_XCTS_Solver_base.hpp
 * @brief Common state and solve loop of the FUKA XCTS solvers.
 *
 * @copyright Copyright (c) 2026, GNU General Public Licensev3
 * This file is part of the KADATH library and published under
 * https://arxiv.org/abs/2103.09911
 */
#pragma once
#include <mpi.h>
#include <array>
#include <cmath>
#include <memory>
#include <optional>
#include <sstream>
#include <string>

#include "FUKA_Solvers/FUKA_Solver_base.hpp"
#include "FUKA_Solvers/utilities/solver/fuka_path_tools.hpp"
#include "FUKA_Solvers/utilities/solver/solver_utilities.hpp"

/** \addtogroup Solver_base
 * \ingroup FUKA
 * @{
 */
namespace Kadath::FUKA_Solvers {

/**
 * @brief State and methods shared by the XCTS solvers: the solver owns its
 * space, support containers and system; the config is owned by the driver.
 *
 * Derived solvers own their solution fields, implement the virtual interface
 * of FUKA_Solver_base and list the members a revert needs in
 * `snapshot_members()` (see utilities/solver/solution_snapshot.hpp).
 *
 * @tparam space_t numerical space type
 * @tparam config_t Configurator type
 */
template <class space_t, class config_t>
struct FUKA_XCTS_Solver_base : public FUKA_Solver_base {
    using base_space_t = space_t;
    using base_config_t = config_t;
    using cfgen_t = CoordFields<base_space_t>;
    using cfary_t = std::array<std::optional<Vector>, NUM_VECTORS>;

    ptr_data_member(base_space_t, space, unique);

   protected:
    /// Non-owning: the driver owns the config and must outlive the solver.
    base_config_t* bconfig;

   public:
    base_config_t* get_bconfig() const { return bconfig; }

   protected:
    // Support containers, rebuilt from the space and fields
    ptr_data_member(Base_tensor, basis, unique);
    ptr_data_member(Metric_flat, fmet, unique);
    ptr_data_member(System_of_eqs, syst, unique);
    ptr_data_member(cfgen_t, cfields, unique);
    ptr_data_member(cfary_t, coord_vectors, unique);

    FUKA_XCTS_Solver_base(base_config_t* config_,
                          std::string const& outputdir_,
                          int const rank_)
        : FUKA_Solver_base(), bconfig(config_) {
        this->rank = rank_;
        this->outputdir = outputdir_;
        auto& stage_enabled = bconfig->return_stages();
        auto [last_stage_, last_stage_idx_] =
            get_last_enabled_no_throw(MSTAGE, stage_enabled);
        this->last_stage_idx = last_stage_idx_;
    }

    /// Release the containers that refer to the space; the system goes first.
    void reset_support_containers() {
        syst.reset(nullptr);
        basis.reset(nullptr);
        fmet.reset(nullptr);
        cfields.reset(nullptr);
        coord_vectors.reset(nullptr);
    }

    /// Update the coordinate fields of the system; isolated object at the
    /// origin by default.
    virtual void update_coordinate_fields() {
        update_fields_co(*cfields, *coord_vectors, {}, 0., syst);
    }

    /// Whether a converged solution found on disk may replace the current one.
    virtual bool is_compatible_solution(base_config_t& /*old_config*/) {
        return true;
    }

   public:
    std::string converged_filename(const std::string stage) const override = 0;

    /**
     * @brief Write the current solution with only the current stage enabled.
     *
     * @param termination_chkpt throw after writing (early termination)
     */
    void checkpoint(bool termination_chkpt = false) const {
        auto& stages = bconfig->return_stages();
        auto const final_stages{stages};
        stages.fill(false);
        stages[this->solver_stage] = true;

        save_solution_to_file();

        stages = final_stages;

        if (termination_chkpt) {
            std::stringstream ss;
            ss << "***Writing early termination chkpt "
               << bconfig->config_filename() << "***\n";
            throw std::runtime_error(ss.str());
        }
    }

    /**
     * @brief Look for the converged solution of the current stage in the
     * output directory, the config's directory, then the central COs
     * directory (SAVE_COS). When found, the config is replaced by the
     * solution's config, keeping the current stages, controls and settings.
     *
     * Rank 0 decides and broadcasts the path: the config's directory can
     * differ between ranks (only rank 0 writes and updates it), and every
     * rank must take the same branch.
     *
     * @return true if a compatible solution was found
     */
    bool solution_exists() {
        std::string const prev_name{converged_filename(stagename)};
        std::string path{};
        if (rank == 0) {
            auto found = [](std::string const& p) {
                return fs::exists(p + ".info") && fs::exists(p + ".dat");
            };
            std::array<std::string, 3> const candidates{
                (fs::path(outputdir) / prev_name).string(),
                bconfig->config_outputdir() + prev_name,
                bconfig->control(SAVE_COS) ? get_cos_path() + prev_name : ""};
            for (auto const& c : candidates) {
                if (!c.empty() && found(c)) {
                    path = c;
                    break;
                }
            }
        }
        broadcast_from_root(path);
        if (path.empty())
            return false;

        base_config_t old_solution(path + ".info");
        if (!is_compatible_solution(old_solution))
            return false;

        for (auto idx = 0; idx < STAGES::NUM_STAGES; ++idx)
            old_solution.set_stage(idx) = bconfig->set_stage(idx);
        for (auto idx : CONTROLS_ARY)
            old_solution.control(idx) = bconfig->control(idx);
        for (auto idx = 0; idx < SEQ_SETTINGS::NUM_SEQ_SETTINGS; ++idx)
            old_solution.seq_setting(idx) = bconfig->seq_setting(idx);

        auto [last_stage_name, last_idx] =
            get_last_enabled(MSTAGE, bconfig->return_stages());
        // the current stage is done, unless it is the last one
        if (solver_stage != last_idx)
            old_solution.set_stage(solver_stage) = false;
        *bconfig = old_solution;
        return true;
    }

    /**
     * @brief Reload a converged solution of the current stage if one exists
     * and RESOLVE is off.
     *
     * @return true if the solution was reloaded (no solve needed)
     */
    bool reuse_previous_solution() {
        if (bconfig->control(RESOLVE) || !solution_exists())
            return false;
        if (rank == 0)
            std::cout << "Solved previously: " << bconfig->config_filename_abs()
                      << std::endl;
        reload();
        return true;
    }

    /// Solve the current stage unless a converged solution can be reused.
    int solve() {
        if (reuse_previous_solution())
            return EXIT_SUCCESS;
        setup_syst();
        return do_newton();
    }

    /**
     * @brief Newton iterations of the current system; writes checkpoints
     * `<stage>_ckpt_<n>` (CHECKPOINT) and the converged solution, then
     * releases the system.
     */
    int do_newton() override {
        bool endloop = false;
        int ite = 1;
        double conv = std::nan("1");
        if (rank == 0)
            print_diagnostics(ite - 1, conv);

        while (!endloop) {
            endloop = syst->do_newton(bconfig->seq_setting(PREC), conv);

            update_config_quantities();

            std::stringstream ss;
            ss << stagename << "_ckpt_" << ite - 1;
            bconfig->set_filename(ss.str());
            if (rank == 0) {
                print_diagnostics(ite, conv);
                std::cout << std::endl;
                if (bconfig->control(CHECKPOINT))
                    checkpoint();
            }

            // the domain extents may have changed
            update_coordinate_fields();

            ite++;
            check_max_iter_exceeded(*this, ite, conv);
        }

        bconfig->set_filename(converged_filename(stagename));
        if (rank == 0)
            checkpoint();
        // rank 0 moved its config to the output directory while saving
        bconfig->set_outputdir(outputdir);
        syst.reset(nullptr);
        MPI_Barrier(MPI_COMM_WORLD);
        return EXIT_SUCCESS;
    }

    /**
     * @brief Write the current solution regridded to `target` (its
     * resolution and domain settings) as `target`'s file. Called on rank 0
     * only; the solver itself is unchanged, so its state can still be kept
     * (e.g. in a Solution_snapshot) before the regridded file is loaded.
     *
     * @param target config of the regridded solution, filename and outputdir
     * set
     */
    virtual void write_regridded_solution(base_config_t& target) const = 0;

    /// Load the solution written by write_regridded_solution on every rank.
    void load_regridded_solution(std::string const& filename) {
        bconfig->set_outputdir(outputdir);
        bconfig->set_filename(filename);
        bconfig->open_config();
        reload();
    }

    /// Regrid to the resolution in the config: rank 0 writes `filename`,
    /// then every rank loads it.
    void regrid_to(std::string const& filename) {
        base_config_t target{*bconfig};
        target.set_outputdir(outputdir);
        target.set_filename(filename);
        if (rank == 0)
            write_regridded_solution(target);
        MPI_Barrier(MPI_COMM_WORLD);
        load_regridded_solution(filename);
    }

    /// Rebuild the support containers after the space or fields were
    /// replaced from outside, e.g. by Solution_snapshot::restore.
    void rebuild_support_containers() {
        reset_support_containers();
        ndom = space->get_nbr_domains();
        initialize_support_containers();
    }
};

/** @}*/
}  // namespace Kadath::FUKA_Solvers
