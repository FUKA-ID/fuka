namespace Kadath::FUKA_Solvers {

namespace BH_XCTS_detail {
/// Above this |chi| a new setup is first solved at a lower spin.
constexpr double spinup_chi_threshold = 0.5;
/// Intermediate spin of the spin-up for |chi| above it.
constexpr double spinup_chi_high = 0.8;
/// Above this |chi| solutions are not necessarily stable.
constexpr double stable_chi_limit = 0.84;

inline void require_solution_file(BH_XCTS_config_t& bconfig, int const rank) {
    std::string const spacein = bconfig.space_filename();
    if (fs::exists(spacein))
        return;
    std::stringstream ss;
    ss << "File: " << spacein << " not found"
       << ((rank == 0) ? "" : " for another rank") << ".\n";
    throw std::runtime_error(ss.str());
}

/**
 * @brief Settings of a stationary solve: store the final chi and, for a new
 * setup with |chi| > 0.5, start the spin-up at |chi| = 0.5.
 */
inline void prepare_stationary_solve(BH_XCTS_config_t& bconfig,
                                     int const rank) {
    double const chi = bconfig(BCO_PARAMS::CHI);
    if (rank == 0 && std::abs(chi) > stable_chi_limit)
        std::cerr << "Warning: Solutions above |chi| > " << stable_chi_limit
                  << " are not necessarily stable solutions\n";

    bconfig.control(CONTROLS::ITERATIVE_CHI) =
        bconfig.control(CONTROLS::SEQUENCES) &&
        std::abs(chi) > spinup_chi_threshold;
    bconfig.seq_setting(SEQ_SETTINGS::FINAL_CHI) = chi;
    if (bconfig.control(CONTROLS::ITERATIVE_CHI))
        bconfig(BCO_PARAMS::CHI) = std::copysign(spinup_chi_threshold, chi);
}

/**
 * @brief Stationary stages: TOTAL (if enabled), then TOTAL_BC, stepping chi
 * 0.5 -> 0.8 -> final during a spin-up.
 */
inline void solve_stationary_stages(BH_XCTS& solver,
                                    BH_XCTS_config_t& bconfig) {
    auto& stage_enabled = bconfig.return_stages();
    if (stage_enabled[STAGES::TOTAL]) {
        solver.set_stage(STAGES::TOTAL);
        solver.solve();
    }
    if (!stage_enabled[STAGES::TOTAL_BC])
        return;

    solver.set_stage(STAGES::TOTAL_BC);
    if (bconfig.control(CONTROLS::ITERATIVE_CHI)) {
        double const final_chi = bconfig.seq_setting(SEQ_SETTINGS::FINAL_CHI);
        while (bconfig.control(CONTROLS::ITERATIVE_CHI)) {
            solver.solve();
            stage_enabled[STAGES::TOTAL_BC] = true;
            if (std::abs(final_chi) <= spinup_chi_high ||
                std::abs(bconfig(BCO_PARAMS::CHI)) >= spinup_chi_high)
                bconfig.control(CONTROLS::ITERATIVE_CHI) = false;
            else
                bconfig(BCO_PARAMS::CHI) =
                    std::copysign(spinup_chi_high, final_chi);
        }
        bconfig(BCO_PARAMS::CHI) = final_chi;
    }
    solver.solve();
}

inline std::string absolute_outputdir(std::string outputdir, int const rank) {
    if (outputdir == "./")
        outputdir = fs::current_path().string();
    if (rank == 0)
        std::cout << "Solutions will be stored in: " << outputdir << "\n"
                  << "Directory will be created if it doesn't exist.\n";
    fs::create_directory(outputdir);
    return outputdir;
}
}  // namespace BH_XCTS_detail

inline int BH_XCTS_driver(BH_XCTS_config_t& bconfig,
                          Parameter_sequence<BCO_PARAMS>& resolution,
                          std::string outputdir) {
    using namespace BH_XCTS_detail;
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    require_solution_file(bconfig, rank);

    // the stationary solution is computed without the linear boost
    double const xboost = bconfig.set(BCO_PARAMS::BVELX);
    double const yboost = bconfig.set(BCO_PARAMS::BVELY);
    bconfig.set(BCO_PARAMS::BVELX) = 0.;
    bconfig.set(BCO_PARAMS::BVELY) = 0.;

    auto& stage_enabled = bconfig.return_stages();
    auto [last_stage, last_stage_idx] = get_last_enabled(MSTAGE, stage_enabled);
    bool const linear_boost = stage_enabled[STAGES::LINBOOST];

    prepare_stationary_solve(bconfig, rank);
    outputdir = absolute_outputdir(outputdir, rank);
    if (rank == 0)
        std::cout << "Config File: " << bconfig.config_filename_abs()
                  << std::endl
                  << "Fields File: " << bconfig.space_filename() << std::endl
                  << bconfig << std::endl;

    // load from the input location, write to outputdir
    BH_XCTS solver(&bconfig, resolution, outputdir, rank);
    bconfig.set_outputdir(outputdir);

    auto solve_at_current_resolution = [&]() {
        if (linear_boost) {
            bconfig.set(BCO_PARAMS::BVELX) = xboost;
            bconfig.set(BCO_PARAMS::BVELY) = yboost;
            solver.set_stage(STAGES::LINBOOST);
            solver.solve();
        } else {
            solve_stationary_stages(solver, bconfig);
        }
    };

    solve_stationary_stages(solver, bconfig);
    if (linear_boost)
        solve_at_current_resolution();

    while (solver.increment_resolution()) {
        if (rank == 0)
            std::cout << "Increasing resolution to: "
                      << bconfig(BCO_PARAMS::BCO_RES) << "\n";
        solver.regrid();
        // no new setup: no spin-up
        bconfig.control(CONTROLS::SEQUENCES) = false;
        stage_enabled[last_stage_idx] = true;
        if (!linear_boost)
            prepare_stationary_solve(bconfig, rank);
        solve_at_current_resolution();
    }
    MPI_Barrier(MPI_COMM_WORLD);
    return EXIT_SUCCESS;
}

template <class Seq_t>
BH_XCTS_config_t BH_XCTS_sequence(
    BH_XCTS_config_t& seqconfig,
    Seq_t const& seq,
    Parameter_sequence<BCO_PARAMS> const& resolution,
    std::string outputdir) {
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    BH_XCTS_config_t base_config =
        generate_sequence_config(seqconfig, outputdir);
    if (rank == 0)
        base_config.write_config();
    base_config.set(resolution.get_indices()) = resolution.init();

    BH_XCTS_config_t bconfig{};
    auto single_solution = [&](auto const val) {
        bconfig = base_config;
        if (seq.is_set())
            bconfig.set(seq.get_indices()) = val;

        if (bconfig.control(CONTROLS::SEQUENCES)) {
            if (rank == 0)
                setup_3d_BH_xcts(bconfig);
            MPI_Barrier(MPI_COMM_WORLD);
            // every rank reads the new setup
            bconfig.open_config();
            MPI_Barrier(MPI_COMM_WORLD);
        }
        auto res{resolution};
        BH_XCTS_driver(bconfig, res, outputdir);
    };

    if (seq.is_set()) {
        auto const& dx = seq.step_size();
        for (auto val = seq.init(); seq.loop_condition(val); val += dx)
            single_solution(val);
    } else {
        single_solution(0);
    }
    return bconfig;
}

inline int BH_XCTS_binary_boost_driver(
    BH_XCTS_config_t& bconfig,
    Parameter_sequence<BCO_PARAMS> const& resolution,
    std::string outputdir,
    kadath_config_boost<BIN_INFO>& binconfig) {
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int const final_res = bconfig(BCO_PARAMS::BCO_RES);
    bool const res_inc = bconfig.seq_setting(SEQ_SETTINGS::INIT_RES) <
                         final_res;
    bconfig.set(BCO_PARAMS::BCO_RES) = resolution.init();

    // stationary solution at the initial resolution
    Parameter_sequence<BCO_PARAMS> init_res{resolution};
    init_res.set(resolution.init(), resolution.init(), resolution.init());
    bconfig = BH_XCTS_sequence(bconfig,
                               Parameter_sequence<BCO_PARAMS>{},
                               init_res,
                               outputdir);

    if (rank == 0)
        std::cout << "Config File: " << bconfig.config_filename_abs()
                  << std::endl
                  << "Fields File: " << bconfig.space_filename() << std::endl
                  << bconfig << std::endl;
    Boosted_BH_XCTS solver(&bconfig, resolution, outputdir, &binconfig, rank);
    solver.solve();

    if (res_inc) {
        bconfig.set(BCO_PARAMS::BCO_RES) = final_res;
        solver.regrid();
        solver.solve();
    }
    MPI_Barrier(MPI_COMM_WORLD);
    return EXIT_SUCCESS;
}

}  // namespace Kadath::FUKA_Solvers
