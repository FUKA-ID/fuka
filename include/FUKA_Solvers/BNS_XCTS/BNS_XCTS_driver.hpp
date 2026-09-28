#include "BNS_XCTS.hpp"
#include "EOS/FUKA_EOS_Utilities.hh"
#include "Solvers/sequences/bco_sequence.hpp"
#include "utilities/BNS_XCTS_setup.hpp"

#if defined __cpp_lib_filesystem && __cpp_lib_filesystem < 201703L
#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;
#else
#include <filesystem>
namespace fs = std::filesystem;
#endif

/**
 * \addtogroup BNS_XCTS
 * \ingroup FUKA
 * @{*/
namespace Kadath {
namespace FUKA_Solvers {

template <class eos_t>
struct launch_BHNS_solver {
    template <class config_t>
    int operator()(const int rank,
                   config_t& bconfig,
                   Parameter_sequence<BIN_PARAMS> const& resolution,
                   bco_sequence<BCO_PARAMS, NODES> const& ns1_seq,
                   bco_sequence<BCO_PARAMS, NODES> const& ns2_seq,
                   std::string outputdir) {

        std::string spacein = bconfig.space_filename();

        if (!fs::exists(spacein)) {
            // mainly for debugging MPI bugs
            if (rank == 0) {
                std::cerr << "File: " << spacein << " not found.\n\n";
            } else {
                std::cerr << "File: " << spacein
                          << " not found for another rank.\n\n";
            }
            std::_Exit(EXIT_FAILURE);
        }

        // just so you really know
        if (rank == 0) {
            std::cout << "Config File: " << bconfig.config_filename_abs()
                      << std::endl
                      << "Fields File: " << spacein << std::endl
                      << bconfig << std::endl;
        }
        FILE* ff1 = fopen(spacein.c_str(), "r");
        if (ff1 == NULL) {
            // mainly for debugging MPI bugs
            std::stringstream ss;
            ss << spacein.c_str() << " failed to open for rank " << rank
               << "\n";
            throw std::runtime_error(ss.str().c_str());
        }
        fclose(ff1);
        int exit_status = EXIT_SUCCESS;
        BNS_XCTS<eos_t> solver(&bconfig,
                               ns1_seq,
                               ns2_seq,
                               resolution,
                               outputdir,
                               rank);

        auto launch = [](auto& solver,
                         bool ignore_resinc = false,
                         bool ignore_seq = false) {
            int exit_status = EXIT_SUCCESS;
            do {
                // initial solution
                solver.setup_syst();
                solver.do_newton();

                if (!ignore_resinc) {
                    // Make sure final solution uses optimal domain decomposition
                    solver.regrid();

                    // resolve at current resolution
                    solver.setup_syst();
                    solver.do_newton();
                }

                // Obtain final resolution for the desired
                // solution or the first solution in a sequence
                // All remaining sequences will be computed
                // at the final resolution only
                // Note: only occurs if the last stage is NOROT_BC
                while (!ignore_resinc && solver.increment_resolution() &&
                       (exit_status != EXIT_FAILURE)) {
                    // regrid to new resolution
                    solver.regrid();

                    // initial solution
                    solver.setup_syst();
                    exit_status = solver.do_newton();
                }
            } while (!ignore_seq);  // && solver.increment_seq());
            return exit_status;
        };

        if (bconfig.set_stage(STAGES::PRE)) {
            solver.set_solver_stage() = STAGES::PRE;
            solver.set_stagename() = "PRE";
            launch(solver, true, true);
        }

        if (bconfig.set_stage(STAGES::TOTAL)) {
            solver.set_solver_stage() = STAGES::TOTAL;
            solver.set_stagename() = "TOTAL";
            launch(solver, false, true);
        }

        if (bconfig.set_stage(STAGES::TOTAL_BC)) {
            solver.set_solver_stage() = STAGES::TOTAL_BC;
            solver.set_stagename() = "TOTAL_BC";
            launch(solver, false, true);
        }

        if (bconfig.set_stage(STAGES::ECC_RED)) {
            solver.set_solver_stage() = STAGES::ECC_RED;
            solver.set_stagename() = "ECC_RED";
            launch(solver, false, true);
        }
        return exit_status;
    };
};

template <class config_t>
int BNS_XCTS_solution_driver(config_t& bconfig,
                             Parameter_sequence<BIN_PARAMS> const& resolution,
                             bco_sequence<BCO_PARAMS, NODES> const& ns2_seq,
                             bco_sequence<BCO_PARAMS, NODES> const& ns1_seq,
                             std::string outputdir) {
    int exit_status = 0;
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    using namespace Kadath::FUKA_EOS;

    const std::string eos_type =
        bconfig.template eos<std::string>(EOS_PARAMS::EOSTYPE, NODES::BCO1);

    exit_status =
        EOS_Function_Dispatcher::dispatch<launch_BHNS_solver>(bconfig,
                                                              eos_type,
                                                              rank,
                                                              bconfig,
                                                              resolution,
                                                              ns2_seq,
                                                              ns1_seq,
                                                              outputdir);

    return exit_status;
}

template <class config_t>
config_t BNS_XCTS_sequence_setup(config_t& seqconfig, std::string outputdir) {
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    config_t bconfig =
        binary_generate_sequence_config(seqconfig, {"ns", "ns"}, outputdir);

    update_eos_parameters(seqconfig, bconfig, NODES::BCO1);
    update_eos_parameters(seqconfig, bconfig, NODES::BCO2);

    if (rank == 0)
        bconfig.write_config();
    return bconfig;
}

template <class Seq_t, class config_t>
inline int BNS_XCTS_sequence(config_t& seqconfig,
                             Seq_t const& seq,
                             Parameter_sequence<BIN_PARAMS> const& resolution,
                             bco_sequence<BCO_PARAMS, NODES> const& ns2_seq,
                             bco_sequence<BCO_PARAMS, NODES> const& ns1_seq,
                             std::string outputdir) {
    int rank = 0, exit_status = EXIT_SUCCESS;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    // Initialize sequence variables
    auto sequence_var_indices = seq.get_indices();
    auto resolution_indices = resolution.get_indices();

    auto const& dx = seq.step_size();

    // Initialize full configurator
    config_t base_config = BNS_XCTS_sequence_setup(seqconfig, outputdir);
    base_config.set(resolution_indices) = resolution.init();

#ifdef DEBUG
    if (rank == 0) {
        std::cout << seq << std::endl;
        std::cout << resolution << std::endl;
    }
#endif

    auto single_seq = [&](auto val) {
        config_t bconfig = base_config;
        if (seq.is_set())
            bconfig.set(sequence_var_indices) = val;

        if (bconfig.control(CONTROLS::SEQUENCES)) {
            // Retain adot in case it is set from the start manually
            auto const adot = bconfig.set(BIN_PARAMS::ADOT);

            // Adot is deleted here
            BNS_XCTS_setup_bin_config(bconfig);

            // Reset ADOT
            bconfig.set(BIN_PARAMS::ADOT) = adot;

            // Generate initial guess
            // FIXME Needs to be updated for DR
            BNS_XCTS_setup_space(bconfig);
        }

        exit_status = BNS_XCTS_solution_driver(bconfig,
                                               resolution,
                                               ns2_seq,
                                               ns1_seq,
                                               outputdir);
        return exit_status;
    };

    // Loop if a valid sequence is set
    if (seq.is_set()) {
        for (auto val = seq.init(); seq.loop_condition(val); val += dx) {
            exit_status = single_seq(val);
        }
    } else {
        exit_status = single_seq(0);
    }
    return exit_status;
}

/** @}*/
}  // namespace FUKA_Solvers
}  // namespace Kadath
