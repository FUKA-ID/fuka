#include "BNS_XCTS.hpp"
#include "EOS/FUKA_EOS_Utilities.hh"
#include "FUKA_Solvers/utilities/binary/binary_driver.hpp"
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
struct launch_BNS_XCTS_solver {
    template <class config_t>
    int operator()(const int rank,
                   config_t& bconfig,
                   Parameter_sequence<BIN_PARAMS> const& resolution,
                   bco_sequence<BCO_PARAMS, NODES> const& ns1_seq,
                   bco_sequence<BCO_PARAMS, NODES> const& ns2_seq,
                   std::string outputdir) {

        require_solution_file(bconfig, rank);
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
                solver.solve();

                if (!ignore_resinc) {
                    // Make sure final solution uses optimal domain decomposition
                    solver.regrid();

                    // resolve at current resolution
                    solver.solve();
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
                    exit_status = solver.solve();
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
        EOS_Function_Dispatcher::dispatch<launch_BNS_XCTS_solver>(bconfig,
                                                                  eos_type,
                                                                  rank,
                                                                  bconfig,
                                                                  resolution,
                                                                  ns2_seq,
                                                                  ns1_seq,
                                                                  outputdir);

    return exit_status;
}

template <class Seq_t, class config_t>
inline int BNS_XCTS_sequence(config_t& seqconfig,
                             Seq_t const& seq,
                             Parameter_sequence<BIN_PARAMS> const& resolution,
                             bco_sequence<BCO_PARAMS, NODES> const& ns2_seq,
                             bco_sequence<BCO_PARAMS, NODES> const& ns1_seq,
                             std::string outputdir) {
    return binary_sequence(
        seqconfig,
        seq,
        resolution,
        outputdir,
        {"ns", "ns"},
        {NODES::BCO1, NODES::BCO2},
        [](config_t& c) { BNS_XCTS_setup_bin_config(c); },
        [](config_t& c) { BNS_XCTS_setup_space(c); },
        [&](config_t& c) {
            return BNS_XCTS_solution_driver(c,
                                            resolution,
                                            ns2_seq,
                                            ns1_seq,
                                            outputdir);
        });
}

/** @}*/
}  // namespace FUKA_Solvers
}  // namespace Kadath
