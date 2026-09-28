#include "FUKA_Solvers/utilities/solver_utilities.hpp"

namespace Kadath::FUKA_Solvers {
int BNS_XCTS_base::do_newton() {
    int exit_status = EXIT_SUCCESS;
    // parameters for the solver loop
    bool endloop = false;
    int ite = 1;
    double conv;
    if (rank == 0) {
        print_diagnostics(ite - 1, std::nan("1"));
    }
    // solve until convergence is achieved
    while (!endloop) {
        // do exactly one newton step, given the system above
        endloop =
            syst->do_newton(bconfig->seq_setting(SEQ_SETTINGS::PREC), conv);

        // update_config_quantities();
        // output files at this iteration and print diagnostics
        std::stringstream ss;
        ss << stagename << "_ckpt_" << ite - 1;

        if (logh_const) {
            // overwrite logh with scaled version for later use and output
            *logh = syst->give_val_def("H");
        }

        bconfig->set_filename(ss.str());
        if (rank == 0) {
            print_diagnostics(ite, conv);
            std::cout << std::endl;
            if (bconfig->control(CHECKPOINT))
                checkpoint();
        }

        // update all coordinate fields, in case the domain extents have changed
        Kadath::update_fields(*cfields,
                              *coord_vectors,
                              {},
                              xo,
                              xc1,
                              xc2,
                              syst.get());

        ite++;
        check_max_iter_exceeded(*this, ite, conv);
    }

    bconfig->set_filename(converged_filename(stagename));
    if (rank == 0) {
        checkpoint();
    }
    syst.reset(nullptr);
    MPI_Barrier(MPI_COMM_WORLD);
    return exit_status;
}
}  // namespace Kadath::FUKA_Solvers