/**
 * @file test_solution_snapshot.cpp
 * @brief Solution_snapshot on the isolated BH solver: take the solution,
 * load a solution regridded to a higher resolution, restore, and check that
 * the original space, fields and config are back and a system builds on them.
 *
 * usage: mpirun -np <n> test_solution_snapshot [scratch directory]
 */
#include <mpi.h>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>

#include "FUKA_Solvers/BHNS_XCTS/BHNS_XCTS.hpp"
#include "FUKA_Solvers/BH_XCTS/BH_XCTS_driver.hpp"
#include "FUKA_Solvers/BNS_XCTS/BNS_XCTS.hpp"
#include "FUKA_Solvers/NS_XCTS/NS_XCTS_driver.hpp"
#include "FUKA_Solvers/utilities/solver/solution_snapshot.hpp"

using namespace Kadath;
using namespace Kadath::FUKA_Config;
using namespace Kadath::FUKA_Solvers;

namespace {
// compile check of the NS, BNS and BHNS member lists (moved unique_ptr, copied config)
using table_eos_t =
    FUKA_EOS::FUKA_EOS_Wrapper<FUKA_EOS::fuka_eos_t, FUKA_EOS::margherita_1d>;
}  // namespace
template class Kadath::FUKA_Solvers::Solution_snapshot<
    NS_XCTS_DIFF_ROT<table_eos_t>>;
template class Kadath::FUKA_Solvers::Solution_snapshot<
    Boosted_NS_XCTS<table_eos_t>>;
template class Kadath::FUKA_Solvers::Solution_snapshot<BNS_XCTS<table_eos_t>>;
template class Kadath::FUKA_Solvers::Solution_snapshot<BHNS_XCTS<table_eos_t>>;

namespace {
int failures = 0;

void check(bool const ok, std::string const& what, int const rank) {
    if (!ok)
        ++failures;
    if (rank == 0)
        std::cout << (ok ? "PASS " : "FAIL ") << what << std::endl;
}

bool same_value(double const a, double const b) {
    return std::abs(a - b) <= 1e-15 * std::abs(b);
}

int order(double const config_value) {
    return static_cast<int>(std::lround(config_value));
}

int radial_order(BH_XCTS const& solver) {
    return solver.get_space()->get_domain(0)->get_nbr_points()(0);
}
}  // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    std::string const dir = (argc > 1) ? argv[1] : fs::current_path().string();

    using config_t = BH_XCTS_config_t;
    config_t bconfig;
    bconfig.set_defaults();
    bconfig.set_outputdir(dir);
    bconfig.set_filename("snapshot_setup");
    if (rank == 0)
        setup_3d_BH_xcts(bconfig);
    MPI_Barrier(MPI_COMM_WORLD);
    bconfig.open_config();

    Parameter_sequence<BCO_PARAMS> resolution("res", BCO_PARAMS::BCO_RES);
    resolution.set(9, 9, 11);
    BH_XCTS solver(&bconfig, resolution, dir, rank);

    Point probe(3);
    probe.set(1) = 1.1;
    probe.set(2) = 0.3;
    probe.set(3) = -0.2;
    auto const* space_before = solver.get_space().get();
    double const conf_before = solver.get_conformal_factor()->val_point(probe);
    double const rin_before = bconfig(BCO_PARAMS::RIN);
    std::string const file_before = bconfig.config_filename();

    Solution_snapshot<BH_XCTS> snapshot;
    check(snapshot.empty(), "new snapshot is empty", rank);

    config_t target{bconfig};
    target.set(BCO_PARAMS::BCO_RES) = 11;
    target.set_outputdir(dir);
    target.set_filename("snapshot_regrid");
    if (rank == 0)
        solver.write_regridded_solution(target);
    MPI_Barrier(MPI_COMM_WORLD);

    snapshot.take(solver);
    check(!snapshot.empty() && !solver.get_space() &&
              !solver.get_conformal_factor(),
          "take moves the space and fields out",
          rank);

    solver.load_regridded_solution("snapshot_regrid");
    check(radial_order(solver) == 11 &&
              order(bconfig(BCO_PARAMS::BCO_RES)) == 11,
          "regridded solution loaded at N_r = 11",
          rank);

    solver.setup_syst();
    bool threw = false;
    try {
        Solution_snapshot<BH_XCTS> other;
        other.take(solver);
    } catch (std::logic_error const&) {
        threw = true;
    }
    check(threw, "take refuses a solver that holds a system", rank);
    solver.get_syst().reset(nullptr);

    snapshot.restore(solver);
    check(snapshot.empty(), "restore empties the snapshot", rank);
    check(solver.get_space().get() == space_before && radial_order(solver) == 9,
          "restore brings back the original space (same object, N_r = 9)",
          rank);
    double const conf_after = solver.get_conformal_factor()->val_point(probe);
    check(same_value(conf_after, conf_before),
          "restore brings back the fields",
          rank);
    check(order(bconfig(BCO_PARAMS::BCO_RES)) == 9 &&
              same_value(bconfig(BCO_PARAMS::RIN), rin_before) &&
              bconfig.config_filename() == file_before,
          "restore brings back the config values",
          rank);

    solver.setup_syst();
    check(solver.get_syst() != nullptr,
          "the system builds on the restored solution",
          rank);
    solver.get_syst().reset(nullptr);

    if (rank == 0)
        std::cout << (failures == 0 ? "ALL PASS" : "FAILURES") << std::endl;
    MPI_Finalize();
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
