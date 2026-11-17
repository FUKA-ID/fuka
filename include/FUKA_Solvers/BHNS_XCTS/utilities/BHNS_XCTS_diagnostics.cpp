#include "FUKA_Solvers/utilities/solver/format_settings.hpp"
#include "bco_utilities.hpp"

namespace Kadath::FUKA_Solvers {
inline void BHNS_XCTS_base::print_diagnostics(const int ite,
                                              const double conv) const {
    using namespace ::Kadath::bco_utils;
    auto& cfg = *bconfig;
    auto surface_integral = [&](char const* def, int const dom, int const bc) {
        Val_domain integrand(syst->give_val_def(def)()(dom));
        return space->get_domain(dom)->integ(integrand, bc);
    };

    double baryonic_massNS = 0.;
    double ql_massNS = 0.;
    for (int d = space->NS; d <= space->ADAPTEDNS; ++d) {
        baryonic_massNS += syst->give_val_def("intMb")()(d).integ_volume();
        ql_massNS += syst->give_val_def("intM")()(d).integ_volume();
    }

    auto const [rmin, rmax] = get_rmin_rmax(*space, space->ADAPTEDNS);
    double const r_bh = get_radius(space->get_domain(space->ADAPTEDBH), EQUI);

    double const mirrsq = surface_integral("intMsq", space->BH + 2, INNER_BC);
    double const Mirr = std::sqrt(mirrsq);

    double const S1 = surface_integral("intS1", space->ADAPTEDNS + 1, OUTER_BC);
    double const chi1 = S1 / cfg(BCO_PARAMS::MADM, NODES::BCO1) /
                        cfg(BCO_PARAMS::MADM, NODES::BCO1);
    double const chiql1 = S1 / ql_massNS / ql_massNS;

    double const S2 = surface_integral("intS2", space->ADAPTEDBH + 1, OUTER_BC);
    // Christodoulou mass
    double const Mch = std::sqrt(mirrsq + S2 * S2 / 4. / mirrsq);
    double const chi2 = S2 / Mch / Mch;

    double const Px = surface_integral("intPx", ndom - 1, OUTER_BC);
    double const Py = surface_integral("intPy", ndom - 1, OUTER_BC);
    double const Pz = surface_integral("intPz", ndom - 1, OUTER_BC);

    std::ios_base::fmtflags f(std::cout.flags());
    std::cout << "=======================================" << std::endl
              << FORMAT << "Iter: " << ite << std::endl
              << FORMAT << "Error: " << conv << "\n"
              << FORMAT << "Omega: " << cfg(BIN_PARAMS::GOMEGA) << std::endl
              << FORMAT << "Axis: " << cfg(BIN_PARAMS::COM) << std::endl
              << FORMAT << "Padm: " << "[" << Px << ", " << Py << ", " << Pz
              << "]\n\n";

    std::cout << FORMAT << "NS-Mb: " << baryonic_massNS << "["
              << cfg(BCO_PARAMS::MB, NODES::BCO1) << "]\n"
              << FORMAT << "NS-Madm_ql: " << ql_massNS << "["
              << cfg(BCO_PARAMS::MADM, NODES::BCO1) << "]\n"
              << FORMAT << "NS-R: " << rmin << " " << rmax << std::endl
              << FORMAT << "NS-S: " << S1 << std::endl
              << FORMAT << "NS-Chi: " << chi1 << "["
              << cfg(BCO_PARAMS::CHI, NODES::BCO1) << "]\n"
              << FORMAT << "NS-qlChi: " << chiql1 << std::endl
              << FORMAT << "NS-Omega: " << cfg(BCO_PARAMS::OMEGA, NODES::BCO1)
              << "\n\n";

    std::cout << FORMAT << "BH-Mirr: " << Mirr << "["
              << cfg(BCO_PARAMS::MIRR, NODES::BCO2) << "]\n"
              << FORMAT << "BH-Mch: " << Mch << "["
              << cfg(BCO_PARAMS::MCH, NODES::BCO2) << "]\n"
              << FORMAT << "BH-R: " << r_bh << std::endl
              << FORMAT << "BH-S: " << S2 << std::endl
              << FORMAT << "BH-Chi: " << chi2 << "["
              << cfg(BCO_PARAMS::CHI, NODES::BCO2) << "]\n"
              << FORMAT << "BH-Omega: " << cfg(BCO_PARAMS::OMEGA, NODES::BCO2)
              << std::endl;
    std::cout.flags(f);
    std::cout << "=======================================" << std::endl;
}
}  // namespace Kadath::FUKA_Solvers
