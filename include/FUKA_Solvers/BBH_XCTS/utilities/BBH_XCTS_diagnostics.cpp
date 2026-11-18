#include "FUKA_Solvers/utilities/solver/format_settings.hpp"
#include "bco_utilities.hpp"

namespace Kadath::FUKA_Solvers {
inline void BBH_XCTS::print_diagnostics(const int ite,
                                        const double conv) const {
    using namespace ::Kadath::bco_utils;
    auto& cfg = *bconfig;
    auto integral = [&](char const* def, int const dom, int const bc) {
        Val_domain integrand(syst->give_val_def(def)()(dom));
        return space->get_domain(dom)->integ(integrand, bc);
    };

    double const Sm = integral("intSm", space->BH1 + 2, INNER_BC);
    double const Sp = integral("intSp", space->BH2 + 2, INNER_BC);
    double const Mirrmsq = integral("intMsq", space->BH1 + 2, INNER_BC);
    double const Mirrpsq = integral("intMsq", space->BH2 + 2, INNER_BC);
    // Christodoulou masses
    double const Mchm = std::sqrt(Mirrmsq + Sm * Sm / 4. / Mirrmsq);
    double const Mchp = std::sqrt(Mirrpsq + Sp * Sp / 4. / Mirrpsq);

    double const Px = integral("intPx", ndom - 1, OUTER_BC);
    double const Py = integral("intPy", ndom - 1, OUTER_BC);
    double const Pz = integral("intPz", ndom - 1, OUTER_BC);

    std::ios_base::fmtflags f(std::cout.flags());
    std::cout << std::string(42, '=') << std::endl
              << FORMAT << "Iter: " << ite << std::endl
              << FORMAT << "Error: " << conv << std::endl
              << FORMAT << "Omega: " << cfg(BIN_PARAMS::GOMEGA) << std::endl;
    if (!std::isnan(cfg.set(BIN_PARAMS::ADOT)))
        std::cout << FORMAT << "Adot: " << cfg(BIN_PARAMS::ADOT) << std::endl;
    std::cout << FORMAT << "Axis: " << cfg(BIN_PARAMS::COM) << std::endl
              << FORMAT << "P: " << "[" << Px << ", " << Py << ", " << Pz
              << "]\n\n";

    auto print_hole = [&](char const* name,
                          int const nucleus,
                          double const S,
                          double const Mirrsq,
                          double const Mch,
                          NODES const bco) {
        std::string const n{name};
        std::cout << FORMAT << (n + "-Mirr: ") << std::sqrt(Mirrsq) << std::endl
                  << FORMAT << (n + "-Mch: ") << Mch << std::endl
                  << FORMAT << (n + "-R: ")
                  << get_radius(space->get_domain(nucleus + 1), EQUI)
                  << std::endl
                  << FORMAT << (n + "-S: ") << S << std::endl
                  << FORMAT << (n + "-Chi: ") << S / Mch / Mch << std::endl
                  << FORMAT << (n + "-Omega: ") << cfg(BCO_PARAMS::OMEGA, bco)
                  << "\n\n";
    };
    print_hole("BH1", space->BH1, Sm, Mirrmsq, Mchm, NODES::BCO1);
    print_hole("BH2", space->BH2, Sp, Mirrpsq, Mchp, NODES::BCO2);
    std::cout.flags(f);
    std::cout << std::string(42, '=') << std::endl;
}
}  // namespace Kadath::FUKA_Solvers
