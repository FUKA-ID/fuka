#include "FUKA_Solvers/utilities/format_settings.hpp"
#include "bco_utilities.hpp"

namespace Kadath::FUKA_Solvers {
inline void BNS_XCTS_base::print_diagnostics(const int ite, const double conv) {
    std::ios_base::fmtflags f(std::cout.flags());
    using namespace ::Kadath::bco_utils;

    double baryonic_mass1 = 0.;
    double ql_mass1 = 0.;
    for (int d = space->NS1; d <= space->ADAPTED1; ++d) {
        baryonic_mass1 += syst->give_val_def("intMb")()(d).integ_volume();
        ql_mass1 += syst->give_val_def("intM")()(d).integ_volume();
    }

    double baryonic_mass2 = 0.;
    double ql_mass2 = 0.;
    for (int d = space->NS2; d <= space->ADAPTED2; ++d) {
        baryonic_mass2 += syst->give_val_def("intMb")()(d).integ_volume();
        ql_mass2 += syst->give_val_def("intM")()(d).integ_volume();
    }

    // surface integration of the ADM linear momentum at infinity
    Val_domain integPx(syst->give_val_def("intPx")()(ndom - 1));
    double Px = space->get_domain(ndom - 1)->integ(integPx, OUTER_BC);

    Val_domain integPy(syst->give_val_def("intPy")()(ndom - 1));
    double Py = space->get_domain(ndom - 1)->integ(integPy, OUTER_BC);

    Val_domain integPz(syst->give_val_def("intPz")()(ndom - 1));
    double Pz = space->get_domain(ndom - 1)->integ(integPz, OUTER_BC);

    // surface integration of the quasi-local (dimensionless) spin
    // angular momentum of both stars
    Val_domain integS1(syst->give_val_def("intS1")()(space->ADAPTED1 + 1));
    double S1 =
        space->get_domain(space->ADAPTED1 + 1)->integ(integS1, OUTER_BC);
    double chi1 = S1 / (*bconfig)(MADM, BCO1) / (*bconfig)(MADM, BCO1);
    double chiql1 = S1 / ql_mass1 / ql_mass1;

    Val_domain integS2(syst->give_val_def("intS2")()(space->ADAPTED2 + 1));
    double S2 =
        space->get_domain(space->ADAPTED2 + 1)->integ(integS2, OUTER_BC);
    double chi2 = S2 / (*bconfig)(MADM, BCO2) / (*bconfig)(MADM, BCO2);
    double chiql2 = S2 / ql_mass2 / ql_mass2;

    // print mininmal and maximal radius of the surface adapted domains
    auto print_rs = [&](int dom) {
        auto rs = get_rmin_rmax(*space, dom);
        std::cout << FORMAT << "NS-Rs: " << rs[0] << " " << rs[1] << std::endl;
    };

#ifndef FORMAT
#define FORMAT std::setw(13) << std::left << std::showpos
#endif

    std::cout << "=======================================" << endl;
    std::cout << FORMAT << "Iter: " << ite << std::endl
              << FORMAT << "Error: " << conv << "\n";
    std::cout << FORMAT << "Omega: " << (*bconfig)(GOMEGA) << endl;
    std::cout << FORMAT << "Axis: " << (*bconfig)(COM) << endl;
    std::cout << FORMAT << "Padm: " << "[" << Px << ", " << Py << ", " << Pz
              << "]\n\n";

    // baryonic mass
    std::cout
        << FORMAT << "NS1-Mb: " << baryonic_mass1
        << std::endl
        // quasi-local ADM mass
        << FORMAT << "NS1-Madm_ql: " << ql_mass1
        << std::endl
        // quasi-local spin angular momentum
        << FORMAT << "NS1-S: " << S1
        << std::endl
        // quasi-local dimensionless spin
        << FORMAT << "NS1-Chi: " << chi1
        << std::endl
        // quasi-local dimensionless spin, normalized by the quasi-local ADM mass
        << FORMAT << "NS1-Chi_ql: " << chiql1
        << std::endl
        // angular frequency paramter of the stellar rotation
        << FORMAT << "NS1-OmegaS: " << (*bconfig)(OMEGA, BCO1) << "\n";
    print_rs(space->ADAPTED1);

    std::cout << "\n"
              << FORMAT << "NS2-Mb: " << baryonic_mass2 << std::endl
              << FORMAT << "NS2-Madm_ql: " << ql_mass2 << std::endl
              << FORMAT << "NS2-S: " << S2 << std::endl
              << FORMAT << "NS2-Chi: " << chi2 << std::endl
              << FORMAT << "NS2-Chi_ql: " << chiql2 << std::endl
              << FORMAT << "NS2-OmegaS: " << (*bconfig)(OMEGA, BCO2)
              << std::endl;
    print_rs(space->ADAPTED2);
    std::cout.flags(f);
    std::cout << "=======================================" << endl;
}  // end print_diagnostics
}  // namespace Kadath::FUKA_Solvers