#include <cmath>
#include <iomanip>
#include <sstream>

#include "FUKA_Solvers/utilities/solver/format_settings.hpp"
#include "bco_utilities.hpp"

namespace Kadath::FUKA_Solvers {

inline BH_XCTS_base::BH_XCTS_base(base_config_t* config_,
                                  res_seq_t const& res_,
                                  std::string const& outputdir_,
                                  int const rank_)
    : solver_base_t(config_, outputdir_, rank_),
      resolution(std::make_unique<res_seq_t>(res_)) {}

inline void BH_XCTS_base::initialize_support_containers() {
    basis = std::make_unique<Base_tensor>(shift->get_basis());
    fmet = std::make_unique<Metric_flat>(*space, *basis);
    cfields = std::make_unique<cfgen_t>(*space);
    coord_vectors = std::make_unique<cfary_t>(default_co_vector_ary(*space));
    update_fields_co(*cfields, *coord_vectors, {}, 0.);
}

inline void BH_XCTS_base::save_solution_to_file() const {
    bconfig->set_outputdir(outputdir);
    Kadath::bco_utils::save_to_file(*space,
                                    *bconfig,
                                    *conformal_factor,
                                    *lapse,
                                    *shift);
}

inline void BH_XCTS_base::load_solution_from_file() {
    std::string const spacein{bconfig->space_filename()};
    FILE* ff1 = fopen(spacein.c_str(), "r");
    if (ff1 == nullptr)
        throw std::runtime_error("BH_XCTS: cannot open " + spacein);

    space = std::make_unique<base_space_t>(ff1);
    conformal_factor = std::make_unique<Scalar>(*space, ff1);
    lapse = std::make_unique<Scalar>(*space, ff1);
    shift = std::make_unique<Vector>(*space, ff1);
    fclose(ff1);

    ndom = space->get_nbr_domains();
}

inline void BH_XCTS_base::reset_all_ptrs() {
    reset_support_containers();
    conformal_factor.reset(nullptr);
    lapse.reset(nullptr);
    shift.reset(nullptr);
    space.reset(nullptr);
}

inline std::string BH_XCTS_base::converged_filename(
    const std::string stage) const {
    auto const res = space->get_domain(0)->get_nbr_points()(0);
    std::stringstream ss;
    ss << "BH";
    if (stage != "")
        ss << "_" << stage << ".";
    else
        ss << ".";
    ss << (*bconfig)(MCH) << "." << (*bconfig)(CHI) << "." << std::setfill('0')
       << std::setw(1) << (*bconfig)(NSHELLS) << "." << std::setfill('0')
       << std::setw(2) << res;
    return ss.str();
}

inline void BH_XCTS_base::syst_init() {
    // the (flat) conformal metric
    fmet->set_system(*syst, "f");

    syst->add_cst("4piG", (*bconfig)(BCO_QPIG));
    syst->add_cst("PI", M_PI);
    syst->add_cst("M", (*bconfig)(MIRR));
    syst->add_cst("CM", (*bconfig)(MCH));
    syst->add_cst("chi", (*bconfig)(CHI));
    syst->add_var("ome", (*bconfig)(OMEGA));

    syst->add_cst("mg", *(*coord_vectors)[GLOBAL_ROT]);
    syst->add_cst("sm", *(*coord_vectors)[S_BCO1]);
    syst->add_cst("ex", *(*coord_vectors)[EX]);
    syst->add_cst("ey", *(*coord_vectors)[EY]);
    syst->add_cst("einf", *(*coord_vectors)[S_INF]);

    syst->add_var("P", *conformal_factor);
    syst->add_var("N", *lapse);
    syst->add_var("bet", *shift);

    syst->add_def("NP = P*N");
    syst->add_def("Ntilde = N / P^6");
    syst->add_def(
        "A^ij = (D^i bet^j + D^j bet^i - 2. / 3.* D_k bet^k * f^ij) / 2. / "
        "Ntilde");

    // integrand of the irreducible mass squared on the excision surface
    syst->add_def("intMsq= P^4 / 16. / PI");

    // ADM and Komar masses at infinity
    syst->add_def(ndom - 1, "intMadm = - einf^i * D_i P / 4piG * 2");
    syst->add_def(ndom - 1, "intMk = einf^i * D_i N / 4piG");
}

inline void BH_XCTS_base::add_constraint_definitions() {
    // quasi-local spin on the excision surface
    syst->add_def("intS  = A_ij * mg^i * sm^j / 8. / PI");

    // Hamiltonian and momentum constraints and the trace of the evolution
    // equation of K (K = 0, conformally flat, stationary)
    syst->add_def("eqP = D^i D_i P + A_ij * A^ij / P^7 / 8");
    syst->add_def("eqNP = D^i D_i NP - 7. / 8. * NP / P^8 * A_ij * A^ij");
    syst->add_def(
        "eqbet^i = D_j D^j bet^i + D^i D_j bet^j / 3. - 2. * A^ij * D_j "
        "Ntilde");
}

inline void BH_XCTS_base::add_equations(std::string const& lapse_bc,
                                        std::string const& shift_inf_bc,
                                        std::string const& shift_bh_bc) {
    space->add_eq(*syst, "eqNP = 0", "N", "dn(N)");
    space->add_eq(*syst, "eqP = 0", "P", "dn(P)");
    space->add_eq(*syst, "eqbet^i = 0", "bet^i", "dn(bet^i)");

    space->add_bc_inf(*syst, "NP = 1");
    space->add_bc_inf(*syst, "P = 1");
    space->add_bc_inf(*syst, shift_inf_bc.c_str());

    // apparent horizon conditions on the excision surface
    space->add_bc_bh(*syst, lapse_bc.c_str());
    space->add_bc_bh(
        *syst,
        "sm^j * D_j P + P / 4 * D^j sm_j + A_ij * sm^i * sm^j / P^3 / 4 = 0");
    space->add_bc_bh(*syst, shift_bh_bc.c_str());

    // fix the spin by chi and the horizon area by the irreducible mass
    space->add_eq_int_bh(*syst, "integ(intS) - chi * CM * CM = 0 ");
    space->add_eq_int_bh(*syst, "integ(intMsq) - M * M = 0 ");

    for (int d : excluded_doms) {
        syst->add_eq_full(d, "N = 0");
        syst->add_eq_full(d, "P = 0");
        syst->add_eq_full(d, "bet^i = 0");
    }
}

inline void BH_XCTS_base::update_config_quantities() {
    bconfig->set(RMID) =
        Kadath::bco_utils::get_radius(space->get_domain(1), OUTER_BC);
    // the converged von Neumann lapse is the fixed lapse of a later TOTAL stage
    if (solver_stage == STAGES::TOTAL_BC || solver_stage == STAGES::LINBOOST)
        bconfig->set(FIXED_LAPSE) =
            Kadath::bco_utils::get_boundary_val(2, *lapse, INNER_BC);
}

inline void BH_XCTS_base::print_diagnostics(const int ite,
                                            const double conv) const {
    double const r =
        Kadath::bco_utils::get_radius(space->get_domain(1), OUTER_BC);

    Val_domain integMsq(syst->give_val_def("intMsq")()(2));
    double const Mirrsq = space->get_domain(2)->integ(integMsq, INNER_BC);
    double const Mirr = std::sqrt(Mirrsq);

    Val_domain integMadm(syst->give_val_def("intMadm")()(ndom - 1));
    double const Madm = space->get_domain(ndom - 1)->integ(integMadm, OUTER_BC);

    Val_domain integMk(syst->give_val_def("intMk")()(ndom - 1));
    double const Mk = space->get_domain(ndom - 1)->integ(integMk, OUTER_BC);

    Val_domain integS(syst->give_val_def("intS")()(2));
    double const S = space->get_domain(2)->integ(integS, INNER_BC);
    // Christodoulou mass
    double const Mch = std::sqrt(Mirrsq + S * S / 4. / Mirrsq);

    std::ios_base::fmtflags f(std::cout.flags());
    std::cout << "=======================================" << std::endl
              << FORMAT << "Iter: " << ite << std::endl
              << FORMAT << "Error: " << conv << std::endl
              << FORMAT << "Madm: " << Madm << std::endl
              << FORMAT << "Mk: " << Mk << " [" << std::abs(Madm - Mk) / Madm
              << "]" << std::endl
              << FORMAT << "Mirr: " << Mirr << std::endl
              << FORMAT << "R: " << r << std::endl
              << FORMAT << "Omega: " << (*bconfig)(OMEGA) << std::endl
              << FORMAT << "S: " << S << std::endl
              << FORMAT << "Mch: " << Mch << std::endl
              << FORMAT << "Chi: " << S / Mch / Mch << std::endl;
    std::cout.flags(f);
    std::cout << "=======================================" << "\n\n";
}

inline bool BH_XCTS_base::increment_resolution() {
    auto const final_res = resolution->final();
    if (!(final_res > resolution->init()) || last_stage_idx != solver_stage)
        return false;

    auto const resolution_indices = resolution->get_indices();
    int const res = (*bconfig)(resolution_indices);
    if (res >= final_res)
        return false;

    bconfig->set(resolution_indices) = next_resolution_step(res, final_res);
    return true;
}

}  // namespace Kadath::FUKA_Solvers
