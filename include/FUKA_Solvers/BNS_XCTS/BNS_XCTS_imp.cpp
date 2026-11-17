
namespace Kadath::FUKA_Solvers {
inline BNS_XCTS_base::BNS_XCTS_base(base_config_t* config_,
                                    seq_t const& ns1_seq_,
                                    seq_t const& ns2_seq_,
                                    res_seq_t const& res_,
                                    std::string outputdir_,
                                    int const rank_)
    : solver_base_t(config_, outputdir_, rank_),
      ns1_seq(new seq_t(ns1_seq_)),
      ns2_seq(new seq_t(ns2_seq_)),
      resolution(new res_seq_t(res_)),
      xc1(0),
      xc2(0),
      xo(0),
      H_scale1(0),
      H_scale2(0) {
    if (rank == 0) {
        cout << *ns1_seq << endl;
        cout << *ns2_seq << endl;
        cout << *resolution << endl;
    }
}

inline void BNS_XCTS_base::reset_all_ptrs() {
    reset_support_containers();

    conformal_factor.reset(nullptr);
    lapse.reset(nullptr);
    shift.reset(nullptr);
    logh.reset(nullptr);
    phi.reset(nullptr);

    space.reset(nullptr);
}

inline void BNS_XCTS_base::reset_support_containers() {
    // refers to the space, rebuilt by the hydro-rescaling setup
    logh_const.reset(nullptr);
    solver_base_t::reset_support_containers();
}

inline void BNS_XCTS_base::post_newton_step() {
    // overwrite logh with the scaled version for later use and output
    if (logh_const)
        *logh = syst->give_val_def("H");
}

inline void BNS_XCTS_base::update_coordinate_fields() {
    Kadath::update_fields(*cfields,
                          *coord_vectors,
                          {},
                          xo,
                          xc1,
                          xc2,
                          syst.get());
}

inline bool BNS_XCTS_base::is_compatible_solution(base_config_t& old_config) {
    return bconfig->set(BCO_PARAMS::NSHELLS, NODES::BCO1) ==
               old_config.set(BCO_PARAMS::NSHELLS, NODES::BCO1) &&
           bconfig->set(BCO_PARAMS::NSHELLS, NODES::BCO2) ==
               old_config.set(BCO_PARAMS::NSHELLS, NODES::BCO2);
}

inline void BNS_XCTS_base::initialize_support_containers() {
    basis.reset(new Base_tensor(shift->get_basis()));
    fmet.reset(new Metric_flat(*space, *basis));

    cfields.reset(new cfgen_t(*space));
    coord_vectors =
        std::make_unique<cfary_t>(default_binary_vector_ary(*space));

    xc1 = bco_utils::get_center(*space, space->NS1);
    xc2 = bco_utils::get_center(*space, space->NS2);
    xo = bco_utils::get_center(*space, ndom - 1);
}

inline void BNS_XCTS_base::save_solution_to_file() const {
    bconfig->set_outputdir(outputdir);

    Kadath::bco_utils::save_to_file(*space,
                                    *bconfig,
                                    *conformal_factor,
                                    *lapse,
                                    *shift,
                                    *logh,
                                    *phi);
}

inline void BNS_XCTS_base::load_solution_from_file() {
    std::string spacein{bconfig->space_filename()};
    FILE* ff1 = fopen(spacein.c_str(), "r");

    space.reset(new base_space_t{ff1, !bconfig->control(CONTROLS::NEW_ID)});
    conformal_factor.reset(new Scalar(*space, ff1));
    lapse.reset(new Scalar(*space, ff1));
    shift.reset(new Vector(*space, ff1));
    logh.reset(new Scalar(*space, ff1));
    phi.reset(new Scalar(*space, ff1));

    conformal_factor->coef();
    lapse->coef();
    shift->coef();
    logh->coef();
    phi->coef();

    fclose(ff1);

    ndom = space->get_nbr_domains();
}

inline std::string BNS_XCTS_base::converged_filename(
    const std::string stage) const {
    // FIXME assumes a fix resolution for all domains
    auto res = space->get_domain(0)->get_nbr_points()(0);
    const std::string eosname{extract_eos_name(*bconfig, NODES::BCO1)};
    std::stringstream ss;
    ss << "BNS";

    auto M1 = ns2_seq->get_mass_fixing_val();
    auto M2 = ns1_seq->get_mass_fixing_val();
    if (M2 > M1)
        std::swap(M1, M2);
    bconfig->set(Q) = M2 / M1;
    auto Mtot = M1 + M2;

    if (stage != "")
        ss << "_" << stage << ".";
    else
        ss << ".";
    ss << eosname << "." << (*bconfig)(DIST) << "." << (*bconfig)(CHI, BCO1)
       << "." << (*bconfig)(CHI, BCO2) << "." << Mtot << ".q" << (*bconfig)(Q)
       << "." << std::setfill('0') << std::setw(1) << (*bconfig)(NSHELLS, BCO1)
       << "." << std::setfill('0') << std::setw(1) << (*bconfig)(NSHELLS, BCO2)
       << "." << std::setfill('0') << std::setw(2) << res;
    return ss.str();
}

}  // namespace Kadath::FUKA_Solvers