
namespace Kadath::FUKA_Solvers {
BNS_XCTS_base::BNS_XCTS_base()
    : FUKA_Solver_base(),
      bconfig(nullptr),
      basis(nullptr),
      fmet(nullptr),
      cfields(nullptr),
      coord_vectors(nullptr),
      syst(nullptr),
      conformal_factor(nullptr),
      lapse(nullptr),
      shift(nullptr),
      logh(nullptr),
      logh_const(nullptr),
      phi(nullptr),
      xc1(0),
      xc2(0),
      xo(0),
      H_scale1(0),
      H_scale2(0) {};

BNS_XCTS_base::BNS_XCTS_base(BNS_XCTS_base::base_config_t* config_,
                             seq_t const& ns1_seq_,
                             seq_t const& ns2_seq_,
                             res_seq_t const& res_,
                             std::string outputdir_,
                             int const rank_)
    : BNS_XCTS_base() {
    this->rank = rank_;
    this->outputdir = outputdir_;

    bconfig.reset(config_);

    std::array<bool, NUM_STAGES>& stage_enabled = bconfig->return_stages();
    auto [last_stage_, last_stage_idx_] =
        get_last_enabled(MSTAGE, stage_enabled);
    last_stage_idx = last_stage_idx_;

    ns1_seq.reset(new seq_t(ns1_seq_));
    ns2_seq.reset(new seq_t(ns2_seq_));
    resolution.reset(new res_seq_t(res_));

    if (rank == 0) {
        cout << *ns1_seq << endl;
        cout << *ns2_seq << endl;
        cout << *resolution << endl;
    }
}

inline void BNS_XCTS_base::reset_base_ptrs() {
    // Fields
    conformal_factor.reset(nullptr);
    lapse.reset(nullptr);
    shift.reset(nullptr);
    logh.reset(nullptr);
    phi.reset(nullptr);

    logh_const.reset(nullptr);

    // Containers
    basis.reset(nullptr);
    fmet.reset(nullptr);
    syst.reset(nullptr);
    cfields.reset(nullptr);
    coord_vectors.reset(nullptr);

    // Space
    space.reset(nullptr);
}

/**
 * @brief Consistent interface for writing a checkpoint
 *
 * @param termination_chkpt Toggle writing to stdout for termination
 */
inline void BNS_XCTS_base::checkpoint(bool termination_chkpt) const {
    // Backup activated stages
    auto const final_stages{bconfig->return_stages()};

    // Clear stages and only set the current stage as active
    auto& stages = bconfig->return_stages();
    stages.fill(false);
    stages[this->solver_stage] = true;

    // Save to file
    save_solution_to_file();

    // Reset to original stages
    stages = final_stages;

    if (termination_chkpt) {
        std::stringstream ss;
        ss << "***Writing early termination chkpt "
           << bconfig->config_filename() << "***\n";
        throw std::runtime_error(ss.str().c_str());
    }
}

inline void BNS_XCTS_base::initialize_support_containers() {
    basis.reset(new Base_tensor(shift->get_basis()));
    fmet.reset(new Metric_flat(*space, *basis));

    cfields.reset(new cfgen_t(*space));
    coord_vectors =
        std::make_unique<cfary_t>(default_binary_vector_ary(*space));
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

    space.reset(new base_space_t{ff1});
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