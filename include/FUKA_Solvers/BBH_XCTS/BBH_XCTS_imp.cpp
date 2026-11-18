namespace Kadath::FUKA_Solvers {
inline BBH_XCTS::BBH_XCTS(base_config_t* config_,
                          bco_seq_t const& bh1_seq_,
                          bco_seq_t const& bh2_seq_,
                          bin_res_seq_t const& res_,
                          std::string outputdir_,
                          int const rank_)
    : binary_base_t(config_, bh1_seq_, bh2_seq_, res_, outputdir_, rank_) {
    reload();
}

inline std::string BBH_XCTS::converged_filename(const std::string stage) const {
    return binary_converged_filename("BBH",
                                     stage,
                                     (*bconfig)(BCO_PARAMS::MCH, NODES::BCO1),
                                     (*bconfig)(BCO_PARAMS::MCH, NODES::BCO2),
                                     "");
}

inline void BBH_XCTS::update_config_quantities() {
    // with a fixed spin frequency (or corotation) the Christodoulou mass
    // follows from the solution
    auto& cfg = *bconfig;
    bool const corot = cfg.control(CONTROLS::COROT_BIN);
    if (corot || !std::isnan(cfg.set(BCO_PARAMS::FIXED_BCOMEGA, NODES::BCO1)))
        cfg(BCO_PARAMS::MCH, NODES::BCO1) =
            bco_utils::syst_mch(*syst, *space, "intSm", space->BH1 + 2);
    if (corot || !std::isnan(cfg.set(BCO_PARAMS::FIXED_BCOMEGA, NODES::BCO2)))
        cfg(BCO_PARAMS::MCH, NODES::BCO2) =
            bco_utils::syst_mch(*syst, *space, "intSp", space->BH2 + 2);
}

inline bool BBH_XCTS::increment_resolution() {
    auto const final_res = resolution->final();
    if (!(final_res > resolution->init()))
        return false;

    auto const resolution_indices = resolution->get_indices();
    int const res = (*bconfig)(resolution_indices);
    if (res >= final_res)
        return false;

    bconfig->set(resolution_indices) = next_resolution_step(res, final_res);
    return true;
}

inline void BBH_XCTS::syst_init() {
    auto& cfg = *bconfig;

    // call the (flat) conformal metric "f"
    fmet->set_system(*syst, "f");

    syst->add_cst("4piG", cfg(BIN_PARAMS::QPIG));
    syst->add_cst("PI", M_PI);

    // fixed irreducible masses, spins and Christodoulou masses of the holes
    syst->add_cst("Mm", cfg(BCO_PARAMS::MIRR, NODES::BCO1));
    syst->add_cst("chim", cfg(BCO_PARAMS::CHI, NODES::BCO1));
    syst->add_cst("CMm", cfg(BCO_PARAMS::MCH, NODES::BCO1));
    syst->add_cst("Mp", cfg(BCO_PARAMS::MIRR, NODES::BCO2));
    syst->add_cst("chip", cfg(BCO_PARAMS::CHI, NODES::BCO2));
    syst->add_cst("CMp", cfg(BCO_PARAMS::MCH, NODES::BCO2));

    binary_syst::add_coordinate_fields(*syst, *coord_vectors);

    syst->add_var("P", *conformal_factor);
    syst->add_var("N", *lapse);
    syst->add_var("bet", *shift);

    // spin of each hole: rotation frequency fixed (FIXED_BCOMEGA) or solved
    // for chi; none for corotation
    if (cfg.control(CONTROLS::COROT_BIN)) {
        cfg.set(BCO_PARAMS::OMEGA, NODES::BCO1) = 0.;
        cfg.set(BCO_PARAMS::OMEGA, NODES::BCO2) = 0.;
    } else {
        auto add_omega = [&](NODES const bco,
                             char const* name,
                             int const dom,
                             char const* def) {
            if (!std::isnan(cfg.set(BCO_PARAMS::FIXED_BCOMEGA, bco))) {
                syst->add_cst(name, cfg(BCO_PARAMS::FIXED_BCOMEGA, bco));
                cfg.set(BCO_PARAMS::OMEGA, bco) =
                    cfg(BCO_PARAMS::FIXED_BCOMEGA, bco);
            } else {
                syst->add_var(name, cfg(BCO_PARAMS::OMEGA, bco));
            }
            syst->add_def(dom, def);
        };
        add_omega(NODES::BCO1,
                  "omesm",
                  space->BH1 + 2,
                  "ExOme^i = omesm * mm^i");
        add_omega(NODES::BCO2,
                  "omesp",
                  space->BH2 + 2,
                  "ExOme^i = omesp * mp^i");
    }

    syst->add_def("NP = P*N");
    syst->add_def("Ntilde = N / P^6");
    syst->add_def("Morb^i  = mg^i + xaxis * ey^i + yaxis * ex^i");
    syst->add_def(
        "A^ij = (D^i bet^j + D^j bet^i - 2. / 3.* D_k bet^k * f^ij) / 2. / "
        "Ntilde");

    // ADM linear momentum at infinity, spins on the horizons
    syst->add_def("intPx   = A_ij * ex^j * einf^i / 8 / PI");
    syst->add_def("intPy   = A_ij * ey^j * einf^i / 8 / PI");
    syst->add_def("intPz   = A_ij * ez^j * einf^i / 8 / PI");
    syst->add_def("intSm   = A_ij * mm^i * sm^j   / 8 / PI");
    syst->add_def("intSp   = A_ij * mp^i * sp^j   / 8 / PI");

    // center-of-mass diagnostic, arXiv:1506.01689
    syst->add_def(ndom - 1, "COMx  = -3 * P^4 * einf^i * ex_i / 8. / PI");
    syst->add_def(ndom - 1, "COMy  = 3 * P^4 * einf^i * ey_i / 8. / PI");
    syst->add_def(ndom - 1, "COMz  = 3 * P^4 * einf^i * ez_i / 8. / PI");

    // irreducible mass integrand on the horizons; ADM and Komar masses
    syst->add_def("intMsq  = P^4 / 16. / PI");
    syst->add_def(ndom - 1, "intMadm = - einf^i * D_i P / 4piG * 2");
    syst->add_def(ndom - 1, "intMk = einf^i * D_i N / 4piG");
}

}  // namespace Kadath::FUKA_Solvers
