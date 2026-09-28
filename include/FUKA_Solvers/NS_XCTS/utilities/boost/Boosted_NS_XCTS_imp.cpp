namespace Kadath::FUKA_Solvers {
template <class eos_t>
Boosted_NS_XCTS<eos_t>::Boosted_NS_XCTS(
    base_config_t* config_,
    ns_sequence const& seq_,
    Parameter_sequence<BCO_PARAMS> const& res_,
    std::string outputdir_,
    binary_config_t* binary_config_,
    int const rank_)
    : NS_XCTS_UNIFORM_ROT<eos_t>(config_, seq_, res_, outputdir_, rank_) {

    this->solver_stage = Kadath::FUKA_Config::STAGES::BIN_BOOST;
    this->stagename = "BIN_BOOST";

    if (binary_config_ != nullptr) {
        binary_config.reset(binary_config_);
        std::stringstream stage_ss;
        stage_ss << "BIN_BOOST_" << (*binary_config)(BIN_PARAMS::DIST) << "_"
                 << (*binary_config)(BIN_PARAMS::GOMEGA);
        this->stagename = stage_ss.str();
    }
}

template <class eos_t>
void Boosted_NS_XCTS<eos_t>::save_to_file() const {
    bconfig->set_outputdir(this->outputdir);
    Scalar rescaled_logh(syst->give_val_def("H")());
    if (this->diff_omega) {
        bconfig->set_field(Kadath::FUKA_Config::BCO_FIELDS::DIFF_OMEGA) = true;
        Kadath::bco_utils::save_to_file(*this->space,
                                        *this->bconfig,
                                        *this->conformal_factor,
                                        *this->lapse,
                                        *this->shift,
                                        rescaled_logh,
                                        *this->phi,
                                        *this->diff_omega);
    } else {
        Kadath::bco_utils::save_to_file(*this->space,
                                        *this->bconfig,
                                        *this->conformal_factor,
                                        *this->lapse,
                                        *this->shift,
                                        rescaled_logh,
                                        *this->phi);
    }
}

template <class eos_t>
void Boosted_NS_XCTS<eos_t>::setup_syst() {
    if (this->rank == 0) {
        std::cout << "############################" << std::endl
                  << "Boosted Rigid NS Solver" << std::endl
                  << "Fixing parameters:\n"
                  << "Madm: " << (*bconfig)(BCO_PARAMS::MADM) << std::endl
                  << "Mb: " << (*bconfig)(BCO_PARAMS::MB) << std::endl
                  << "Chi: " << (*bconfig)(BCO_PARAMS::CHI) << std::endl
                  << "############################" << std::endl;
    }
    // We use `config_filename()` vs `config_filename_abs()` since
    // `solution_exists` will probe the HOME_KADATH/COs directory
    auto const current = bconfig->config_filename();
    if (!bconfig->control(RESOLVE) && solution_exists()) {
        if (rank == 0) {
            std::cout << "Solved previously: " << bconfig->config_filename_abs()
                      << std::endl;
        }
        reload();
    }

    // flag needs to be set in order to be used during import into
    // a BNS or BHNS setup
    this->bconfig->set_field(PHI) = true;

    update_fields_co(*cfields, *coord_vectors, {}, 0.);
    syst.reset(new System_of_eqs(*space));

    syst->add_var("phi", *phi);

    syst->add_cst("Hconst", *(this->logh));
    syst->add_var("Hscale", H_scale);
    for (int d = 0; d < ndom; d++)
        // the enthalpy is equal to the constant part everywhere
        // outside of the stars, i.e. zero
        if (d > 1)
            syst->add_def(d, "H  = Hconst");
        else
            syst->add_def(d, "H  = Hconst * (1. + Hscale)");
    this->syst_init();

    // MB and MADM are now fixed as they would be in a binary
    syst->add_cst("Mb", (*bconfig)(BCO_PARAMS::MB));
    syst->add_cst("Madm", (*bconfig)(BCO_PARAMS::MADM));

    // Fix Spin with CHI for now
    syst->add_cst("chi", (*bconfig)(BCO_PARAMS::CHI));
    syst->add_var("ome", (*bconfig)(BCO_PARAMS::OMEGA));

    syst->add_cst("omega_boost", (*binary_config)(BIN_PARAMS::GOMEGA));
    syst->add_def("B^i = bet^i + omega_boost * (mg^i)");

    // define the irrotational + spinning parts of the fluid velocity
    for (int d = 0; d <= 1; ++d) {
        syst->add_def(d, "s^i  = ome * mg^i");
        syst->add_def(d, "eta_i  = D_i phi + P^4 * s_i");
    }

    for (int d = 0; d < ndom; d++) {
        switch (d) {
            case 0:
            case 1:
                // definitions for the fluid 3-velocity
                // and its Lorentz factor
                syst->add_def(d, "Wsquare= eta^i * eta_i / h^2 / P^4 + 1.");
                syst->add_def(d, "W      = sqrt(Wsquare)");
                syst->add_def(d, "U^i    = eta^i / P^4 / h / W");
                syst->add_def(d, "Usquare= P^4 * U_i * U^i");
                syst->add_def(d, "V^i    = N * U^i - B^i");
                syst->add_def(d, "firstint = log(h * N / W + D_i phi * V^i)");
                syst->add_def(d,
                              "eqphi  = P^6 * W * V^i * D_i H + dHdlnrho * D_i "
                              "(P^6 * W * V^i)");

                // rescaled sources and constraint equations
                syst->add_def(d,
                              "Etilde = press * h * Wsquare - press * delta");
                syst->add_def(d,
                              "Stilde = 4 * press * delta + (Etilde + press * "
                              "delta) * Usquare");
                syst->add_def(d, "ptilde^i = press * h * Wsquare * U^i");

                syst->add_def(
                    d,
                    "eqP    = delta * D^i D_i P + A_ij * A^ij / P^7 / 8 * "
                    "delta + 4piG / 2. * P^5 * Etilde");
                syst->add_def(
                    d,
                    "eqNP   = delta * D^i D_i NP - 7. / 8. * NP / P^8 * delta "
                    "* A_ij *A^ij "
                    "- 4piG / 2. * N * P^5 * (Etilde + 2. * Stilde)");
                syst->add_def(
                    d,
                    "eqbet^i= delta * D_j D^j bet^i + delta * D^i D_j bet^j / "
                    "3. "
                    "- 2. * delta * A^ij * D_j Ntilde - 4. * 4piG * N * P^4 * "
                    "ptilde^i");

                // integrant of the baryonic mass integral
                syst->add_def(d, "intMb = P^6 * rho(h) * W");

                break;
            default:
                // outside the star the matter is absent and the sources are zero
                // syst->add_eq_full(d, "H = 0");
                syst->add_eq_full(d, "phi = 0");

                syst->add_def(d, "eqP = D^i D_i P + A_ij * A^ij / P^7 / 8");
                syst->add_def(
                    d,
                    "eqNP = D^i D_i NP - 7. / 8. * NP / P^8 * A_ij * A^ij");
                syst->add_def(
                    d,
                    "eqbet^i = D_j D^j bet^i + D^i D_j bet^j / 3. - 2. * "
                    "A^ij * D_j Ntilde");
                break;
        }
    }

    space->add_eq(*syst, "eqNP= 0", "N", "dn(N)");
    space->add_eq(*syst, "eqP= 0", "P", "dn(P)");
    space->add_eq(*syst, "eqbet^i= 0", "bet^i", "dn(bet^i)");

    syst->add_eq_bc(ndom - 1, OUTER_BC, "N=1");
    syst->add_eq_bc(ndom - 1, OUTER_BC, "P=1");
    syst->add_eq_bc(ndom - 1, OUTER_BC, "bet^i=0");

    syst->add_eq_bc(1, OUTER_BC, "H = 0");
    syst->add_eq_bc(1, OUTER_BC, "V^i * D_i H = 0");
    // in case of the stellar domains
    syst->add_eq_vel_pot(0, 2, "eqphi = 0", "phi=0");
    syst->add_eq_matching(0, OUTER_BC, "phi");
    syst->add_eq_matching(0, OUTER_BC, "dn(phi)");
    syst->add_eq_vel_pot(1, 2, "eqphi = 0", "phi=0");

    space->add_eq_int(*syst,
                      2,
                      OUTER_BC,
                      "integ(intS) - chi * Madm * Madm = 0");
    space->add_eq_int_volume(*syst, 2, "integvolume(intMb) = Mb");
}

template <class eos_t>
void Boosted_NS_XCTS<eos_t>::load_solution_from_file() {
    std::string spacein{bconfig->space_filename()};
    FILE* ff1 = fopen(spacein.c_str(), "r");

    this->space.reset(new base_space_t{ff1});
    this->conformal_factor.reset(new Scalar(*space, ff1));
    this->lapse.reset(new Scalar(*space, ff1));
    this->shift.reset(new Vector(*space, ff1));
    this->logh.reset(new Scalar(*space, ff1));

    if (bconfig->field(Kadath::FUKA_Config::BCO_FIELDS::PHI)) {
        phi.reset(new Scalar(*space.get(), ff1));
    } else if (binary_config) {
        phi.reset(new Kadath::Scalar(*space));
        phi->annule_hard();
        phi->std_base();
    }
    fclose(ff1);

    ndom = space->get_nbr_domains();
}
}  // namespace Kadath::FUKA_Solvers
