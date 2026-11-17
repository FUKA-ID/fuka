namespace Kadath::FUKA_Solvers {
inline void BHNS_XCTS_base::syst_init() {
    auto& cfg = *bconfig;

    // call the (flat) conformal metric "f"
    fmet->set_system(*syst, "f");

    syst->add_cst("4piG", cfg(BIN_PARAMS::QPIG));

    // fixed parameters of the star (ADM mass at infinite separation, baryon
    // mass, spin) and of the hole (Christodoulou and irreducible mass, spin)
    syst->add_cst("Madm1", cfg(BCO_PARAMS::MADM, NODES::BCO1));
    syst->add_cst("Mb1", cfg(BCO_PARAMS::MB, NODES::BCO1));
    syst->add_cst("chi1", cfg(BCO_PARAMS::CHI, NODES::BCO1));

    syst->add_cst("Mch", cfg(BCO_PARAMS::MCH, NODES::BCO2));
    syst->add_cst("Mirr", cfg(BCO_PARAMS::MIRR, NODES::BCO2));
    syst->add_cst("chi2", cfg(BCO_PARAMS::CHI, NODES::BCO2));

    binary_syst::add_coordinate_fields(*syst, *coord_vectors);

    syst->add_var("P", *conformal_factor);
    syst->add_var("N", *lapse);
    syst->add_var("bet", *shift);

    // quasi-local ADM mass of the star
    syst->add_var("qlMadm1", cfg(BCO_PARAMS::QLMADM, NODES::BCO1));

    if (!binary_syst::add_spin_parameters(*syst, cfg, *phi)) {
        binary_syst::add_star_spin_defs(*syst,
                                        space->NS,
                                        space->ADAPTEDNS,
                                        "omes1",
                                        "mm");
        syst->add_def(space->ADAPTEDBH + 1, "s^i = omes2 * mp^i");
    }

    syst->add_def("NP = P*N");
    syst->add_def("Ntilde = N / P^6");

    // ADM and Komar masses at infinity, irreducible mass of the horizon
    syst->add_def(ndom - 1, "intMadm = - einf^i * D_i P / 4piG * 2");
    syst->add_def(ndom - 1, "intMk = einf^i * D_i N / 4piG");
    syst->add_def(ndom - 1, "intMadmalt = -dr(P) * 2 / 4piG");
    syst->add_def("intMsq  = P^4 / 4. / 4piG");

    binary_syst::add_matter_defs(*syst);

    syst->add_def(
        "A^ij   = (D^i bet^j + D^j bet^i - 2. / 3.* D_k bet^k * f^ij) / 2. / "
        "Ntilde");

    binary_syst::add_momentum_integrands(*syst, ndom);

    // quasi-local spins, outside the star and on the horizon
    syst->add_def(space->ADAPTEDNS + 1,
                  "intS1 = A_ij * mm^i * sm^j / 2. / 4piG");
    syst->add_def(space->ADAPTEDBH + 1,
                  "intS2 = A_ij * mp^i * sp^j / 2. / 4piG");
}
}  // namespace Kadath::FUKA_Solvers
