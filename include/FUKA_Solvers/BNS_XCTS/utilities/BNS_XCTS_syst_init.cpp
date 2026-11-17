namespace Kadath::FUKA_Solvers {
inline void BNS_XCTS_base::syst_init() {
    auto& cfg = *bconfig;

    // call the (flat) conformal metric "f"
    fmet->set_system(*syst, "f");

    syst->add_cst("4piG", cfg(BIN_PARAMS::QPIG));

    // baryonic mass and dimensionless spin are fixed input parameters
    // along with the ADM of each NS at infinite separation
    syst->add_cst("Madm1", cfg(BCO_PARAMS::MADM, NODES::BCO1));
    syst->add_cst("Mb1", cfg(BCO_PARAMS::MB, NODES::BCO1));
    syst->add_cst("chi1", cfg(BCO_PARAMS::CHI, NODES::BCO1));

    syst->add_cst("Madm2", cfg(BCO_PARAMS::MADM, NODES::BCO2));
    syst->add_cst("Mb2", cfg(BCO_PARAMS::MB, NODES::BCO2));
    syst->add_cst("chi2", cfg(BCO_PARAMS::CHI, NODES::BCO2));

    binary_syst::add_coordinate_fields(*syst, *coord_vectors);

    syst->add_var("P", *conformal_factor);
    syst->add_var("N", *lapse);
    syst->add_var("bet", *shift);

    if (!binary_syst::add_spin_parameters(*syst, cfg, *phi)) {
        binary_syst::add_star_spin_defs(*syst,
                                        space->NS1,
                                        space->ADAPTED1,
                                        "omes1",
                                        "mm");
        binary_syst::add_star_spin_defs(*syst,
                                        space->NS2,
                                        space->ADAPTED2,
                                        "omes2",
                                        "mp");
    }

    syst->add_def("NP = P*N");
    syst->add_def("Ntilde = N / P^6");
    syst->add_def(
        "A^ij = (D^i bet^j + D^j bet^i - 2. / 3.* D_k bet^k * f^ij) / "
        "2. / Ntilde");

    // ADM and Komar masses, ADM angular and linear momenta at infinity
    syst->add_def(ndom - 1, "intMadm = - einf^i * D_i P / 4piG * 2");
    syst->add_def(ndom - 1, "intMk = einf^i * D_i N / 4piG");
    syst->add_def(ndom - 1, "intMadmalt = -dr(P) * 2 / 4piG");
    syst->add_def(ndom - 1, "intJ = multr(A_ij * mg^j * einf^i) / 2. / 4piG");
    binary_syst::add_momentum_integrands(*syst, ndom);

    // quasi-local spins, outside the stellar matter distributions
    syst->add_def(space->ADAPTED1 + 1,
                  "intS1 = A_ij * mm^i * sm^j / 2. / 4piG");
    syst->add_def(space->ADAPTED2 + 1,
                  "intS2 = A_ij * mp^i * sp^j / 2. / 4piG");

    binary_syst::add_matter_defs(*syst);
}
}  // namespace Kadath::FUKA_Solvers
