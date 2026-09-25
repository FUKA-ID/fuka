namespace Kadath::FUKA_Solvers {
void BNS_XCTS_base::syst_init() {
    using namespace ::Kadath::Margherita;

    // call the (flat) conformal metric "f"
    fmet->set_system(*syst, "f");

    // define numerical constants
    syst->add_cst("4piG", (*bconfig)(BCO_QPIG));

    // baryonic mass and dimensionless spin are fixed input parameters
    // along with the ADM of each NS at infinite separation
    syst->add_cst("Madm1", (*bconfig)(BCO_PARAMS::MADM, NODES::BCO1));
    syst->add_cst("Mb1", (*bconfig)(BCO_PARAMS::MB, NODES::BCO1));
    syst->add_cst("chi1", (*bconfig)(BCO_PARAMS::CHI, NODES::BCO1));

    syst->add_cst("Madm2", (*bconfig)(BCO_PARAMS::MADM, NODES::BCO2));
    syst->add_cst("Mb2", (*bconfig)(BCO_PARAMS::MB, NODES::BCO2));
    syst->add_cst("chi2", (*bconfig)(BCO_PARAMS::CHI, NODES::BCO2));

    // include the coordinate fields
    syst->add_cst("mg", *(*coord_vectors)[GLOBAL_ROT]);
    syst->add_cst("mm", *(*coord_vectors)[BCO1_ROT]);
    syst->add_cst("mp", *(*coord_vectors)[BCO2_ROT]);

    // cartesianbasis vector fields
    syst->add_cst("ex", *(*coord_vectors)[EX]);
    syst->add_cst("ey", *(*coord_vectors)[EY]);
    syst->add_cst("ez", *(*coord_vectors)[EZ]);

    syst->add_cst("sm", *(*coord_vectors)[S_BCO1]);
    syst->add_cst("sp", *(*coord_vectors)[S_BCO2]);
    syst->add_cst("einf", *(*coord_vectors)[S_INF]);

    // the basic fields, conformal factor, lapse and (log) enthalpy
    syst->add_var("P", *conformal_factor);
    syst->add_var("N", *lapse);
    syst->add_var("bet", *shift);

    // check if corotation is considered and adjust
    // rotational velocity components accordingly
    if (bconfig->control(CONTROLS::COROT_BIN)) {
        // for a corotating binary, there is no angular frequency parameter
        // since stellar rotation is fixed by the orbital frequency
        bconfig->set(BCO_PARAMS::OMEGA, NODES::BCO1) = 0.;
        syst->add_cst("omes1", (*bconfig)(BCO_PARAMS::OMEGA, NODES::BCO1));

        bconfig->set(BCO_PARAMS::OMEGA, NODES::BCO2) = 0.;
        syst->add_cst("omes2", (*bconfig)(BCO_PARAMS::OMEGA, NODES::BCO2));
    } else {
        // in the arbitrary spinning / irrotational case
        // the irrotational part is defined by the velocity potential
        syst->add_var("phi", *phi);

        // check if stellar omega is fixed or has to be solved
        // for a specific dimensionless spin
        if (!std::isnan(bconfig->set(BCO_PARAMS::FIXED_BCOMEGA, NODES::BCO1))) {
            syst->add_cst("omes1",
                          (*bconfig)(BCO_PARAMS::FIXED_BCOMEGA, NODES::BCO1));
            bconfig->set(BCO_PARAMS::OMEGA, NODES::BCO1) =
                (*bconfig)(BCO_PARAMS::FIXED_BCOMEGA, NODES::BCO1);
        } else {
            syst->add_var("omes1", (*bconfig)(BCO_PARAMS::OMEGA, NODES::BCO1));
        }

        if (!std::isnan(bconfig->set(BCO_PARAMS::FIXED_BCOMEGA, NODES::BCO2))) {
            syst->add_cst("omes2",
                          (*bconfig)(BCO_PARAMS::FIXED_BCOMEGA, NODES::BCO2));
            bconfig->set(BCO_PARAMS::OMEGA, NODES::BCO2) =
                (*bconfig)(BCO_PARAMS::FIXED_BCOMEGA, NODES::BCO2);
        } else {
            syst->add_var("omes2", (*bconfig)(BCO_PARAMS::OMEGA, NODES::BCO2));
        }

        // for mixed spins, we have additional definitions
        // that are required for both objects that describe
        // the local rotation field
        for (int d = space->NS1; d <= space->ADAPTED1; ++d) {
            syst->add_def(d, "s^i  = omes1 * mm^i");
            syst->add_def(d, "eta_i  = D_i phi + P^4 * s_i");
        }
        for (int d = space->NS2; d <= space->ADAPTED2; ++d) {
            syst->add_def(d, "s^i  = omes2 * mp^i");
            syst->add_def(d, "eta_i  = D_i phi + P^4 * s_i");
        }
    }

    // define common combinations of conformal factor and lapse
    syst->add_def("NP = P*N");
    syst->add_def("Ntilde = N / P^6");
    syst->add_def(
        "A^ij = (D^i bet^j + D^j bet^i - 2. / 3.* D_k bet^k * f^ij) / "
        "2. / Ntilde");

    // define quantity to be integrated at infinity
    // two (in this case) equivalent definitions of ADM mass
    // as well as the Komar mass
    syst->add_def(ndom - 1, "intMadm = - einf^i * D_i P / 4piG * 2");
    syst->add_def(ndom - 1, "intMk = einf^i * D_i N / 4piG");
    syst->add_def(ndom - 1, "intMadmalt = -dr(P) * 2 / 4piG");

    // ADM Angular momentum
    syst->add_def(ndom - 1, "intJ = multr(A_ij * mg^j * einf^i) / 2. / 4piG");
    // Quasi-local spin angular momentum
    // ADM linear momentum, surface integrant at infinity
    syst->add_def(ndom - 1, "intPx = A_i^j * ex_j * einf^i");
    syst->add_def(ndom - 1, "intPy = A_i^j * ey_j * einf^i");
    syst->add_def(ndom - 1, "intPz = A_i^j * ez_j * einf^i");

    // quasi-local spin, surface integral outside the stellar matter distribution
    syst->add_def(space->ADAPTED1 + 1,
                  "intS1 = A_ij * mm^i * sm^j / 2. / 4piG");
    syst->add_def(space->ADAPTED2 + 1,
                  "intS2 = A_ij * mp^i * sp^j / 2. / 4piG");

    // enthalpy from the logarithmic enthalpy, the latter is the actual variable
    // in this system
    syst->add_def("h = exp(H)");

    // define rest-mass density, internal energy and pressure through the
    // enthalpy
    syst->add_def("rho = rho(h)");
    syst->add_def("eps = eps(h)");
    syst->add_def("press = press(h)");
    syst->add_def("dHdlnrho = dHdlnrho(h)");

    // definition to rescale the equations
    // delta = p / rho
    syst->add_def("delta = h - eps - 1.");
}
}  // namespace Kadath::FUKA_Solvers