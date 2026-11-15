/*
 * T8: the MUMPS orderings use 64-bit indices.
 *
 * MUMPS refuses a graph with more than 2^31 - 1 entries (INFOG(1) = -51) when METIS/ParMETIS or (PT-)Scotch
 * were built with 32-bit indices; FUKA then falls back to the AMF ordering, which is much slower and gives
 * larger factors. MUMPS decides this from the index width it was compiled against, which it reports with
 * mumps_metis_idxsize_ / mumps_scotch_intsize_ (libmumps_common). This test checks both are 64 (Scotch only if
 * MUMPS was built with it), so a MUMPS linked against 32-bit orderings is caught at build time instead of in
 * a large production run.
 *
 * The end-to-end check (a MUMPS analysis of a graph above 2^31 entries) is test_ordering_large, which needs a
 * full node and is run separately (see the README).
 * Serial; returns 0 if all checks pass, 77 when built without MUMPS.
 */
#include <iostream>

#ifdef KADATH_USE_MUMPS
extern "C" {
void mumps_metis_idxsize_(int* size);
void mumps_scotch_intsize_(int* size);
}
#endif

int main() {
#ifndef KADATH_USE_MUMPS
    std::cout << "test_ordering_int64: SKIP (built without MUMPS)" << std::endl;
    return 77;
#else
    int failures = 0;
    int metis = 0, scotch = 0;
    mumps_metis_idxsize_(&metis);
    mumps_scotch_intsize_(&scotch);

    bool const metis_ok = metis == 64;
    failures += !metis_ok;
    std::cout << (metis_ok ? "ok:   " : "FAIL: ") << "METIS/ParMETIS index width seen by MUMPS = " << metis
              << " bits (need 64; rebuild metis/parmetis with +int64, see installation.md)" << std::endl;

    if (scotch == -99999) {
        std::cout << "ok:   MUMPS built without (PT-)Scotch" << std::endl;
    } else {
        bool const scotch_ok = scotch == 64;
        failures += !scotch_ok;
        std::cout << (scotch_ok ? "ok:   " : "FAIL: ") << "(PT-)Scotch integer width seen by MUMPS = " << scotch
                  << " bits (need 64; rebuild scotch with +int64)" << std::endl;
    }

    std::cout << "test_ordering_int64: " << (failures == 0 ? "PASS" : "FAIL") << " (" << failures << " failures)"
              << std::endl;
    return failures == 0 ? 0 : 1;
#endif
}
