/*
    Copyright 2017 Philippe Grandclement & Gregoire Martinon

    This file is part of Kadath.

    Kadath is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    Kadath is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with Kadath.  If not, see <http://www.gnu.org/licenses/>.
*/

#include "mpi.h"
#include "system_of_eqs.hpp"
#include "matrice.hpp"
#include "scalar.hpp"
#include "metric.hpp"
#include "array.hpp"
#include "Linear_solvers/newton_linear_solve.hpp"
#include "Linear_solvers/sparse_triplets.hpp"
#ifdef KADATH_USE_MUMPS
#include "Linear_solvers/gmres.hpp"
#include "Linear_solvers/mumps_solver.hpp"
#endif

#include <chrono>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <memory>
#include <string>
#include <vector>

namespace Kadath {
extern "C" {
    void sl_init_ (int*, int*, int*);
    void blacs_gridexit_ (int*);
    void blacs_gridinfo_ (int*, int*, int*, int*, int*);
    void blacs_gridinit_ (int*, char*, int*, int*);
    int numroc_ (int*, int*, int*, int*, int*);
    int indxl2g_ (int*, int*, int*, int*, int*);
    void descinit_ (int*, int*, int*, int*, int* , int*, int*, int*, int*, int*);

    void pdgesv_ (int*, int*, double*, int*, int*, int*, int*, double*, int*, int*, int*, int*);
    void pdtran_ (int*, int*, double*, double*, int*, int*, int*, double*, double*, int*, int*, int*);
    void pdgemv_ (char*, int*, int*, double*, double*, int*, int*, int*, double*, int*, int*, int*, int*, double*, double*, int*, int*, int*, int*);
    void pdgemm_ (char* transa, char* transb, int* m, int* n, int* k, double* alpha, double* a, int* ia, int* ja, int* desc_a, double* b, int* ib, int* jb, int* desc_b, double* beta, double* c, int* ic, int* jc, int* desc_c);

    int blacs_pnum_ (int*, int*, int*);
    void pdgemr2d_ (int *m , int *n , double *a , int *ia , int *ja , int *desca , double *b , int *ib , int *jb , int *descb , int *ictxt );
    void Cpdgemr2d (int, int, double*, int, int, int*, double*, int, int, int*, int);
}

double elapsed_time(std::chrono::time_point<std::chrono::system_clock> const & begin) {
  auto end = std::chrono::system_clock::now();
  std::chrono::duration<double> diff = end - begin;
  return diff.count();
}

void split_2d (int target, int& low, int& up) {
  up = std::ceil(std::sqrt(target));

  while(target % up)
    ++up;

  low = target / up;
}

namespace linear_solvers {

Array<double> newton_update_scalapack(System_of_eqs& syst, Array<double> const& second, System_of_eqs::SOLVER solver) {
	int rank, nproc;
	MPI_Comm_rank (MPI_COMM_WORLD, &rank);
	MPI_Comm_size (MPI_COMM_WORLD, &nproc);

	int m = second.get_size(0);
	int n = syst.get_nbr_unknowns();

	// Computation in a 1D distributed matrice
	int zero_i = 0;
	int one_i = 1;

	double zero_d = 0.;
	double one_d = 1.;

  // create an input process grid to generate J
  // processor grid context
	int ictxt_in;

  // processor grid extents
  // we want to fill full cols locally, so generate a 1d process grid
  int nprow_in = 1;
  int npcol_in = nproc;

  // init context
  sl_init_ (&ictxt_in, &nprow_in, &npcol_in);

	// Get processor row and col
  int prow_in, pcol_in;
  blacs_gridinfo_ (&ictxt_in, &nprow_in, &npcol_in, &prow_in, &pcol_in);

  int bsize = 256;

  while(nproc * bsize > m)
    bsize = bsize / 2;

  if(rank == 0) {
    if(bsize < 64) {
      std::cout << "Block size is " << bsize << ", consider lowering the number of cores." << std::endl;
    }
  }

  // local row and col extent
	int nrow_in = numroc_ (&m, &bsize, &prow_in, &zero_i, &nprow_in);
	int ncol_in = numroc_ (&n, &bsize, &pcol_in, &zero_i, &npcol_in);

  // collect global columns assigned to this proc, covers full rows because of the linear process grid above
  std::vector<int> globcol_in;
  for(int j = 1; j <= ncol_in; ++j)
    globcol_in.push_back(indxl2g_(&j,&bsize,&pcol_in,&zero_i,&npcol_in) - 1);

  auto begin = std::chrono::system_clock::now();

  // actually compute the local cols of J
  std::vector<Array<double>> col_J;
  for(int j = 0; j < ncol_in; ++j) {
    col_J.push_back(syst.do_col_J(globcol_in[j]));
  }

  // distributed storage for local columns of J
  // we need col major order!
	Array<double> J_in (ncol_in, nrow_in);

  // every proc spans all rows in this case because of the 1d processor grid
  for(int j = 0; j < ncol_in; ++j) {
    for(int i = 0; i < nrow_in; ++i) {
      J_in.set(j,i) = col_J[j](i);
    }
  }

  // catching errors
  int info;

  // descriptor of J
  Array<int> J_desc_in(9);
  descinit_ (J_desc_in.set_data(), &m, &n, &bsize, &bsize, &zero_i, &zero_i, &ictxt_in, &nrow_in, &info);

  // create the compute processor grid
	int ictxt;

  // processor grid extents
  int nprow;
  int npcol;
  split_2d(nproc, nprow, npcol);

  // init context
  sl_init_ (&ictxt, &nprow, &npcol);

	// Get processor row and col
  int prow, pcol;
  blacs_gridinfo_ (&ictxt, &nprow, &npcol, &prow, &pcol);

  MPI_Barrier(MPI_COMM_WORLD);

  // local row and col extent
	int nrow = numroc_ (&m, &bsize, &prow, &zero_i, &nprow);
	int ncol = numroc_ (&n, &bsize, &pcol, &zero_i, &npcol);

  // local copy of J for the computation
  // we need col major order!
	Array<double> J (ncol, nrow);

  // descriptor of J
  Array<int> J_desc(9);
  descinit_ (J_desc.set_data(), &m, &n, &bsize, &bsize, &zero_i, &zero_i, &ictxt, &nrow, &info);

  // now distribute J_in over the compute grid to J
  pdgemr2d_ (&m, &n,
             J_in.set_data(), &one_i , &one_i, J_desc_in.set_data(),
             J.set_data(), &one_i , &one_i, J_desc.set_data(),
             &ictxt);

  J_in.delete_data();

  // local rows of B
	int nrow_B = numroc_ (&m, &bsize, &prow, &zero_i, &nprow);
	Array<double> B (nrow_B);

  for(int i = 1; i <= nrow_B; ++i) {
    int glob_row = indxl2g_(&i,&bsize,&prow,&zero_i,&nprow) - 1;
    B.set(i-1) = second(glob_row);
  }

  // descriptor of B
  Array<int> B_desc(9);
  descinit_ (B_desc.set_data(), &m, &one_i, &bsize, &one_i, &zero_i, &zero_i, &ictxt, &nrow_B, &info);

  double time = elapsed_time(begin);

  double maxtime;
  double mintime;
  double avgtime;

  MPI_Reduce(&time, &maxtime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
  MPI_Reduce(&time, &mintime, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
  MPI_Reduce(&time, &avgtime, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

  if (rank == 0) {
    avgtime /= nproc;
    std::cout << "Load and redistrib Jacobian: " << avgtime << " / " << mintime << " / " << maxtime << " seconds (avg/min/max)" << endl;
  }

  begin = std::chrono::system_clock::now();

  // local copy of the solution
    Array<double>& X_sol = B;

    if(solver == System_of_eqs::GAUSS_NEWTON) {
    // local rows of J^T B
  	int nrow_X = numroc_ (&n, &bsize, &prow, &zero_i, &nprow);
  	Array<double> JTB (nrow_X);

    // descriptor of J^T B
    Array<int> JTB_desc(9);
    descinit_ (JTB_desc.set_data(), &n, &one_i, &bsize, &one_i, &zero_i, &zero_i, &ictxt, &nrow_X, &info);

    // local copy of J^T J
  	int nrow_sym = numroc_ (&n, &bsize, &prow, &zero_i, &nprow);
  	int ncol_sym = numroc_ (&n, &bsize, &pcol, &zero_i, &npcol);
  	Array<double> JTJ (ncol_sym, nrow_sym);

    // descriptor of J^T J
    Array<int> JTJ_desc(9);
    descinit_ (JTJ_desc.set_data(), &n, &n, &bsize, &bsize, &zero_i, &zero_i, &ictxt, &nrow_sym, &info);

    // FIXME precomputing J^T and storing it could be a bit faster, since we need it after this again
    // compute J^T B
    char trans = 'T';
    pdgemv_(&trans, &m, &n,
            &one_d,
            J.set_data(), &one_i, &one_i, J_desc.set_data(),
            B.set_data(), &one_i, &one_i, B_desc.set_data(), &one_i,
            &zero_d,
            JTB.set_data(), &one_i, &one_i, JTB_desc.set_data(), &one_i);

    // compute J^T J
    char no_trans = 'N';
    pdgemm_ (&trans, &no_trans,
             &n, &n, &m, &one_d,
             J.set_data(), &one_i, &one_i, J_desc.set_data(),
             J.set_data(), &one_i, &one_i, J_desc.set_data(),
             &zero_d,
             JTJ.set_data(), &one_i, &one_i, JTJ_desc.set_data());

  	// solve the linear system
    Array<int> ipiv (n);

    pdgesv_ (&n, &one_i,
             JTJ.set_data(), &one_i, &one_i, JTJ_desc.set_data(),
             ipiv.set_data(),
             JTB.set_data(), &one_i, &one_i, JTB_desc.set_data(), &info);

    // local copy of the solution
    X_sol = JTB;
  }

  if(solver == System_of_eqs::NEWTON_RAPHSON) {
  	// solve the linear system
    Array<int> ipiv (n);

    pdgesv_ (&n, &one_i,
             J.set_data(), &one_i, &one_i, J_desc.set_data(),
             ipiv.set_data(),
             B.set_data(), &one_i, &one_i, B_desc.set_data(), &info);
  }

  if(info != 0) {
    if (rank == 0) {
      std::cout << "Jacobian is singular! Stopping." << endl;
    }
    MPI_Barrier(MPI_COMM_WORLD);

    abort();
  }

  if (rank == 0) {
    std::cout << "Inverting the matrix: " << elapsed_time(begin) << " seconds" << endl;
  }

  begin = std::chrono::system_clock::now();

  // map number of elements to rank
  std::vector<int> n_elements(nproc);

  for(int i = 0; i < nprow; ++i) {
    for(int j = 0; j < npcol; ++j) {
      // rank as function of row and col proc number (default ordering)
      int r = i * npcol + j;
      // vectors are only distributed over first proc column
    	int n_elem = j == 0 ? numroc_(&n, &bsize, &i, &zero_i, &nprow) : 0;

      n_elements[r] = n_elem;
    }
  }

  // no displacement
  std::vector<int> disp(nproc, 0);

  int offset = 0;
  for(int r = 0; r < nproc; ++r) {
    disp[r] = offset;
    offset += n_elements[r];
  }

  // recieve buffer
  std::vector<double> recv(n);

  // distribute full solution to all processors
  MPI_Allgatherv(X_sol.set_data(), n_elements[rank], MPI_DOUBLE,
                 recv.data(), n_elements.data(), disp.data(), MPI_DOUBLE,
                 MPI_COMM_WORLD);

  // rearrange data to correct ordering
  Array<double>  X(n);

  for(int i = 0; i < nprow; ++i) {
    for(int j = 0; j < npcol; ++j) {
      // rank as function of row and col proc number (default ordering)
      int r = i * npcol + j;

      if(n_elements[r] > 0) {
        // data is ordered by ranks, so get correct offset
        int pos = 0;
        for(int k = 0; k < r; ++k)
          pos += n_elements[k];

        for(int k = 0; k < n_elements[r]; ++k) {
          int elem = k + 1;
          int glob_row = indxl2g_(&elem,&bsize,&i,&zero_i,&nprow) - 1;

          X.set(glob_row) = recv[pos + k];
        }
      }
    }
  }

	blacs_gridexit_ (&ictxt_in);
	blacs_gridexit_ (&ictxt);

	return X;
}

namespace {

void report_rank_times(char const* label, double time) {
	int rank, nproc;
	MPI_Comm_rank (MPI_COMM_WORLD, &rank);
	MPI_Comm_size (MPI_COMM_WORLD, &nproc);

	double maxtime, mintime, avgtime;
	MPI_Reduce(&time, &maxtime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
	MPI_Reduce(&time, &mintime, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
	MPI_Reduce(&time, &avgtime, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

	if (rank == 0)
		std::cout << label << ": " << avgtime / nproc << " / " << mintime << " / " << maxtime
		          << " seconds (avg/min/max)" << std::endl;
}

/// Peak resident memory (VmHWM) per rank, per node (sum over its ranks) and in total, printed by rank 0 after each
/// Newton step so runs can be sized to the fewest nodes; collective.
void report_memory() {
	long hwm_kb = 0;
	std::ifstream status ("/proc/self/status");
	std::string key;
	while (status >> key) {
		if (key == "VmHWM:") {
			status >> hwm_kb;
			break;
		}
		status.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
	}
	static MPI_Comm node = MPI_COMM_NULL;
	if (node == MPI_COMM_NULL)
		MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &node);
	int rank, nproc;
	MPI_Comm_rank (MPI_COMM_WORLD, &rank);
	MPI_Comm_size (MPI_COMM_WORLD, &nproc);
	long node_kb = 0, rank_max_kb = 0, node_max_kb = 0, total_kb = 0, other_max_kb = 0;
	long const other_kb = rank == 0 ? 0 : hwm_kb;
	MPI_Allreduce(&hwm_kb, &node_kb, 1, MPI_LONG, MPI_SUM, node);
	MPI_Reduce(&hwm_kb, &rank_max_kb, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
	MPI_Reduce(&other_kb, &other_max_kb, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
	MPI_Reduce(&node_kb, &node_max_kb, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
	MPI_Reduce(&hwm_kb, &total_kb, 1, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
	// Rank 0 gathers the matrix and runs the sequential analysis, so it is reported apart from the others.
	if (rank == 0)
		std::cout << "Memory high-water mark (VmHWM): max " << rank_max_kb / 1024 << " MB per rank, rank 0 "
		          << hwm_kb / 1024 << " MB, other ranks max " << other_max_kb / 1024 << " / mean "
		          << (nproc > 1 ? (total_kb - hwm_kb) / (nproc - 1) / 1024 : 0) << " MB, max "
		          << node_max_kb / 1024. / 1024. << " GB per node, total " << total_kb / 1024. / 1024. << " GB"
		          << std::endl;
}

}  // namespace

#ifdef KADATH_USE_MUMPS
namespace {

/// Entries at most this fraction of both their row and their column maximum are dropped from every assembled
/// Jacobian. About two thirds of a Kadath Jacobian are such round-off (sums of spectral terms that cancel); they
/// only cost fill (BH: 47M -> 10M entries, analysis + factorization 122 -> 19 s). Tolerances 1e-16 ... 1e-13 left
/// the BH Newton history and solution unchanged (parameters <= 1e-13 relative).
constexpr double kJacobianNoise = 1e-14;

/// Newton-Krylov settings (validated on the BH, BBH, NS and BNS CI cases):
constexpr int kGmresMaxIterations = 48;       ///< GMRES iterations before the preconditioner is refactored.
constexpr double kEtaMin = 1e-8;              ///< Bounds of the Eisenstat-Walker forcing term.
constexpr double kEtaMax = 1e-3;
/// The preconditioner is factored from the Jacobian without entries below this fraction of their row and column
/// maximum (about 75 % of the entries); the exact step is then finished by GMRES with the true J v.
constexpr double kPreconditionerDrop = 1e-13;

/// kPreconditionerDrop, or KADATH_KRYLOV_DROP if set (experiments with larger drop tolerances: fewer entries for
/// rank 0 to gather and order, more GMRES iterations); read on rank 0 once and broadcast. Collective.
double preconditioner_drop() {
	static double drop = -1.;
	if (drop < 0.) {
		int rank;
		MPI_Comm_rank (MPI_COMM_WORLD, &rank);
		double value = kPreconditionerDrop;
		if (rank == 0) {
			char const* env = std::getenv("KADATH_KRYLOV_DROP");
			if (env != nullptr && std::atof(env) > 0.) {
				value = std::atof(env);
				std::cout << "Newton-Krylov: preconditioner drop tolerance " << value << " (KADATH_KRYLOV_DROP)"
				          << std::endl;
			}
		}
		MPI_Bcast(&value, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		drop = value;
	}
	return drop;
}
constexpr double kExactStepTolerance = 1e-10;  ///< GMRES tolerance of that exact step.
/// MUMPS residual check of the dropped factors: GMRES with the true J v finishes the step, so a slightly
/// inaccurate preconditioner must not send MUMPS down the retry ladder.
constexpr double kPreconditionerResidual = 1e-6;

/// Assembles the sparse Jacobian, solves J x = second along the MUMPS retry ladder and reports statistics; the
/// successful solver (with its factors) is returned in @p solver. Collective. @return 0 or the last status.
/// @param drop_tolerance > kJacobianNoise removes more entries (drop_negligible): the factors and x then belong to
/// a perturbed matrix, so only preconditioners may use it, and the MUMPS residual check is relaxed to
/// kPreconditionerResidual.
/// @param row_max if given, receives the largest |entry| of every row of the assembled Jacobian (all ranks).
int sparse_newton_step(System_of_eqs& syst, Array<double> const& second, std::vector<double>& x,
                       std::unique_ptr<MumpsSolver>& solver, double drop_tolerance = 0.,
                       std::vector<double>* row_max_out = nullptr) {
	int rank, nproc;
	MPI_Comm_rank (MPI_COMM_WORLD, &rank);
	MPI_Comm_size (MPI_COMM_WORLD, &nproc);

	int const n = syst.get_nbr_unknowns();
	auto begin = std::chrono::system_clock::now();

	SparseTriplets local;
	// Round-robin columns: unknowns are ordered domain by domain, so every rank gets the same share of every
	// domain and the per-domain cost differences of do_col_J average out.
	for (int col : block_cyclic_columns(n, 1, rank, nproc))
		append_column(syst.do_col_J(col), col, local);
	report_rank_times("Assemble sparse Jacobian", elapsed_time(begin));

	bool const preconditioner = drop_tolerance > kJacobianNoise;
	double const tolerance = std::max(drop_tolerance, kJacobianNoise);
	{
		std::vector<double> row_max, col_max;
		local_row_column_max(local, n, row_max, col_max);
		MPI_Allreduce(MPI_IN_PLACE, row_max.data(), n, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
		if (row_max_out != nullptr)
			*row_max_out = row_max;
		long long counts[2] = {static_cast<long long>(local.size()), 0};
		counts[1] = drop_negligible(local, row_max, col_max, tolerance);
		MPI_Allreduce(MPI_IN_PLACE, counts, 2, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
		if (rank == 0)
			std::cout << (preconditioner ? "Preconditioner" : "Jacobian") << ": dropped " << counts[1] << " of "
			          << counts[0] << " entries (" << 100. * counts[1] / std::max(1LL, counts[0])
			          << " %) at relative tolerance " << tolerance << std::endl;
	}

	// Retry ladder on the same assembled matrix; every attempt passes the residual check or counts as failed.
	// Default KADATH_MUMPS_INPUT=distributed: distributed input with the column permutation computed in the wrapper
	// (MumpsOptions::matching_entries/growth) instead of gathering the matrix on rank 0 for MUMPS. Rank 0 gathers
	// a sample of large entries (about 10 % of large matrices), not the matrix plus MUMPS's copy: on the BBH/BHNS
	// test cases its peak is 36-39 % lower and the runs are faster than with centralized input (MUMPS's own
	// permutation takes most of its sequential analysis). KADATH_MUMPS_INPUT=centralized restores the gather.
	static int input_mode = -1;
	if (input_mode < 0) {
		int mode = 0;
		if (rank == 0) {
			char const* env = std::getenv("KADATH_MUMPS_INPUT");
			std::string const value = env ? env : "distributed";
			mode = 1;
			if (value == "centralized")
				mode = 0;
			else if (value != "distributed")
				std::cerr << "KADATH_MUMPS_INPUT=" << value << " unknown (centralized or distributed); using distributed"
				          << std::endl;
			if (mode == 0)
				std::cout << "MUMPS input: centralized (KADATH_MUMPS_INPUT=centralized), matrix gathered on rank 0"
				          << std::endl;
		}
		MPI_Bcast(&mode, 1, MPI_INT, 0, MPI_COMM_WORLD);
		input_mode = mode;
	}
	// KADATH_MUMPS_BLR=<eps>: block low-rank factors at relative accuracy eps (MumpsOptions::blr_tolerance), only
	// for the Newton-Krylov preconditioner, whose step GMRES finishes with the true J v.
	// KADATH_MUMPS_OOC=auto (default) | off | on: out-of-core factors (MumpsOptions::out_of_core). auto writes the
	// factors to disk only when they do not fit in memory, and says so in the log; off stops with an insufficient
	// memory error instead; on always uses the disk. KADATH_MUMPS_OOC_DIR: directory (default $TMPDIR or /tmp).
	static bool env_read = false;
	static double blr_tolerance = 0.;
	static int out_of_core = kOutOfCoreAuto;
	static std::string ooc_dir;
	if (!env_read) {
		char buffer[1024] = {0};
		if (rank == 0) {
			char const* blr = std::getenv("KADATH_MUMPS_BLR");
			blr_tolerance = blr ? std::atof(blr) : 0.;
			char const* ooc = std::getenv("KADATH_MUMPS_OOC");
			std::string const mode = ooc ? ooc : "auto";
			if (mode == "off" || mode == "0")
				out_of_core = kOutOfCoreNever;
			else if (mode == "on" || mode == "1")
				out_of_core = kOutOfCoreAlways;
			else if (mode != "auto")
				std::cerr << "KADATH_MUMPS_OOC=" << mode << " unknown (auto, off, on); using auto" << std::endl;
			char const* dir = std::getenv("KADATH_MUMPS_OOC_DIR");
			if (dir)
				std::snprintf(buffer, sizeof buffer, "%s", dir);
			if (blr_tolerance > 0.)
				std::cout << "MUMPS: block low-rank factors at tolerance " << blr_tolerance
				          << " for the Newton-Krylov preconditioner" << std::endl;
			if (out_of_core == kOutOfCoreAlways)
				std::cout << "MUMPS: out-of-core factors (KADATH_MUMPS_OOC=on)" << std::endl;
			else if (out_of_core == kOutOfCoreNever)
				std::cout << "MUMPS: out-of-core disabled (KADATH_MUMPS_OOC=off): stop if the factors do not fit in memory"
				          << std::endl;
		}
		MPI_Bcast(&blr_tolerance, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		MPI_Bcast(&out_of_core, 1, MPI_INT, 0, MPI_COMM_WORLD);
		MPI_Bcast(buffer, sizeof buffer, MPI_CHAR, 0, MPI_COMM_WORLD);
		ooc_dir = buffer;
		env_read = true;
	}
	MumpsOptions base_options;
	base_options.row_domains = syst.condition_domains();
	base_options.col_domains = syst.unknown_domains();
	base_options.out_of_core = out_of_core;
	base_options.ooc_dir = ooc_dir;
	if (preconditioner)
		base_options.blr_tolerance = blr_tolerance;
	if (input_mode == 1) {
		base_options.centralized = false;
		base_options.matching_entries = 16;
		base_options.matching_growth = 4;
		// KADATH_MUMPS_MATCHING="entries=16:rows=4:growth=4:warm=1" (any separator): sampling of the wrapper's matching.
		static int matching[4] = {-1, -1, -1, -1};  // entries, rows, growth, warm
		static bool matching_read = false;
		if (!matching_read) {
			if (rank == 0 && std::getenv("KADATH_MUMPS_MATCHING") != nullptr) {
				std::string const spec = std::getenv("KADATH_MUMPS_MATCHING");
				char const* names[4] = {"entries=", "rows=", "growth=", "warm="};
				for (int f = 0; f < 4; ++f) {
					std::size_t const at = spec.find(names[f]);
					if (at != std::string::npos)
						matching[f] = std::atoi(spec.c_str() + at + std::string(names[f]).size());
				}
				std::cout << "MUMPS matching: " << spec << std::endl;
			}
			MPI_Bcast(matching, 4, MPI_INT, 0, MPI_COMM_WORLD);
			matching_read = true;
		}
		if (matching[0] >= 0)
			base_options.matching_entries = matching[0];
		if (matching[1] >= 0)
			base_options.matching_row_entries = matching[1];
		if (matching[2] >= 0)
			base_options.matching_growth = matching[2];
		if (matching[3] >= 0)
			base_options.matching_warm_start = matching[3] != 0;
	}
	std::vector<MumpsAttempt> const ladder = mumps_retry_ladder(base_options);
	int status = 0;
	bool internal_ordering = false;  // after -51 every attempt uses AMF: the graph is the same for all of them
	for (std::size_t attempt = 0; attempt < ladder.size(); ++attempt) {
		x.assign(second.get_data(), second.get_data() + n);
		MumpsOptions options = ladder[attempt].options;
		if (internal_ordering)
			options = with_internal_ordering(options);
		if (preconditioner)
			options.residual_tolerance = std::max(options.residual_tolerance, kPreconditionerResidual);
		solver.reset();
		solver = std::make_unique<MumpsSolver>(MPI_COMM_WORLD, options);
		status = solver->solve(n, local, x);
		if (rank == 0 && status != 0) {
			MumpsStats const& f = solver->stats();
			std::cout << "MUMPS attempt " << attempt + 1 << " (" << ladder[attempt].name
			          << ") failed: " << mumps_error_message(status) << "; residual = " << f.relative_residual
			          << ", pivots delayed = " << f.delayed_pivots << ", off-diagonal = " << f.off_diagonal_pivots
			          << ", estimated factor entries = " << f.estimated_factor_entries << std::endl;
		}
		if (status == kInsufficientMemory)  // the other configurations need as much memory
			break;
		if (status == kGraphTooLarge && !internal_ordering) {
			internal_ordering = true;
			if (rank == 0)
				std::cout << "MUMPS: retrying " << ladder[attempt].name << " with the AMF ordering" << std::endl;
			--attempt;
			continue;
		}
		if (rank == 0 && status != 0 && attempt + 1 < ladder.size())
			std::cout << "MUMPS: retrying with " << ladder[attempt + 1].name << std::endl;
		if (status == 0) {
			if (rank == 0 && attempt > 0)
				std::cout << "MUMPS: attempt " << attempt + 1 << " (" << ladder[attempt].name << ") succeeded"
				          << std::endl;
			break;
		}
	}

	if (status == 0 && rank == 0) {
		MumpsStats const& s = solver->stats();
		std::cout << "MUMPS: nnz = " << s.nnz << " (" << 100. * s.nnz / (1.0 * n * n) << " % of dense), factors = "
		          << s.factor_entries;
		if (s.out_of_core)
			std::cout << " (OUT-OF-CORE: factors on disk)";
		if (s.compressed_factor_entries != s.factor_entries)
			std::cout << " (" << s.compressed_factor_entries << " stored with BLR)";
		std::cout << ", memory max/total = " << s.peak_mem_mb_max << " / " << s.peak_mem_mb_total
		          << " MB, relative residual = " << s.relative_residual << ", pivots delayed = " << s.delayed_pivots
		          << std::endl;
		std::cout << "Inverting the matrix: " << s.time_gather + s.time_matching + s.time_analysis + s.time_factor_solve
		          << " seconds (gather " << s.time_gather << ", matching " << s.time_matching << ", analysis "
		          << s.time_analysis << ", factor+solve "
		          << s.time_factor_solve << ")" << std::endl;
	}
	return status;
}

/// Stops the run after every configuration of the retry ladder failed; collective.
[[noreturn]] void abort_after_mumps_failure(int status) {
	int rank;
	MPI_Comm_rank (MPI_COMM_WORLD, &rank);
	if (rank == 0)
		std::cout << "MUMPS failed with every configuration of the retry ladder (" << mumps_error_message(status)
		          << "); stopping. KADATH_LINSOLVE=scalapack solves densely if memory allows." << std::endl;
	MPI_Barrier(MPI_COMM_WORLD);
	abort();
}

}  // namespace

Array<double> newton_update_mumps(System_of_eqs& syst, Array<double> const& second, int* status_out) {
	int rank;
	MPI_Comm_rank (MPI_COMM_WORLD, &rank);
	int const n = syst.get_nbr_unknowns();
	std::vector<double> x;
	std::unique_ptr<MumpsSolver> solver;
	int const status = sparse_newton_step(syst, second, x, solver);

	if (status != 0) {
		if (rank == 0 && (status == -6 || status == -10))
			std::cout << "Jacobian is singular! Stopping. (" << mumps_error_message(status) << ")" << std::endl;
		if (status_out == nullptr)
			abort_after_mumps_failure(status);
		*status_out = status;
		Array<double> none (n);
		none = 0.;
		return none;
	}
	if (status_out != nullptr)
		*status_out = 0;

	Array<double> X (n);
	for (int i = 0; i < n; ++i)
		X.set(i) = x[i];
	return X;
}

namespace {

/// Preconditioner and history kept between Newton-Krylov steps of one system.
struct KrylovState {
	unsigned long syst_id = 0;     ///< System_of_eqs::get_instance_id() of the factored system (0: none).
	int n = 0;
	std::unique_ptr<MumpsSolver> preconditioner;
	double previous_norm = -1.;    ///< ||second||_2 of the previous step.
	double previous_eta = -1.;     ///< Forcing term of the previous GMRES step (< 0: none).
	/// Row scaling of the GMRES system: 1 / (largest |entry| of the row) of the factored Jacobian. Kadath rows
	/// differ by orders of magnitude (integral conditions grow as M^2, field equations as 1/M^2), and without
	/// it GMRES and the forcing term would only see the largest rows. One factor per equation instead (the
	/// largest entry of all its rows) diverged on the BH CI case: an equation's largest entries come from a few
	/// rows, and the others were again ignored.
	std::vector<double> row_scale;
	NewtonKrylovStats stats;
};

KrylovState& krylov_state();

/// MPI_COMM_SELF attributes are deleted first in MPI_Finalize, while MPI still works: release the kept MUMPS
/// instance there (its JOB=-2 communicates) instead of in the static destructor after MPI_Finalize.
int release_krylov_state(MPI_Comm, int, void*, void*) {
	krylov_state() = KrylovState{};
	return MPI_SUCCESS;
}

KrylovState& krylov_state() {
	static KrylovState state;
	static bool registered = false;
	if (!registered) {
		registered = true;
		int keyval;
		MPI_Comm_create_keyval(MPI_COMM_NULL_COPY_FN, release_krylov_state, &keyval, nullptr);
		MPI_Comm_set_attr(MPI_COMM_SELF, keyval, nullptr);
	}
	return state;
}

}  // namespace

NewtonKrylovStats const& newton_krylov_stats() {
	return krylov_state().stats;
}

void reset_newton_krylov() {
	krylov_state() = KrylovState{};
}

/// Newton-Krylov after arXiv:2608.07653: GMRES with the matrix-free J v = do_JX(v), right-preconditioned by the
/// MUMPS factors of an earlier Jacobian; Eisenstat-Walker forcing (choice 2, Eisenstat & Walker, SIAM J. Sci.
/// Comput. 17, 16 (1996), eq. 2.6 with gamma = 0.9, alpha = 2, safeguard 3.4) bounded to [kEtaMin, kEtaMax].
/// The preconditioner is refactored (an exact sparse Newton step) when absent, when n or the system (instance
/// id) changed, when ||F|| grew, or when GMRES does not converge within kGmresMaxIterations. It is factored from
/// the Jacobian without entries below kPreconditionerDrop; the exact step is then solved by GMRES with the true
/// J v to kExactStepTolerance, and refactored without dropping if that fails. The fields only change after this
/// returns, so a failed GMRES leaves the system untouched for the exact step.
Array<double> newton_update_krylov(System_of_eqs& syst, Array<double> const& second) {
	int rank;
	MPI_Comm_rank (MPI_COMM_WORLD, &rank);
	int const n = syst.get_nbr_unknowns();
	KrylovState& st = krylov_state();

	// GMRES solves the row-scaled system (D J) dx = D second, D = st.row_scale, right-preconditioned by the
	// kept factors: M^{-1} = J_old^{-1} D^{-1}. dx is unchanged by the scaling; residuals and the forcing term are
	// measured in the scaled norm, which weighs every equation alike.
	std::vector<double> b;
	double norm = 0.;
	auto scale_rhs = [&]() {
		bool const scaled = static_cast<int>(st.row_scale.size()) == n;
		b.assign(second.get_data(), second.get_data() + n);
		norm = 0.;
		for (int i = 0; i < n; ++i) {
			if (scaled)
				b[i] *= st.row_scale[i];
			norm += b[i] * b[i];
		}
		MPI_Bcast(b.data(), n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		norm = std::sqrt(norm);
		MPI_Bcast(&norm, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	};
	scale_rhs();

	auto apply_j = [&syst, &st, n](std::vector<double> const& v, std::vector<double>& w) {
		Array<double> av (n);
		for (int i = 0; i < n; ++i)
			av.set(i) = v[i];
		Array<double> const jv (syst.do_JX(av));
		w.assign(jv.get_data(), jv.get_data() + n);
		MPI_Bcast(w.data(), n, MPI_DOUBLE, 0, MPI_COMM_WORLD);  // every rank continues with identical data
		if (static_cast<int>(st.row_scale.size()) == n)
			for (int i = 0; i < n; ++i)
				w[i] *= st.row_scale[i];
	};
	int preconditioner_status = 0;
	auto apply_m = [&st, &preconditioner_status, n](std::vector<double> const& v, std::vector<double>& z) {
		z = v;
		if (static_cast<int>(st.row_scale.size()) == n)
			for (int i = 0; i < n; ++i)
				z[i] /= st.row_scale[i];
		int const s = st.preconditioner->solve_factored(z);
		if (s != 0)
			preconditioner_status = s;
	};

	std::string reason;  // why this step refactors (empty: GMRES with the kept preconditioner)
	if (!st.preconditioner || !st.preconditioner->has_factors())
		reason = "no preconditioner";
	else if (st.n != n)
		reason = "new resolution";
	else if (st.syst_id != syst.get_instance_id())
		reason = "new system";
	else if (st.previous_norm > 0. && norm > st.previous_norm)
		reason = "residual grew";
	std::vector<double> x;

	if (reason.empty()) {
		double const gamma = 0.9, alpha = 2.;
		double eta = gamma * std::pow(norm / st.previous_norm, alpha);
		if (st.previous_eta > 0. && gamma * std::pow(st.previous_eta, alpha) > 0.1)
			eta = std::max(eta, gamma * std::pow(st.previous_eta, alpha));
		eta = std::min(kEtaMax, std::max(kEtaMin, eta));

		auto begin = std::chrono::system_clock::now();
		GmresResult const result = gmres(apply_j, apply_m, b, x, eta, kGmresMaxIterations);
		MPI_Bcast(x.data(), n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		double const t = elapsed_time(begin);

		if (result.converged && preconditioner_status == 0) {
			st.previous_norm = norm;
			st.previous_eta = eta;
			++st.stats.krylov_steps;
			st.stats.krylov_iterations += result.iterations;
			if (rank == 0)
				std::cout << "Newton-Krylov: GMRES " << result.iterations << " iterations, relative residual "
				          << result.relative_residual << " <= forcing term " << eta << ", " << t << " seconds"
				          << std::endl;
			Array<double> X (n);
			for (int i = 0; i < n; ++i)
				X.set(i) = x[i];
			return X;
		}
		++st.stats.gmres_failures;
		reason = "GMRES missed the forcing term";
		if (rank == 0)
			std::cout << "Newton-Krylov: GMRES did not reach " << eta << " in " << result.iterations
			          << " iterations (relative residual " << result.relative_residual << ", preconditioner status "
			          << preconditioner_status << "); refactoring the Jacobian" << std::endl;
	}

	if (rank == 0)
		std::cout << "Newton-Krylov: exact sparse Newton step, new preconditioner (" << reason << ")" << std::endl;
	std::vector<double> row_max;
	int status = sparse_newton_step(syst, second, x, st.preconditioner, preconditioner_drop(), &row_max);
	if (status == 0) {
		// New row scaling from this Jacobian (rows without entries keep 1).
		st.row_scale.assign(n, 1.);
		for (int i = 0; i < n; ++i)
			if (row_max[i] > 0.)
				st.row_scale[i] = 1. / row_max[i];
		scale_rhs();
		// The factors belong to the dropped matrix: solve with the true J v, preconditioned by them.
		preconditioner_status = 0;
		auto begin = std::chrono::system_clock::now();
		GmresResult const result = gmres(apply_j, apply_m, b, x, kExactStepTolerance, kGmresMaxIterations);
		MPI_Bcast(x.data(), n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		if (result.converged && preconditioner_status == 0) {
			st.stats.krylov_iterations += result.iterations;
			if (rank == 0)
				std::cout << "Newton-Krylov: exact step finished by GMRES, " << result.iterations
				          << " iterations, relative residual " << result.relative_residual << ", "
				          << elapsed_time(begin) << " seconds" << std::endl;
		} else {
			if (rank == 0)
				std::cout << "Newton-Krylov: GMRES with the dropped preconditioner did not reach "
				          << kExactStepTolerance << " (relative residual " << result.relative_residual
				          << "); refactoring without dropping" << std::endl;
			status = sparse_newton_step(syst, second, x, st.preconditioner);
		}
	}
	if (status != 0) {
		st = KrylovState{};
		abort_after_mumps_failure(status);
	}
	st.syst_id = syst.get_instance_id();
	st.n = n;
	st.previous_norm = norm;
	st.previous_eta = -1.;
	++st.stats.refactorizations;

	Array<double> X (n);
	for (int i = 0; i < n; ++i)
		X.set(i) = x[i];
	return X;
}
#endif

NewtonBackend newton_backend_from_env() {
	int rank;
	MPI_Comm_rank (MPI_COMM_WORLD, &rank);
#ifdef KADATH_USE_MUMPS
	std::string choice = "krylov";
#else
	std::string choice = "scalapack";
#endif
	if (rank == 0 && std::getenv("KADATH_LINSOLVE") != nullptr)
		choice = std::getenv("KADATH_LINSOLVE");
	int len = static_cast<int>(choice.size());
	MPI_Bcast(&len, 1, MPI_INT, 0, MPI_COMM_WORLD);
	choice.resize(len);
	if (len > 0)
		MPI_Bcast(&choice[0], len, MPI_CHAR, 0, MPI_COMM_WORLD);

	if (choice == "scalapack")
		return NewtonBackend::SCALAPACK;
#ifdef KADATH_USE_MUMPS
	if (choice == "krylov")
		return NewtonBackend::KRYLOV;
	if (choice == "mumps")
		return NewtonBackend::MUMPS;
#endif
	if (rank == 0)
		std::cerr << "KADATH_LINSOLVE=" << choice << " is not available in this build (use scalapack"
#ifdef KADATH_USE_MUMPS
		          << ", krylov or mumps"
#endif
		          << ")" << std::endl;
	MPI_Barrier(MPI_COMM_WORLD);
	abort();
}

Array<double> newton_update(System_of_eqs& syst, Array<double> const& second, System_of_eqs::SOLVER solver) {
	int rank;
	MPI_Comm_rank (MPI_COMM_WORLD, &rank);
	NewtonBackend const backend =
	    (solver == System_of_eqs::GAUSS_NEWTON) ? NewtonBackend::SCALAPACK : newton_backend_from_env();

	Array<double> X (syst.get_nbr_unknowns());
	switch (backend) {
#ifdef KADATH_USE_MUMPS
		case NewtonBackend::KRYLOV:
			if (rank == 0)
				std::cout << "Newton linear solver: Newton-Krylov (GMRES + MUMPS preconditioner)" << std::endl;
			X = newton_update_krylov(syst, second);
			break;
		case NewtonBackend::MUMPS:
			if (rank == 0)
				std::cout << "Newton linear solver: MUMPS (sparse)" << std::endl;
			X = newton_update_mumps(syst, second);
			break;
#endif
		default:
			if (rank == 0)
				std::cout << "Newton linear solver: ScaLAPACK (dense)" << std::endl;
			X = newton_update_scalapack(syst, second, solver);
	}
	report_memory();
	return X;
}

}  // namespace linear_solvers

bool System_of_eqs::do_newton(double precision, double& error, System_of_eqs::SOLVER solver) {
	// rank and nproc from MPI :
	int rank, nproc;
	MPI_Comm_rank (MPI_COMM_WORLD, &rank);
	MPI_Comm_size (MPI_COMM_WORLD, &nproc);

	vars_to_terms();
	Array<double> second (sec_member());
	Array<double> abs_errs(check_equations());
  error = max(abs_errs);

	if (rank==0) {
    if(abs_errs.nbr != eq_list.size() + eq_int_list.size()) {
      std::cout << "There is a mismatch between the number of eqs and the registered eq strings."
                   "Please check the add_eq routines of your space." << std::endl;
  		//abort();
    }

    double max = abs_errs.data[0];
    int i_max = 0;

  	for (int i=0; i<abs_errs.nbr; i++)
  		if (abs_errs.data[i] > max) {
  			max = abs_errs.data[i];
  			i_max = i;
  		}

		std::cout << "Error init = " << error << ",";

		if(i_max < neq_int && neq_int > 0) {
      auto [ eq, dom, boundary ] = eq_int_list[i_max];
      std::cout << " Eq: " << eq << " Dom: " << dom << " Bounday: " << boundary << std::endl;
		}
		else {
      i_max -= neq_int;
      auto [ eq, dom, boundary ] = eq_list[i_max];
      std::cout << " Eq: " << eq << " Dom: " << dom << " Bounday: " << boundary << std::endl;
		}
	}

	if (error < precision) {
		return true;
	}
	else {

	int m = second.get_size(0);
  int n = nbr_unknowns;

	if (solver == NEWTON_RAPHSON && n != m) {
		cerr << "Number of unknowns is different from the number of equations" << endl;
		cerr << "nbr equations = " << m << endl;
		cerr << "nbr unknowns  = " << n << endl;
		abort();
	}

	if (rank==0) {
    double size = 1.0 * n * m;
		std::cout << "DOF / rank = " << n << " x " << m << " / " << nproc <<  " = " << size / nproc << endl;
	}

	Array<double> X (linear_solvers::newton_update(*this, second, solver));

	auto begin = std::chrono::system_clock::now();

	int conte = 0;
	espace.xx_to_vars_variable_domains (this, X, conte);

	double* old_var_double = (nvar_double==0) ? 0x0 : new double[nvar_double];

	for (int i=0 ; i<nvar_double ; i++)
		old_var_double[i] = *var_double[i];

	Tensor** old_fields = new Tensor* [nvar];
	for (int i=0 ; i<nvar ; i++)
		old_fields[i] = new Tensor(*var[i]);

	xx_to_vars(X, conte);

	for (int i=0 ; i<nvar ; i++)
		*var[i] = *old_fields[i] - *var[i];

	for (int i=0 ; i<nvar_double ; i++)
		*var_double[i] = old_var_double[i] - *var_double[i];

	if (old_var_double!=0x0)
	    delete [] old_var_double;

	for (int i=0 ; i<nvar ; i++)
		delete old_fields[i];

	delete [] old_fields;

  if (rank == 0) {
    std::cout << "Distributing solution: " << elapsed_time(begin) << " seconds" << endl;
  }

	return false;
  }
}

void split_two_d (int target, int& low, int& up) {
      int start = int(sqrt(target));
      low = start;
      up = start;

      bool endloop = (low*up==target) ? true : false;

      bool first = true;
      while (!endloop) {
	    if (!first)
	      low--;
	    else
	      first= false;

	    up = int(target/low);

	if (low*up==target)
	    endloop = true;
      }

}

bool System_of_eqs::do_newton_with_linesearch(double precision, double& error, int ntrymax, double stepmax)
{
   // Numerical recipes 2007, section 9.7.1
   bool res(false);
   int rank;
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);

   Array<double> second(sec_member());  // rhs of J.x = F
   error = max(fabs(second));
   if (rank == 0) cout << "Error init = " << error << endl;
   if (error < precision) res = true;
   else
   {
      int nn(second.get_size(0));
      check_size_VS_unknowns(nn);
      if (rank == 0) cout << "Size of the system " << nn << endl;

      Array<double> p(nn);             // solution of J.x = F
      double fold, f, slope, f2(0.0);
      double lambda(1.0), lambdatmp(0.0), lambda2(0.0);
      const double lambdamin(1.0e-10);
      const double alpha(1.0e-4);

      fold = compute_f(second);    // old value of f = 0.5*F.F
      slope = -2.0*fold;
      check_negative(slope);
      compute_p(p, second, nn);    // all the parallel inversion of the jacobian is crammed into this function
      double pmax(max(p));
      if (rank == 0) cout << "max(newton step) = " << pmax << endl;

      if (pmax > stepmax) p *= stepmax/pmax;      // rescale in case p is too large
      vector<double> old_var_double, p_var_double;
      vector<Tensor> old_var_fields, p_var_fields;
      compute_old_and_var(p, old_var_double, old_var_fields, p_var_double, p_var_fields);
      for (int itry(1) ; itry <= ntrymax ; ++itry)
      {
         update_fields(lambda, old_var_double, old_var_fields, p_var_double, p_var_fields);    // var now updated
         second = sec_member();
         f = compute_f(second);
         if (f <= 5.0e-13) break; // in double precision, linesearch in spoiled by numerical error as soon as the error is close to 1.0e-7 (f \propto error^2)
         if (f <= fold + alpha*lambda*slope or lambda <= lambdamin) break;
         if (rank == 0) cout << "line search " << itry << " ====> ";
         if (lambda == 1.0) lambdatmp = -0.5*slope/(f - fold - slope);
         else
         {
            double rhs1(f - fold - lambda*slope);
            double rhs2(f2 - fold - lambda2*slope);
            double a((rhs1/pow(lambda,2) - rhs2/pow(lambda2,2))/(lambda-lambda2));
            double b((-lambda2*rhs1/pow(lambda,2) + lambda*rhs2/pow(lambda2,2))/(lambda-lambda2));
            if (a == 0.0) lambdatmp = -0.5*slope/b;
            double delta(pow(b,2) - 3.0*a*slope);
            check_positive(delta);
            lambdatmp = (-b + sqrt(delta))/(3.0*a);
            if (lambdatmp > 0.5*lambda) lambdatmp = 0.5*lambda;
         }
         lambda2 = lambda;
         f2 = f;
         lambda = max(lambdatmp, 0.1*lambda);
         if (rank == 0) cout << "lambda = " << lambda << endl;
         if (itry == ntrymax) update_fields(lambda, old_var_double, old_var_fields, p_var_double, p_var_fields);
      }
   }
   return res;
}

bool System_of_eqs::do_newton_seq(double precision, double &error)
{
   clock_t begin, end;
   vars_to_terms();
   Array<double> second(sec_member());
   error = max(fabs(second));
   cout << "Error init = " << error << endl;
   if (error<precision) return true;
   int nn(second.get_size(0));
   if (nbr_unknowns!=nn)
   {
      cerr << "N unknowns  = " << nbr_unknowns << endl;
      cerr << "N equations = " << nn << endl;
      abort();
   }
   cout << "Size = " << nn << endl;
   cout << "Progression computation : ";
   cout.flush();
   begin = clock();
   Array<double> jx(nn);
   Matrice ope(nn,nn);
   for (int col(nn-1) ; col >= 0 ; col--)
   {
      if ((col != 0) && (col%int(nn/10) == 0))
      {
         cout << "*";
         cout.flush();
      }
      jx = do_col_J(col);
      for (int line(0) ; line < nn ; line++) ope.set(line,col) = jx(line);
   }
   cout << endl;
   end = clock();
   cout << "Loading the matrix : " << static_cast<double>(end - begin)/CLOCKS_PER_SEC << " seconds" << endl;
   begin = clock();
   ope.set_lu();
   Array<double> xx(ope.solve(second));

   end = clock();
   cout << "Inverting the matrix : " << static_cast<double>(end - begin)/CLOCKS_PER_SEC << " seconds" << endl;
   int conte(0);
   espace.xx_to_vars_variable_domains(this, xx, conte);
   double* old_var_double(new double[nvar_double]);
   for (int i(0) ; i < nvar_double ; ++i) old_var_double[i] = *var_double[i];
   Tensor** old_fields(new Tensor* [nvar]);
   for (int i(0) ; i < nvar ; i++) old_fields[i] = new Tensor(*var[i]);
   xx_to_vars(xx, conte);
   for (int i(0) ; i < nvar_double ; i++) *var_double[i] = old_var_double[i] - (*var_double[i]);
   for (int i(0) ; i < nvar ; i++) *var[i] = *old_fields[i] - (*var[i]);
   delete [] old_var_double;
   for (int i(0) ; i<nvar ; i++) delete old_fields[i];
   delete [] old_fields;
   return false;
}



void System_of_eqs::check_positive(double delta)
{
   if (delta < 0.0)
   {
      cerr << "roundoff error in do_newton_with_linesearch" << endl;
      abort();
   }
}

void System_of_eqs::check_negative(double delta)
{
   if (delta >= 0.0)
   {
      cerr << "roundoff error in do_newton_with_linesearch" << endl;
      abort();
   }
}

double System_of_eqs::compute_f(Array<double> const& second)
{
   double f(0.0);
   for (int i(0) ; i < second.get_nbr() ; ++i) f += pow(second.get_data()[i],2) ;
   return f;
}

void System_of_eqs::check_size_VS_unknowns(int n)
{
   if (n != nbr_unknowns)
   {
      cerr << "Number of unknowns is different from the number of equations" << endl;
      cerr << "nbr equations = " << n << endl;
      cerr << "nbr unknowns  = " << nbr_unknowns << endl;
      abort();
   }
}

void System_of_eqs::check_bsize(int bsize)
{
   if (bsize < 1)
   {
      cerr << "Too many processrs in do_newton" << endl;
      abort();
   }
}

void System_of_eqs::compute_matloc(Array<double>& matloc_in, int nn, int bsize)
{
   int rank, nproc;
   MPI_Comm_rank (MPI_COMM_WORLD, &rank);
   MPI_Comm_size (MPI_COMM_WORLD, &nproc);

   int start(bsize*rank);
   bool done(false);
   int current(0);
   while (!done)
   {
      for (int i(0) ; i < bsize ; ++i)
         if (start+i < nn)
         {
            Array<double> column(do_col_J(start+i));
            for (int j(0) ; j < nn ; ++j) matloc_in.set(current,j) = column(j);
            current++;
         }
      start += nproc*bsize;
      if (start >= nn) done = true;
   }
}

void System_of_eqs::translate_second_member(Array<double>& secloc, Array<double> const& second, int nn, int bsize, int nprow, int myrow, int mycol)
{
   for (int row(0) ; row < nn ; ++row)
   {
      int pi(div(row/bsize, nprow).rem);
      int li(int(row/(nprow*bsize)));
      int xi(div(row, bsize).rem);
      if ((pi == myrow) and (mycol == 0)) secloc.set(li*bsize+xi) = second(row);
   }
}

void System_of_eqs::get_global_solution(Array<double>& auxi, Array<double> const& secloc, int nn, int bsize, int nprow, int myrow, int mycol)
{
   auxi = 0.0;
   for (int row(0) ; row < nn ; ++row)
   {
      int pi(div(row/bsize, nprow).rem);
      int li(int(row/(nprow*bsize)));
      int xi(div(row, bsize).rem);
      if ((pi == myrow) and (mycol == 0)) auxi.set(row) = secloc(li*bsize+xi);
   }
}

void System_of_eqs::update_fields(double lambda, vector<double> const& old_var_double, vector<Tensor> const& old_var_fields, vector<double> const& p_var_double, vector<Tensor> const& p_var_fields)
{
   int rank;
   MPI_Comm_rank (MPI_COMM_WORLD, &rank);
   if (nvar_double > 0) for (int i(0) ; i < nvar_double ; ++i) *var_double[i] = old_var_double[i] - lambda*p_var_double[i];
   for (int i(0) ; i < nvar ; ++i) *var[i] = old_var_fields[i] - lambda*p_var_fields[i];
}

void System_of_eqs::compute_old_and_var(Array<double> const& xx, vector<double>& old_var_double, vector<Tensor>& old_var_fields, vector<double>& p_var_double, vector<Tensor>& p_var_fields)
{
   int rank;
   MPI_Comm_rank (MPI_COMM_WORLD, &rank);
   int conte(0);
   espace.xx_to_vars_variable_domains(this, xx, conte);
   if (nvar_double > 0) for (int i(0) ; i < nvar_double ; ++i) old_var_double.push_back(*var_double[i]);
   for (int i(0) ; i < nvar ; ++i) old_var_fields.push_back(Tensor(*var[i]));
   xx_to_vars(xx, conte);
   if (nvar_double > 0) for (int i(0) ; i < nvar_double ; ++i) p_var_double.push_back(*var_double[i]);
   for (int i(0) ; i < nvar ; ++i) p_var_fields.push_back(Tensor(*var[i]));
}

void System_of_eqs::compute_p(Array<double>& xx, Array<double> const& second, int nn)
{
   int bsize(64);
   // rank and nproc from MPI :
   int rank, nproc;
   MPI_Comm_rank (MPI_COMM_WORLD, &rank);
   MPI_Comm_size (MPI_COMM_WORLD, &nproc);

   // Computation in a 1D distributed matrice
   int zero(0);
   int ictxt_in;
   int nprow_in(1);
   int npcol_in(nproc);
   sl_init_(&ictxt_in, &nprow_in, &npcol_in);

   // Get my row and mycol
   int myrow_in, mycol_in;
   blacs_gridinfo_ (&ictxt_in, &nprow_in, &npcol_in, &myrow_in, &mycol_in);
   while (bsize*nproc > nn) bsize = div(bsize,2).quot;
   check_bsize(bsize);
   int nrowloc_in(numroc_(&nn, &bsize, &myrow_in, &zero, &nprow_in));
   int ncolloc_in(numroc_(&nn, &bsize, &mycol_in, &zero, &npcol_in));
   Array<double> matloc_in(ncolloc_in, nrowloc_in);
   compute_matloc(matloc_in, nn, bsize);

    // Descriptor of the matrix :
   Array<int> descamat_in(9);
   int info;
   descinit_(descamat_in.set_data(), &nn, &nn, &bsize, &bsize, &zero, &zero, &ictxt_in, &nrowloc_in, &info);

   // Wait for everybody
   MPI_Barrier(MPI_COMM_WORLD);

   // Now translate to a 2D cyclic decomposition
   int npcol, nprow;
   int ictxt;
   int myrow, mycol;
   split_two_d(nproc, npcol, nprow);
   sl_init_(&ictxt, &nprow, &npcol);
   blacs_gridinfo_(&ictxt, &nprow, &npcol, &myrow, &mycol);
   int nrowloc(numroc_(&nn, &bsize, &myrow, &zero, &nprow));
   int ncolloc(numroc_(&nn, &bsize, &mycol, &zero, &npcol));
   Array<double> matloc(ncolloc, nrowloc);
   Array<int> descamat(9);
   descinit_(descamat.set_data(), &nn, &nn, &bsize, &bsize, &zero, &zero, &ictxt, &nrowloc, &info);
   Cpdgemr2d(nn, nn, matloc_in.set_data(), 1, 1, descamat_in.set_data(), matloc.set_data(), 1, 1, descamat.set_data(), ictxt);
   matloc_in.delete_data();

   // Translate the second member :
   Array<double> secloc(nrowloc);
   translate_second_member(secloc, second, nn, bsize, nprow, myrow, mycol);

   // Descriptor of the second member
   Array<int> descsec(9);
   int one(1);
   descinit_(descsec.set_data(), &nn, &one, &bsize, &bsize, &zero, &zero, &ictxt, &nrowloc, &info);

   // Inversion
   Array<int> ipiv(nn);
   pdgesv_(&nn, &one, matloc.set_data(), &one, &one, descamat.set_data(), ipiv.set_data(), secloc.set_data(), &one, &one, descsec.set_data(), &info);

   // Get the global solution
   Array<double> auxi(nn);
   get_global_solution(auxi, secloc, nn, bsize, nprow, myrow, mycol);
   MPI_Allreduce(auxi.set_data(), xx.set_data(), nn, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
   blacs_gridexit_(&ictxt);
   blacs_gridexit_(&ictxt_in);
}
}

