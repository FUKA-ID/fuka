/*
    Copyright 2017 Philippe Grandclement

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

#include "headcpp.hpp"

#include "spheric.hpp"
#include "scalar.hpp"
#include "tensor.hpp"

namespace Kadath {
int Domain_nucleus::nbr_conditions_val_domain_boundary_vr (const Val_domain& so) const {
	
	int res = 0 ;
	
	for (int k=0 ; k<nbr_coefs(2) ; k++) 
	  for (int j=0 ; j<nbr_coefs(1) ; j++) {
		bool indic = true ;
		// True coef in phi ?
		if ((k==1) || (k==nbr_coefs(2)-1))
			indic = false ;
		// Get base in theta :
		int baset = (*so.get_base().bases_1d[1]) (k) ;
		if (is_null_theta_mode (baset, k/2, j, nbr_coefs(1)))
			indic = false ;
		switch (baset) {
					case COS_EVEN:
						if ((j==0) && (k!=0))
							indic = false ;
						break ;
					case SIN_ODD:
						if (j==nbr_coefs(1)-1)
							indic = false ;
						break ;
					default:
						cerr << "Unknow theta basis in Domain_nucleus::nbr_conditions_val_domain_boundary_vr" << endl ;
						abort() ;
		}
	
		
		if (indic)
			res ++ ;
	}

	return res ;
}

int Domain_nucleus::nbr_conditions_val_domain_boundary_vt (const Val_domain& so) const {
	
	int res = 0 ;
	
	for (int k=0 ; k<nbr_coefs(2) ; k++)
	  for (int j=0 ; j<nbr_coefs(1) ; j++)  {
		bool indic = true ;
		// True coef in phi ?
		if ((k==1) || (k==nbr_coefs(2)-1))
			indic = false ;
		// Get base in theta :
		int baset = (*so.get_base().bases_1d[1]) (k) ;
		if (is_null_theta_mode (baset, k/2, j, nbr_coefs(1)))
			indic = false ;
		switch (baset) {
					case SIN_EVEN:
						if ((j==nbr_coefs(1)-1) || (j==0))
						  indic= false ;
						break ;
					case COS_ODD:
						if (j==nbr_coefs(1)-1)
							indic = false ;
						if ((j==0) && (k>3))
							indic = false ;
						break ;
					default:
						cerr << "Unknow theta basis in Domain_nucleus::nbr_conditions_val_domain_boundary_vt" << endl ;
						abort() ;
		}
	
		if (indic)
			res ++ ;
	}
	
	return res ;
}

int Domain_nucleus::nbr_conditions_val_domain_boundary_vp (const Val_domain& so) const {
	
	int res = 0 ;
	
	for (int k=0 ; k<nbr_coefs(2) ; k++)
	  for (int j=0 ; j<nbr_coefs(1) ; j++) 
	  {
		bool indic = true ;
		// True coef in phi ?
		if ((k==1) || (k==nbr_coefs(2)-1))
			indic = false ;
		// Get base in theta :
		int baset = (*so.get_base().bases_1d[1]) (k) ;
		if (is_null_theta_mode (baset, k/2, j, nbr_coefs(1)))
			indic = false ;
		switch (baset) {
					case COS_EVEN:
					  if ((k>3) && (j==0))
						  indic = false ;
						break ;
					case SIN_ODD:
						if (j==nbr_coefs(1)-1)
							indic = false ;
						break ;
					default:
						cerr << "Unknow theta basis in Domain_nucleus::nbr_conditions_val_domain_boundary_vp" << endl ;
						abort() ;
		}
	
		if (indic)
			res ++ ;
	}
	
	return res ;
}
int Domain_nucleus::nbr_conditions_val_domain_boundary (const Val_domain& so, int mlim) const {
	
	int res = 0 ;
	int kmin = 2*mlim + 2 ;

	for (int k=0 ; k<nbr_coefs(2) ; k++) 
		for (int j=0 ; j<nbr_coefs(1) ; j++) {
		bool indic = true ;
		// True coef in phi ?
		if ((k==1) || (k==nbr_coefs(2)-1))
			indic = false ;
		// Get base in theta :
		int baset = (*so.get_base().bases_1d[1])(k) ;
		if (is_null_theta_mode (baset, k/2, j, nbr_coefs(1)))
			indic = false ;
		switch (baset) {
					case COS_EVEN:
						if ((j==0) && (k>=kmin))
							indic = false ;
						break ;
					case COS_ODD:
						if ((j==nbr_coefs(1)-1) || ((j==0) && (k>=kmin)))
							indic = false ;
						break ;
					case SIN_EVEN:
						if (((j==1) && (k>=kmin+2)) || (j==0) || (j==nbr_coefs(1)-1)) 
							indic = false  ;
						break ;
					case SIN_ODD:
						if (((j==0) && (k>=kmin+2)) || (j==nbr_coefs(1)-1))
							indic = false ;
						break ;
					default:
						cerr << "Unknow theta basis in Domain_nucleus::nbr_conditions_val_boundary" << endl ;
						abort() ;
		}

		if (indic)
			res ++ ;
	}
	return res ;
}

Array<int> Domain_nucleus::nbr_conditions_boundary (const Tensor& tt, int dom, int bound, int n_cmp, Array<int>** p_cmp) const {

	// Check boundary
	if (bound!=OUTER_BC) {
		cerr << "Unknown boundary in Domain_nucleus::nbr_conditions_boundary" << endl ;
		abort() ;
	}

	int size = (n_cmp==-1) ? tt.get_n_comp() : n_cmp ;
	Array<int> res (size) ;
	int val = tt.get_valence() ;
	switch (val) {
		case 0 :
			if (!tt.is_m_order_affected())
			  res.set(0) = nbr_conditions_val_domain_boundary (tt()(dom), 0) ;
			else
			  res.set(0) = nbr_conditions_val_domain_boundary (tt()(dom), tt.get_parameters()->get_m_order()) ;
			break ;
		case 1 : {
			bool found = false ;
			// Cartesian basis
			if (tt.get_basis().get_basis(dom)==CARTESIAN_BASIS) {
				if (n_cmp==-1) {
					res.set(0) = nbr_conditions_val_domain_boundary (tt(1)(dom), 0) ;
					res.set(1) = nbr_conditions_val_domain_boundary (tt(2)(dom), 0) ;
					res.set(2) = nbr_conditions_val_domain_boundary (tt(3)(dom), 0) ;
				}
				else for (int i=0 ; i<n_cmp ; i++) {
					if ((*p_cmp[i])(0)==1)
						res.set(i) = nbr_conditions_val_domain_boundary (tt(1)(dom), 0) ;
					if ((*p_cmp[i])(0)==2)
						res.set(i) = nbr_conditions_val_domain_boundary (tt(2)(dom), 0) ;
					if ((*p_cmp[i])(0)==3)
						res.set(i) = nbr_conditions_val_domain_boundary (tt(3)(dom), 0) ;
				}
				found = true ;
			}
			// Spherical coordinates
			if (tt.get_basis().get_basis(dom)==SPHERICAL_BASIS) {
				if (n_cmp==-1) {
					res.set(0) = nbr_conditions_val_domain_boundary_vr (tt(1)(dom)) ;
					res.set(1) = nbr_conditions_val_domain_boundary_vt (tt(2)(dom)) ;
					res.set(2) = nbr_conditions_val_domain_boundary_vp (tt(3)(dom)) ;
				}
				else for (int i=0 ; i<n_cmp ; i++) {
					if ((*p_cmp[i])(0)==1)
						res.set(i) = nbr_conditions_val_domain_boundary_vr (tt(1)(dom)) ;
					if ((*p_cmp[i])(0)==2)
						res.set(i) = nbr_conditions_val_domain_boundary_vt (tt(2)(dom)) ;
					if ((*p_cmp[i])(0)==3)
						res.set(i) = nbr_conditions_val_domain_boundary_vp (tt(3)(dom)) ;
				}
				found = true ;
			}
			if (!found) {
				cerr << "Unknown type of vector Domain_nucleus::nbr_conditions_boundary" << endl ;
				abort() ;
			}
		}
			break ;
		case 2 : {
			bool found = false ;
			// Cartesian basis and symetric
			if ((tt.get_basis().get_basis(dom)==CARTESIAN_BASIS) && (tt.get_n_comp()==6)) {
				if (n_cmp==-1) {
					res.set(0) = nbr_conditions_val_domain_boundary (tt(1,1)(dom), 0) ;
					res.set(1) = nbr_conditions_val_domain_boundary (tt(1,2)(dom), 0) ;
					res.set(2) = nbr_conditions_val_domain_boundary (tt(1,3)(dom), 0) ;
					res.set(3) = nbr_conditions_val_domain_boundary (tt(2,2)(dom), 0) ;
					res.set(4) = nbr_conditions_val_domain_boundary (tt(2,3)(dom), 0) ;
					res.set(5) = nbr_conditions_val_domain_boundary (tt(3,3)(dom), 0) ;
				}
				else for (int i=0 ; i<n_cmp ; i++) {
					if (((*p_cmp[i])(0)==1) && ((*p_cmp[i])(1)==1))
						res.set(i) = nbr_conditions_val_domain_boundary (tt(1, 1)(dom), 0) ;
					if (((*p_cmp[i])(0)==1) && ((*p_cmp[i])(1)==2))
						res.set(i) = nbr_conditions_val_domain_boundary (tt(1, 2)(dom), 0) ;
					if (((*p_cmp[i])(0)==1) && ((*p_cmp[i])(1)==3))
						res.set(i) = nbr_conditions_val_domain_boundary (tt(1, 3)(dom), 0) ;
					if (((*p_cmp[i])(0)==2) && ((*p_cmp[i])(1)==2))
						res.set(i) = nbr_conditions_val_domain_boundary (tt(2, 2)(dom), 0) ;
					if (((*p_cmp[i])(0)==2) && ((*p_cmp[i])(1)==3))
						res.set(i) = nbr_conditions_val_domain_boundary (tt(2, 3)(dom), 0) ;
					if (((*p_cmp[i])(0)==3) && ((*p_cmp[i])(1)==3))
						res.set(i) = nbr_conditions_val_domain_boundary (tt(3, 3)(dom), 0) ;
				}
				found = true ;
			}
			// Cartesian basis and not symetric
			if ((tt.get_basis().get_basis(dom)==CARTESIAN_BASIS) && (tt.get_n_comp()==9)) {
				if (n_cmp==-1) {
					res.set(0) = nbr_conditions_val_domain_boundary (tt(1,1)(dom), 0) ;
					res.set(1) = nbr_conditions_val_domain_boundary (tt(1,2)(dom), 0) ;
					res.set(2) = nbr_conditions_val_domain_boundary (tt(1,3)(dom), 0) ;
					res.set(3) = nbr_conditions_val_domain_boundary (tt(2,1)(dom), 0) ;
					res.set(4) = nbr_conditions_val_domain_boundary (tt(2,2)(dom), 0) ;
					res.set(5) = nbr_conditions_val_domain_boundary (tt(2,3)(dom), 0) ;	
					res.set(6) = nbr_conditions_val_domain_boundary (tt(3,1)(dom), 0) ;
					res.set(7) = nbr_conditions_val_domain_boundary (tt(3,2)(dom), 0) ;
					res.set(8) = nbr_conditions_val_domain_boundary (tt(3,3)(dom), 0) ;
				}
				else for (int i=0 ; i<n_cmp ; i++) {
					if (((*p_cmp[i])(0)==1) && ((*p_cmp[i])(1)==1))
						res.set(i) = nbr_conditions_val_domain_boundary (tt(1, 1)(dom), 0) ;
					if (((*p_cmp[i])(0)==1) && ((*p_cmp[i])(1)==2))
						res.set(i) = nbr_conditions_val_domain_boundary (tt(1, 2)(dom), 0) ;
					if (((*p_cmp[i])(0)==1) && ((*p_cmp[i])(1)==3))
						res.set(i) = nbr_conditions_val_domain_boundary (tt(1, 3)(dom), 0) ;	
					if (((*p_cmp[i])(0)==1) && ((*p_cmp[i])(1)==1))
						res.set(i) = nbr_conditions_val_domain_boundary (tt(2, 1)(dom), 0) ;
					if (((*p_cmp[i])(0)==2) && ((*p_cmp[i])(1)==2))
						res.set(i) = nbr_conditions_val_domain_boundary (tt(2, 2)(dom), 0) ;
					if (((*p_cmp[i])(0)==2) && ((*p_cmp[i])(1)==3))
						res.set(i) = nbr_conditions_val_domain_boundary (tt(2, 3)(dom), 0) ;
					if (((*p_cmp[i])(0)==3) && ((*p_cmp[i])(1)==1))
						res.set(i) = nbr_conditions_val_domain_boundary (tt(3, 1)(dom), 0) ;
					if (((*p_cmp[i])(0)==3) && ((*p_cmp[i])(1)==2))
						res.set(i) = nbr_conditions_val_domain_boundary (tt(3, 2)(dom), 0) ;
					if (((*p_cmp[i])(0)==3) && ((*p_cmp[i])(1)==3))
						res.set(i) = nbr_conditions_val_domain_boundary (tt(3, 3)(dom), 0) ;
				}
				found = true ;
			}
			// Spherical coordinates and symetric
			if ((tt.get_basis().get_basis(dom)==SPHERICAL_BASIS) && (tt.get_n_comp()==6)) {
				if (n_cmp==-1) {
					res.set(0) = nbr_conditions_val_domain_boundary (tt(1,1)(dom), 0) ;
					res.set(1) = nbr_conditions_val_domain_boundary (tt(1,2)(dom), 1) ;
					res.set(2) = nbr_conditions_val_domain_boundary (tt(1,3)(dom), 1) ;
					res.set(3) = nbr_conditions_val_domain_boundary (tt(2,2)(dom), 2) ;
					res.set(4) = nbr_conditions_val_domain_boundary (tt(2,3)(dom), 2) ;
					res.set(5) = nbr_conditions_val_domain_boundary (tt(3,3)(dom), 2) ;
				}
				else for (int i=0 ; i<n_cmp ; i++) {
					if (((*p_cmp[i])(0)==1) && ((*p_cmp[i])(1)==1))
						res.set(i) = nbr_conditions_val_domain_boundary (tt(1, 1)(dom), 0) ;
					if (((*p_cmp[i])(0)==1) && ((*p_cmp[i])(1)==2))
						res.set(i) = nbr_conditions_val_domain_boundary (tt(1, 2)(dom), 1) ;
					if (((*p_cmp[i])(0)==1) && ((*p_cmp[i])(1)==3))
						res.set(i) = nbr_conditions_val_domain_boundary (tt(1, 3)(dom), 1) ;
					if (((*p_cmp[i])(0)==2) && ((*p_cmp[i])(1)==2))
						res.set(i) = nbr_conditions_val_domain_boundary (tt(2, 2)(dom), 2) ;
					if (((*p_cmp[i])(0)==2) && ((*p_cmp[i])(1)==3))
						res.set(i) = nbr_conditions_val_domain_boundary (tt(2, 3)(dom), 2) ;
					if (((*p_cmp[i])(0)==3) && ((*p_cmp[i])(1)==3))
						res.set(i) = nbr_conditions_val_domain_boundary (tt(3, 3)(dom), 2) ;
				}
				found = true ;
			}
			// Spherical coordinates and not symetric
			if ((tt.get_basis().get_basis(dom)==SPHERICAL_BASIS) && (tt.get_n_comp()==9)) {
				if (n_cmp==-1) {
					res.set(0) = nbr_conditions_val_domain_boundary (tt(1,1)(dom), 0) ;
					res.set(1) = nbr_conditions_val_domain_boundary (tt(1,2)(dom), 1) ;
					res.set(2) = nbr_conditions_val_domain_boundary (tt(1,3)(dom), 1) ;
					res.set(3) = nbr_conditions_val_domain_boundary (tt(2,1)(dom), 1) ;
					res.set(4) = nbr_conditions_val_domain_boundary (tt(2,2)(dom), 2) ;
					res.set(5) = nbr_conditions_val_domain_boundary (tt(2,3)(dom), 2) ;
					res.set(6) = nbr_conditions_val_domain_boundary (tt(3,1)(dom), 1) ;
					res.set(7) = nbr_conditions_val_domain_boundary (tt(3,2)(dom), 2) ;
					res.set(8) = nbr_conditions_val_domain_boundary (tt(3,3)(dom), 2) ;
					
				}
				else for (int i=0 ; i<n_cmp ; i++) {
					if (((*p_cmp[i])(0)==1) && ((*p_cmp[i])(1)==1))
						res.set(i) = nbr_conditions_val_domain_boundary (tt(1, 1)(dom), 0) ;
					if (((*p_cmp[i])(0)==1) && ((*p_cmp[i])(1)==2))
						res.set(i) = nbr_conditions_val_domain_boundary (tt(1, 2)(dom), 1) ;
					if (((*p_cmp[i])(0)==1) && ((*p_cmp[i])(1)==3))
						res.set(i) = nbr_conditions_val_domain_boundary (tt(1, 3)(dom), 1) ;	
					if (((*p_cmp[i])(0)==2) && ((*p_cmp[i])(1)==1))
						res.set(i) = nbr_conditions_val_domain_boundary (tt(2, 1)(dom), 1) ;
					if (((*p_cmp[i])(0)==2) && ((*p_cmp[i])(1)==2))
						res.set(i) = nbr_conditions_val_domain_boundary (tt(2, 2)(dom), 2) ;
					if (((*p_cmp[i])(0)==2) && ((*p_cmp[i])(1)==3))
						res.set(i) = nbr_conditions_val_domain_boundary (tt(2, 3)(dom), 2) ;
					if (((*p_cmp[i])(0)==3) && ((*p_cmp[i])(1)==1))
						res.set(i) = nbr_conditions_val_domain_boundary (tt(3, 1)(dom), 1) ;
					if (((*p_cmp[i])(0)==3) && ((*p_cmp[i])(1)==2))
						res.set(i) = nbr_conditions_val_domain_boundary (tt(3, 2)(dom), 2) ;
					if (((*p_cmp[i])(0)==3) && ((*p_cmp[i])(1)==3))
						res.set(i) = nbr_conditions_val_domain_boundary (tt(3, 3)(dom), 2) ;
				}
				found = true ;
			}
			if (!found) {
				cerr << "Unknown type of 2-tensor Domain_nucleus::nbr_conditions_boundary" << endl ;
				abort() ;
			}
		}
			break ;
		default :
			cerr << "Valence " << val << " not implemented in Domain_nucleus::nbr_conditions_boundary" << endl ;
			break ;
	}
	return res ;
}}
