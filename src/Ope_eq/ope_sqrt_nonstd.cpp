// clang-format off
/*
    Copyright 2018 Philippe Grandclement

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

#include "ope_eq.hpp"
#include "scalar.hpp"
namespace Kadath {
Ope_sqrt_nonstd::Ope_sqrt_nonstd (const System_of_eqs* zesys, Ope_eq* target) : Ope_eq(zesys, target->get_dom(), 1) {
	parts[0] = target ;
}

Ope_sqrt_nonstd::~Ope_sqrt_nonstd() {
}

Term_eq Ope_sqrt_nonstd::compute() const {

	Term_eq target (parts[0]->action()) ;
	switch (target.type_data) {
	    case TERM_T : {
 
	    // Check it is a scalar
	    if (target.val_t->get_valence() != 0) {
		cerr << "Ope_sqrt_nonstd only defined with respect for a scalar" << endl ;
		abort() ;
	    }
	    

	// To get the base
	Val_domain rho (target.val_t->get_space().get_domain(dom)) ;
	rho= 1 ;
	rho.std_base() ;
	rho = target.val_t->get_space().get_domain(dom)->mult_r(target.val_t->get_space().get_domain(dom)->mult_sin_theta(rho)) ;

	    // The value	
	Tensor resval (*target.val_t, false) ;

	for (int i=0 ; i<target.val_t->get_n_comp() ; i++) {
		Array<int> ind (target.val_t->indices(i)) ;
		Val_domain value ((*target.val_t)(ind)(dom)) ;
		if (value.check_if_zero())
			resval.set(ind).set_domain(dom).set_zero() ;
		else {
			resval.set(ind).set_domain(dom) = sqrt(value) ;
			// Force the base
			resval.set(ind).set_domain(dom).set_base() = rho.get_base() ;
		}
	}

	if (target.der_t!=0x0) {
		Tensor resder (*target.der_t, false) ;
		for (int i=0 ; i<target.der_t->get_n_comp() ; i++) {
			Array<int> ind (target.der_t->indices(i)) ;
			Val_domain valder ((*target.der_t)(ind)(dom)) ;
			Val_domain value ((*target.val_t)(ind)(dom)) ;
			if (valder.check_if_zero())
				resder.set(ind).set_domain(dom).set_zero() ;
			else {
				resder.set(ind).set_domain(dom) = target.val_t->get_space().get_domain(dom)->div_sin_theta(valder)/2./
					target.val_t->get_space().get_domain(dom)->div_sin_theta(resval(ind)(dom)) ;
			}
			}
		Term_eq res (dom, resval, resder) ;
		return res ;
	}
	else {
		Term_eq res (dom, resval) ;
		return res ;
	}
	}
	break ;
	case TERM_D : {
	      if (target.der_d==0x0) {
		  Term_eq res (dom, sqrt(*target.val_d)) ;
		  return res ;
	      }
	      else {
		  Term_eq res (dom, sqrt(*target.val_d), (*target.der_d)/2./sqrt(*target.val_d)) ;
		  return res ;
	      }
	  }
	  break ;
	default : {
	  cerr << "Unknown storage in Term_eq..." << endl ;
	  abort() ;
	}
      }
    cerr << "Warning should not be here in Ope_sqrt_nonstd::action..." << endl ;
    abort() ;
}
}
