// clang-format off
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

#include "ope_eq.hpp"
#include "scalar.hpp"
#include "tensor.hpp"
namespace Kadath {
Ope_change_basis::Ope_change_basis (const System_of_eqs* zesys, int base, Ope_eq* target) : Ope_eq(zesys, target->get_dom(), 1), target_basis(base) {
	parts[0] = target ;
}

Ope_change_basis::~Ope_change_basis() {
}

Term_eq Ope_change_basis::compute() const {

	Term_eq target (parts[0]->action()) ;
	// Check it is a tensor
	if (target.type_data != TERM_T) {
		cerr << "Ope_dn only defined with respect for a tensor" << endl ;
		abort() ;
	}

	switch (target_basis) {
	  case SPHERICAL_BASIS: {
	    Tensor res_val (target.val_t->get_space().get_domain(dom)->change_basis_cart_to_spher(dom, *target.val_t)) ;
	    if (target.der_t==0x0) {
		return Term_eq(dom, res_val) ;
	    }
	    else {
	      Tensor res_der (target.val_t->get_space().get_domain(dom)->change_basis_cart_to_spher(dom, *target.der_t)) ;
	      return Term_eq (dom, res_val, res_der) ;
	    }
	  }
	  case CARTESIAN_BASIS: {
	    Tensor res_val (target.val_t->get_space().get_domain(dom)->change_basis_spher_to_cart(dom, *target.val_t)) ;
	    if (target.der_t==0x0) {
		return Term_eq(dom, res_val) ;
	    }
	    else {
	      Tensor res_der (target.val_t->get_space().get_domain(dom)->change_basis_spher_to_cart(dom, *target.der_t)) ;
	      return Term_eq (dom, res_val, res_der) ;
	    }
	  }
	  default: 
	    cerr << "Unknown target tensorial basis in Ope_change_basis::action" << endl ;
	    abort() ;
	}
}}
