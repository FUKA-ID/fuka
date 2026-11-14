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
namespace Kadath {
Ope_dtime::Ope_dtime (const System_of_eqs* zesys, Ope_eq* target) : Ope_eq(zesys, target->get_dom(), 1) {
	parts[0] = target ;
}

Ope_dtime::~Ope_dtime() {
}

Term_eq Ope_dtime::compute() const {

	Term_eq target (parts[0]->action()) ;
	int valence = target.val_t->get_valence() ;
	// Check it is a tensor
	if (target.type_data != TERM_T) {
		cerr << "Ope_dtime only defined with respect for a tensor" << endl ;
		abort() ;
	}

	// The value	
	Tensor resval (*target.val_t, false) ;

	for (int i=0 ; i<target.val_t->get_n_comp() ; i++) {
		Array<int> ind (target.val_t->indices(i)) ;
		Val_domain value ((*target.val_t)(ind)(dom)) ;
		if (value.check_if_zero())
			resval.set(ind).set_domain(dom).set_zero() ;
		else {
			Val_domain auxi (value.get_domain()->dtime(value)) ;
			resval.set(ind).set_domain(dom) = auxi ;
		}
	}

	// Put name indices :
		if (target.val_t->is_name_affected()) {
			resval.set_name_affected() ;
			for (int ncmp = 0 ; ncmp<valence ; ncmp++)
				resval.set_name_ind (ncmp, target.val_t->get_name_ind()[ncmp]) ;
		}

	if (target.der_t!=0x0) {
		Tensor resder (*target.der_t, false) ;
		for (int i=0 ; i<target.der_t->get_n_comp() ; i++) {
			Array<int> ind (target.der_t->indices(i)) ;
			Val_domain value ((*target.der_t)(ind)(dom)) ;
			if (value.check_if_zero())
				resder.set(ind).set_domain(dom).set_zero() ;
			else {
				Val_domain auxi (value.get_domain()->dtime(value)) ;
				resder.set(ind).set_domain(dom) = auxi ;
			}
			}
		// Put name indices :
		if (target.der_t->is_name_affected()) {
			resder.set_name_affected() ;
			for (int ncmp = 0 ; ncmp<valence ; ncmp++)
				resder.set_name_ind (ncmp, target.der_t->get_name_ind()[ncmp]) ;
		}
		Term_eq res (dom, resval, resder) ;
		return res ;
	}
	else {
		Term_eq res (dom, resval) ;
		return res ;
	}
}}
