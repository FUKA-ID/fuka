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
#include "tensor.hpp"
#include "system_of_eqs.hpp"
#include "name_tools.hpp"
namespace Kadath {
// For tensors
Ope_id::Ope_id(const System_of_eqs* zesys, const Term_eq* tt, int val, char* name, Array<int>* ttype) :
		 Ope_eq(zesys, tt->get_dom(), 0), target(tt),
		 valence (val), name_ind(name), type_ind(ttype) {
		need_sum = false ;
		for (int i=0 ; i<valence ; i++)
			for (int j=i+1 ; j<valence ; j++)
				if (name_ind[i]==name_ind[j])
					need_sum = true ;
}

// For scalars and double
Ope_id::Ope_id(const System_of_eqs* zesys, const Term_eq* tt) :
		 Ope_eq(zesys, tt->get_dom(), 0), target(tt),
		 valence (0), name_ind(0x0), type_ind(0x0), need_sum(false) {
}

Ope_id::~Ope_id() {
	if (name_ind!=0x0)
		delete [] name_ind ;
	if (type_ind!=0x0)
		delete type_ind ;
}

// Leaf: the derivative of the target, plus the metric when indices are moved (unless it is flat).
bool Ope_id::der_zero_local() const {
	if (target->type_data==TERM_D) {
		if ((target->der_d!=0x0) && (*target->der_d!=0.))
			return false ;
	}
	else if (target->der_t!=0x0) {
		for (int j=0 ; j<target->der_t->get_n_comp() ; j++)
			if (!(*target->der_t)(target->der_t->indices(j))(dom).check_if_zero())
				return false ;
	}
	for (int i=0 ; i<valence ; i++)
		if ((target->val_t==0x0) || (target->val_t->get_index_type(i)!=(*type_ind)(i)))
			return syst->has_flat_metric() ;
	return true ;
}

Term_eq Ope_id::compute() const {

	Term_eq auxi (*target) ;
	// First put the names (not for doubles or scalars...)
	if (name_ind !=0x0) {
		for (int i=0 ; i<valence ; i++)
			auxi.val_t->set_name_ind(i, name_ind[i]) ;
		auxi.val_t->name_affected = true ;
		if (auxi.der_t!=0x0) {
			for (int i=0 ; i<valence ; i++)
				auxi.der_t->set_name_ind(i, name_ind[i]) ;
			auxi.der_t->name_affected = true ;
		}
	}

	// Manip of the indices if needed :
	for (int i=0 ; i<valence ; i++)
		if (auxi.val_t->get_index_type(i) != (*type_ind)(i)) {
			// Manipulation using the metric :
			syst->get_met()->manipulate_ind (auxi, i) ;
	}

	if (!need_sum)
		return auxi ;
	else {
		// Doit encore sommer sur certains indices :
		if (auxi.der_t==0x0)
			return Term_eq (dom, auxi.val_t->do_summation_one_dom(dom)) ;
		else
			return Term_eq (dom, auxi.val_t->do_summation_one_dom(dom), auxi.der_t->do_summation_one_dom(dom)) ;
	}
}
}
