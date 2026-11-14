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
Ope_pow::Ope_pow (const System_of_eqs*zesys, int nn, Ope_eq* target) : Ope_eq(zesys,target->get_dom(), 1), power(nn) {
	parts[0] = target ;
}

Ope_pow::~Ope_pow() {
}

Term_eq Ope_pow::compute() const {
	if (power==0) 
		return Term_eq (dom, 1., 0.) ;
	else {
		Term_eq res_p0 (parts[0]->action()) ;
		// Check of type and valence :
		//int valence = res_p0.get_val_t().get_valence() ;
		//if (valence !=0) {
		if (res_p0.get_type_data()==TERM_T and res_p0.get_p_val_t()->get_valence() !=0) {
			cerr << "Ope_pow only defined for scalars" << endl ;
			abort() ;
		}
		Term_eq auxi (dom, res_p0.get_type_data()) ;
		auxi = (power>0) ? res_p0 : Term_eq(dom, 1., 0.) / res_p0 ;
		Term_eq res (auxi) ;
		for (int i=1 ; i<fabs(power) ; i++)
			res = res * auxi ;
		return res ;
	}
}
}
