/*
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
#include "space.hpp"
#include "scalar.hpp"
#include "tensor.hpp"

namespace Kadath {
void Domain::zero_top_theta_coef (Tensor& tt, int dom, int mquant) const {

	for (int c=0 ; c<tt.get_n_comp() ; c++) {
		Val_domain& so = (tt.get_valence()==0) ? tt.set().set_domain(dom)
		                                         : tt.set(tt.indices(c)).set_domain(dom) ;
		if (so.check_if_zero())
			continue ;
		so.coef() ;
		int nt = nbr_coefs(1) ;
		int baset = (*so.get_base().get_base_1d(1)) (0) ;
		if (!is_null_theta_mode (baset, mquant, nt-1, nt))
			continue ;

		Index pos (nbr_coefs) ;
		do {
			if ((pos(1)==nt-1) && ((ndim<3) || (pos(2)==0)))
				so.set_coef(pos) = 0. ;
		}
		while (pos.inc()) ;
	}
}
}
