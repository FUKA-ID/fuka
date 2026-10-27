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
#include "spheric.hpp"
#include "scalar.hpp"
#include "tensor.hpp"

namespace Kadath {
void Domain_nucleus::zero_null_theta_mode (Tensor& tt, int dom) const {
	zero_top_theta_coef (tt, dom, 0) ;
}

void Domain_shell::zero_null_theta_mode (Tensor& tt, int dom) const {
	zero_top_theta_coef (tt, dom, 0) ;
}

void Domain_compact::zero_null_theta_mode (Tensor& tt, int dom) const {
	zero_top_theta_coef (tt, dom, 0) ;
}
}
