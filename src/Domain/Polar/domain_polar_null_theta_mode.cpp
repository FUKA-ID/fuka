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
#include "polar.hpp"
#include "adapted_polar.hpp"
#include "scalar.hpp"
#include "tensor.hpp"

namespace Kadath {
static int m_quant_of (const Tensor& tt) {
	return (tt.is_m_quant_affected()) ? tt.get_parameters()->get_m_quant() : 0 ;
}

void Domain_polar_nucleus::zero_null_theta_mode (Tensor& tt, int dom) const {
	zero_top_theta_coef (tt, dom, m_quant_of(tt)) ;
}

void Domain_polar_shell::zero_null_theta_mode (Tensor& tt, int dom) const {
	zero_top_theta_coef (tt, dom, m_quant_of(tt)) ;
}

void Domain_polar_compact::zero_null_theta_mode (Tensor& tt, int dom) const {
	zero_top_theta_coef (tt, dom, m_quant_of(tt)) ;
}

void Domain_polar_shell_inner_adapted::zero_null_theta_mode (Tensor& tt, int dom) const {
	zero_top_theta_coef (tt, dom, m_quant_of(tt)) ;
}

void Domain_polar_shell_outer_adapted::zero_null_theta_mode (Tensor& tt, int dom) const {
	zero_top_theta_coef (tt, dom, m_quant_of(tt)) ;
}
}
