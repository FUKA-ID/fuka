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
#include "system_of_eqs.hpp"
#include "metric_nophi.hpp"
#include <typeindex>
#include <unordered_set>
namespace Kadath {
Ope_eq::Ope_eq(const System_of_eqs* zesys, int dd, int nn) : syst(zesys), dom(dd), n_ope(nn) {
	parts = new Ope_eq* [n_ope] ;
	for (int i=0 ; i<n_ope ; i++)
		parts[i] = 0x0 ;
}


Ope_eq::Ope_eq(const System_of_eqs* zesys, int dd) : syst(zesys), dom(dd) {
}

Ope_eq::Ope_eq (const Ope_eq&) {
	cerr << "Copy constructor not defined for Ope_eq" << endl ;
	abort() ;
}

// Operators whose result depends only on their parts and on the (fixed) geometry of their domain: a zero
// derivative of every part gives a zero derivative. The covariant derivative qualifies only with a flat
// metric (otherwise the metric carries a derivative of its own).
static bool depends_on_parts_only (const Ope_eq* op, const System_of_eqs* syst) {
	static const std::unordered_set<std::type_index> parts_only = {
		typeid(Ope_add), typeid(Ope_sub), typeid(Ope_mult), typeid(Ope_div), typeid(Ope_minus), typeid(Ope_scal),
		typeid(Ope_pow), typeid(Ope_sqrt), typeid(Ope_sqrt_anti), typeid(Ope_sqrt_nonstd), typeid(Ope_exp),
		typeid(Ope_log), typeid(Ope_sin), typeid(Ope_cos), typeid(Ope_sinh), typeid(Ope_cosh), typeid(Ope_atanh),
		typeid(Ope_partial), typeid(Ope_dn), typeid(Ope_dr), typeid(Ope_ddr), typeid(Ope_ddt), typeid(Ope_ddp),
		typeid(Ope_dt), typeid(Ope_srdr), typeid(Ope_lap), typeid(Ope_lap2), typeid(Ope_grad), typeid(Ope_div_r),
		typeid(Ope_mult_r), typeid(Ope_div_rsint), typeid(Ope_mult_rsint), typeid(Ope_div_sint),
		typeid(Ope_mult_sint), typeid(Ope_div_cost), typeid(Ope_div_xpone), typeid(Ope_mult_x),
		typeid(Ope_div_1mx2), typeid(Ope_div_1mrsL), typeid(Ope_mult_1mrsL), typeid(Ope_change_basis),
		typeid(Ope_determinant), typeid(Ope_inverse), typeid(Ope_inverse_nodet), typeid(Ope_user),
		typeid(Ope_user_bin), typeid(Ope_val), typeid(Ope_point), typeid(Ope_val_ori), typeid(Ope_mode),
		typeid(Ope_val_mode), typeid(Ope_int), typeid(Ope_int_volume), typeid(Ope_der_flat)
	} ;
	std::type_index const t (typeid(*op)) ;
	if (parts_only.count(t)!=0)
		return true ;
	if (t==std::type_index(typeid(Ope_der)))
		return syst->has_flat_metric() ;
	return false ;
}

bool Ope_eq::der_zero_local() const {
	if (!depends_on_parts_only(this, syst))
		return false ;
	for (int i=0 ; i<n_ope ; i++)
		if ((parts[i]==0x0) || (!parts[i]->der_is_zero()))
			return false ;
	return true ;
}

bool Ope_eq::der_is_zero() const {
	if (zero_stamp!=syst->column_stamp) {
		zero_flag = der_zero_local() ;
		zero_stamp = syst->column_stamp ;
	}
	return zero_flag ;
}

Term_eq Ope_eq::action() const {
	if ((syst==0x0) || (!syst->prune_active) || (dom==syst->prune_skip_dom[0]) || (dom==syst->prune_skip_dom[1])
	    || (!der_is_zero()))
		return compute() ;
	if ((cache!=nullptr) && (cache_generation==syst->field_generation))
		return *cache ;
	Term_eq res (compute()) ;
	cache.reset(new Term_eq(res)) ;
	cache_generation = syst->field_generation ;
	return res ;
}

Ope_eq::~Ope_eq() {
	for (int i=0 ; i<n_ope ; i++)
		if (parts[i]!=0x0)
			delete parts[i] ;
	delete [] parts ;
}


}
