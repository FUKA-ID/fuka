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

// Linearity of an operator in its parts, with value and derivative computed by the same functions:
// 1 linear in every part, 2 bilinear (product), 3 quotient (linear in the numerator), 0 none.
static int linearity (const Ope_eq* op, const System_of_eqs* syst) {
	static const std::unordered_set<std::type_index> linear = {
		typeid(Ope_add), typeid(Ope_sub), typeid(Ope_minus), typeid(Ope_partial), typeid(Ope_dn), typeid(Ope_dr),
		typeid(Ope_ddr), typeid(Ope_ddt), typeid(Ope_ddp), typeid(Ope_dt), typeid(Ope_srdr), typeid(Ope_div_r),
		typeid(Ope_mult_r), typeid(Ope_div_rsint), typeid(Ope_mult_rsint), typeid(Ope_div_sint),
		typeid(Ope_mult_sint), typeid(Ope_div_cost), typeid(Ope_div_xpone), typeid(Ope_mult_x),
		typeid(Ope_div_1mx2), typeid(Ope_div_1mrsL), typeid(Ope_mult_1mrsL), typeid(Ope_change_basis),
		typeid(Ope_der_flat), typeid(Ope_int), typeid(Ope_val), typeid(Ope_point), typeid(Ope_mode),
		typeid(Ope_val_mode)
	} ;
	std::type_index const t (typeid(*op)) ;
	if (linear.count(t)!=0)
		return 1 ;
	if (t==std::type_index(typeid(Ope_der)))
		return syst->has_flat_metric() ? 1 : 0 ;
	if ((t==std::type_index(typeid(Ope_mult))) || (t==std::type_index(typeid(Ope_scal))))
		return 2 ;
	if (t==std::type_index(typeid(Ope_div)))
		return 3 ;
	return 0 ;
}

// The derivative of a result as the value of a Term_eq (zero if there is none), with the index names of the
// value so that operators acting on it contract the same indices.
Term_eq Ope_eq::derivative_as_value (const Term_eq& r) {
	if (r.get_type_data()==TERM_D)
		return Term_eq (r.get_dom(), (r.der_d!=0x0) ? *r.der_d : 0.) ;
	Tensor der (*r.val_t, false) ;
	if (r.der_t!=0x0)
		der = *r.der_t ;
	else
		der = 0. ;
	if (r.val_t->is_name_affected())
		for (int i=0 ; i<r.val_t->get_valence() ; i++)
			der.set_name_ind(i, r.val_t->get_name_ind()[i]) ;
	return Term_eq (r.get_dom(), der) ;
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

bool Ope_eq::request_derivatives () const {
	int const kind = linearity(this, syst) ;
	if ((kind==0) || (n_ope<1))
		return false ;
	for (int i=0 ; i<n_ope ; i++)
		if (parts[i]==0x0)
			return false ;
	switch (kind) {
		case 1 :
			for (int i=0 ; i<n_ope ; i++)
				parts[i]->der_request = true ;
			return true ;
		case 2 : {
			if (n_ope!=2)
				return false ;
			bool const z0 = parts[0]->der_is_zero() ;
			bool const z1 = parts[1]->der_is_zero() ;
			if (z0==z1)
				return false ;
			parts[z0 ? 1 : 0]->der_request = true ;
			return true ;
		}
		case 3 :
			if ((n_ope!=2) || (parts[0]->der_is_zero()) || (!parts[1]->der_is_zero()))
				return false ;
			parts[0]->der_request = true ;
			return true ;
		default :
			return false ;
	}
}

Term_eq Ope_eq::action() const {
	bool const want_der = der_request ;
	der_request = false ;
	// sec_member: the values used by every Jacobian column (all columns then take the same path).
	if ((syst!=0x0) && syst->record_values) {
		Term_eq res (compute()) ;
		if (linearity(this, syst)!=0) {
			if (res.get_type_data()==TERM_D)
				val_cache.reset(new Term_eq(dom, *res.val_d)) ;
			else
				val_cache.reset(new Term_eq(dom, *res.val_t)) ;
			val_generation = syst->field_generation ;
		}
		return res ;
	}
	if ((syst==0x0) || (!syst->prune_active) || (dom==syst->prune_skip_dom[0]) || (dom==syst->prune_skip_dom[1]))
		return compute() ;
	// Zero derivative: the cached result.
	if (der_is_zero()) {
		if ((cache==nullptr) || (cache_generation!=syst->field_generation)) {
			cache.reset(new Term_eq(compute())) ;
			cache_generation = syst->field_generation ;
		}
		return want_der ? derivative_as_value(*cache) : *cache ;
	}
	// Non-zero derivative: a linear node takes its value from the cache and applies itself to the
	// derivatives of its parts only (they return them as values).
	bool const val_ok = syst->prune_linear && (val_cache!=nullptr) && (val_generation==syst->field_generation)
	    && ((val_cache->get_type_data()==TERM_D) || (!val_cache->val_t->is_m_quant_affected())) ;
	if (val_ok && request_derivatives()) {
		Term_eq der (compute()) ;
		for (int i=0 ; i<n_ope ; i++)
			parts[i]->der_request = false ;
		if (want_der)
			return der ;
		if (der.get_type_data()==TERM_D)
			return Term_eq (dom, *val_cache->val_d, *der.val_d) ;
		return Term_eq (dom, *val_cache->val_t, *der.val_t) ;
	}
	Term_eq res (compute()) ;
	return want_der ? derivative_as_value(res) : res ;
}

Ope_eq::~Ope_eq() {
	for (int i=0 ; i<n_ope ; i++)
		if (parts[i]!=0x0)
			delete parts[i] ;
	delete [] parts ;
}


}
