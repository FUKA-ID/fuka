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
#include "param.hpp"
namespace Kadath {
Ope_user::Ope_user (const System_of_eqs* zesys, Term_eq (*zeope) (const Term_eq&, Param*), Param* parso, Ope_eq* target) : Ope_eq(zesys, target->get_dom(), 1), pope(zeope) {
	par = parso ;
	parts[0] = target ;
	
}

Ope_user::~Ope_user() {
}

Term_eq Ope_user::compute() const {

	Term_eq target (parts[0]->action()) ;
	return pope(target, par) ;
}}
