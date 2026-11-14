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
namespace Kadath {
Ope_mult::Ope_mult (const System_of_eqs* zesys, Ope_eq* aa, Ope_eq* bb) : Ope_eq(zesys, aa->get_dom(), 2) {
	assert (aa->get_dom()==bb->get_dom()) ;
	parts[0] = aa ;
	parts[1] = bb ;
}

Ope_mult::~Ope_mult() {
}

Term_eq Ope_mult::compute() const {

	return (parts[0]->action() * parts[1]->action()) ;
}}
