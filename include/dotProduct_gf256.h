/* Copyright 2024 Jocelyn Ryckeghem

This file is part of the HP-PROV library, based on the HPFA library.

HP-PROV is free software: you can redistribute it and/or modify it under
the terms of the GNU Lesser General Public License as published by
the Free Software Foundation, either version 3 of the License,
or (at your option) any later version.

HP-PROV is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with HP-PROV. If not, see <https://www.gnu.org/licenses/>. */

#ifndef _DOTPRODUCT_GF256_H
#define _DOTPRODUCT_GF256_H
/*! \file
 Computation of the dot product over GF(256).
 \author Jocelyn Ryckeghem
*/

#include <stdint.h>

uint8_t dotProductRevOp2_nbvar_gf256_pclmul(const uint8_t *u, const uint8_t *v);
uint8_t dotProductRevOp2_v_gf256_pclmul(const uint8_t *u, const uint8_t *v);
uint8_t dotProduct_gf256_pclmul(const uint8_t *u, const uint8_t *v,
                                unsigned int n);
#endif

