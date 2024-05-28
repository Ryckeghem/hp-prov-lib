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

#ifndef _PROV_GF256_H
#define _PROV_GF256_H
/*! \file
 Choice of the representation of GF(256) for PROV.
 \author Jocelyn Ryckeghem
*/

#if 1
  /* GF(256)=GF(2)/(x^8+x^4+x^3+x+1), canonical basis */

  /* lookup tables */
  #define mulxtab_x0_x4_gf256 mulxtab_x0_x4_cb_gf256_f_11b
  #define multab_x0_x4_gf256 multab_x0_x4_cb_gf256_f_11b
  #define invtab_gf256 invtab_cb_gf256_f_11b

  /* modular reduction of the higher part of a product, from x^8 to x^14,
     via a lookup multiplication by x^8 mod f = f-x^8 = 0x1b,
     we consider the part x^8 to ^11, then the part x^12 to x^14 */

  /* multiply by x^8 mod f,     TABl=multab_x0_x4_cb_gf256_f_11b[0x1b<<2]
     multiply by x^(4+4) mod f, TABl=mulxtab_x0_x4_cb_gf256_f_11b[(4<<2)+2]
     (we assume c_11*x^11=0) */
  #define remtab_x8l ((uint64_t)0x415a776c2d361b00)
  /* multiply by x^8 mod f,     TABh=multab_x0_x4_cb_gf256_f_11b[(0x1b<<2)+1]
     multiply by x^(4+4) mod f, TABh=mulxtab_x0_x4_cb_gf256_f_11b[(4<<2)+3]
     (we assume c_11*x^11=1)  */
  #define remtab_x8h ((uint64_t)0x9982afb4f5eec3d8)
  /* multiply by x^(8+4) mod f, TABl=multab_x0_x4_cb_gf256_f_11b[(0x1b<<2)+2]
     (we assume c_15*x^15=0) */
  #define remtab_x12 ((uint64_t)0x7cd7319ae64dab00)

  /* modular reduction for a product */
  #define REM_CORE_GF256 REM_CORE_CB_GF256_f_11b
  #define REM_GF256 REM_CB_GF256_f_11b
  #define REM_CORE_GF256_SSE2 REM_CORE_CB_GF256_f_11b_SSE2
  #define REM_GF256_SSE2 REM_CB_GF256_f_11b_SSE2

#endif

#endif

