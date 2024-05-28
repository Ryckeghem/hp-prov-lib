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

#ifndef _ARITH_GF256_H
#define _ARITH_GF256_H
/*! \file
 Arithmetic operations over GF(256).
 \author Jocelyn Ryckeghem
*/

#define REM_CORE_CB_GF256_f_11b(e,S) \
  S=(e>>8)^(e>>12)^(e>>13);\
  /* S+=S*x */\
  S^=S<<1;\
  /* e+=S*(1+x) */\
  e^=S;\
  /* e+=S*(x^3+x^4) */\
  e^=(S<<3);

#define REM_CB_GF256_f_11b(e,S) \
  REM_CORE_CB_GF256_f_11b(e,S)\
  /* e=e mod x^8 */\
  e&=0xff;

#define REM_CORE_CB_GF256_f_11b_SSE2(e,S) \
  S=_mm_srli_epi16(e,8)^_mm_srli_epi16(e,12)^_mm_srli_epi16(e,13);\
  /* S+=S*x */\
  S^=_mm_slli_epi16(S,1);\
  /* e+=S*(1+x) */\
  e^=S;\
  /* e+=S*(x^3+x^4) */\
  e^=_mm_slli_epi16(S,3);

#define REM_CB_GF256_f_11b_SSE2(e,S,mask_00ff) \
  REM_CORE_CB_GF256_f_11b_SSE2(e,S,mask_00ff)\
  /* e=e mod x^8 */\
  e&=mask_00ff;

#endif

