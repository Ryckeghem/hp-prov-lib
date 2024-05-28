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

#ifndef _VECMATPROD_GF256_H
#define _VECMATPROD_GF256_H
/*! \file
 Vector-matrix product over GF(256).
 \author Jocelyn Ryckeghem
*/

#include <stdint.h>

/*************** Variable-time use of the input vector ***************/
/* extra I, (III), V */
void vecMatProd_mulTab_gf256_avx2(uint8_t *w, const uint8_t *u,
                                              const uint8_t *B);
/* I, III */
void vecMatx2Prod_mulTab_gf256_avx2(uint8_t *w, uint8_t *w_, const uint8_t *u,
                                    const uint8_t *B, const uint8_t *B_);
/* I, (III), V */
void vecMatProd_Uo_mulTab_gf256_avx2(uint8_t *w, const uint8_t *u,
                                                 const uint8_t *B);
/* I, III */
void vecMatx2Prod_Uo_mulTab_gf256_avx2(uint8_t *w, uint8_t *w_,
                                       const uint8_t *u,
                                       const uint8_t *B, const uint8_t *B_);
/* extra I, III, V */
void vecMatProd_Uv_mulTab_gf256_avx2(uint8_t *w, const uint8_t *u,
                                                 const uint8_t *B);
/* I */
void vecMatx2Prod_Uv_mulTab_gf256_avx2(uint8_t *w, uint8_t *w_,
                                       const uint8_t *u,
                                       const uint8_t *B, const uint8_t *B_);
/* I */
void vecMatProd_UvP_mulTab_gf256_avx2(uint8_t *w, const uint8_t *u,
                                      const uint8_t *U, const uint8_t *B);
/*************** Constant-time use of the input vector ***************/
void vec_to_mulTab_gf256_avx2(uint8_t *T, const uint8_t *u);
/* I, (III), V */
void vecMatProd_mulTabVec_gf256_avx2(uint8_t *w, const uint8_t *T_u,
                                                 const uint8_t *B);
/* I, III */
void vecMatx2Prod_mulTabVec_gf256_avx2(uint8_t *w, uint8_t *w_,
                                       const uint8_t *T_u,
                                       const uint8_t *B, const uint8_t *B_);
/* I, III, V */
void vecMatProd_Uv_mulTabVec_gf256_avx2(uint8_t *w, const uint8_t *T_u,
                                                    const uint8_t *B);
/* I */
void vecMatx2Prod_Uv_mulTabVec_gf256_avx2(uint8_t *w, uint8_t *w_,
                                          const uint8_t *T_u,
                                          const uint8_t *B, const uint8_t *B_);
void addVecMatTProd_gf256_pclmul(uint8_t *w, const uint8_t *u,
                                             const uint8_t *B);
void addVecMatProd_Ut_gf256_pclmul(uint8_t *w, const uint8_t *u,
                                               const uint8_t *B);
#endif

