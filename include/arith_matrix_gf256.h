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

#ifndef _ARITH_MATRIX_GF256_H
#define _ARITH_MATRIX_GF256_H
/*! \file
 Matrix operations over GF(256).
 \author Jocelyn Ryckeghem
*/

#include <stdint.h>

void copyMatrixPad32_gf256_avx2(uint8_t *C, const uint8_t *A);
void sqrMatrixToUpperT_gf256_avx2(uint8_t *Au, uint8_t *A);
/*************** Variable-time use of the input left matrix ***************/
void addMatrixMul_UPP_mulTab_gf256_avx2(uint8_t *C, const uint8_t *A,
                                                    const uint8_t *B);
void addMatrixMul_UtPP_mulTab_gf256_avx2(uint8_t *C, const uint8_t *A,
                                                     const uint8_t *B);
/*************** Constant-time use of the input left matrix ***************/
void matT_to_mulTab_gf256_avx2(uint8_t *T, const uint8_t *A);
void matrixMul_mulTabA_gf256_avx2(uint8_t *C, const uint8_t *T_A,
                                              const uint8_t *B);
#endif

