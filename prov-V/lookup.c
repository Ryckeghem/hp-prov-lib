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

/*! \file
 avx2 mode to perform the constant-time table lookup.
 \author Jocelyn Ryckeghem
*/
#include "lookup.h"
#include <emmintrin.h>
#include <immintrin.h>


/*! \brief Constant-time computation of ((uint8_t*)T)[ind].
 \details Unrolled version of lookup256_avx2.
 \param[in] T A table of 256 bytes.
 \param[in] ind A 8-bit index.
 \return ((uint8_t*)T)[ind].
 \alloc 256 bytes for T.
 \csttime T, ind.
*/
uint8_t lookup256_r2_avx2(const uint64_t *T, uint8_t ind)
{
  const uint8_t ind32=(ind&15)|((ind<<3)&0x80);
  const __m256i ind16=_mm256_set_epi64x(0,ind32^0x80,0,ind32);
  __m256i acc,b0,b1;
  unsigned int i;

  /* generate an element b0 such that the bit sign of b0<<(ind div 32) is 1 */
  b0=_mm256_set1_epi16(((uint16_t)1)<<(15-(ind>>5)));
  b1=_mm256_slli_epi16(b0,1);
  acc=_mm256_setzero_si256();
  for(i=0;i<4;++i)
  {
    acc|=_mm256_loadu_si256((__m256i*)T)  &_mm256_srai_epi16(b0,15);
    acc|=_mm256_loadu_si256((__m256i*)T+1)&_mm256_srai_epi16(b1,15);
    b0=_mm256_slli_epi16(b0,2);
    b1=_mm256_slli_epi16(b1,2);
    T+=8;
  }
  acc=_mm256_shuffle_epi8(acc,ind16);

  return (uint8_t)_mm_cvtsi128_si32(_mm256_extracti128_si256(acc,1)
                                   |_mm256_castsi256_si128(acc));
}

