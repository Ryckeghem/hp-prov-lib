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
 Computation of the dot product over GF(256).
 \author Jocelyn Ryckeghem
*/
#include "dotProduct_gf256.h"
#include "params.h"
#include "arith_gf256.h"
#include "prov_gf256.h"
#include <emmintrin.h>
#include <tmmintrin.h>
#include <wmmintrin.h>


#define NDIM NB_VAR

/*! \brief Compute the dot product of u and v over GF(256).
 \details v has to be 16-bit reversed in each 64-bit block because we use the
 carry-less multiplication.
 \param[in] u A vector of n elements.
 \param[in] v A vector of n elements.
 \return u.v.
 \req ceil(n/16) mod 2 == 1.
 \req v has to be 16-bit reversed in each 64-bit block.
 \req Zero padding for v (before the 16-bit reverse operation).
 \alloc n+PADDING16(n) bytes for u and v.
 \csttime u, v.
*/
uint8_t dotProductRevOp2_nbvar_gf256_pclmul(const uint8_t *u, const uint8_t *v)
{
  __m128i e,S,uv0,uv1,u0,u1,v0,v1,u00,u10,u01,u11,v00,v10,v01,v11;
  unsigned int i;

  uv0=_mm_setzero_si128();
  uv1=_mm_setzero_si128();
  for(i=0;i<(NDIM>>5);++i)
  {
    u0=_mm_loadu_si128((__m128i*)u);
    u1=_mm_loadu_si128((__m128i*)u+1);
    v0=_mm_loadu_si128((__m128i*)v);
    v1=_mm_loadu_si128((__m128i*)v+1);

    /* even bytes *x^8 */
    u00=_mm_slli_epi16(u0,8);
    u10=_mm_slli_epi16(u1,8);
    v00=_mm_slli_epi16(v0,8);
    v10=_mm_slli_epi16(v1,8);
    /* odd bytes */
    u01=_mm_srli_epi16(u0,8);
    u11=_mm_srli_epi16(u1,8);
    v01=_mm_srli_epi16(v0,8);
    v11=_mm_srli_epi16(v1,8);

    /* each carry-less multiplication performs a dot product */
    uv0^=_mm_clmulepi64_si128(u00,v00,0);
    uv0^=_mm_clmulepi64_si128(u00,v00,0x11);
    uv0^=_mm_clmulepi64_si128(u10,v10,0);
    uv0^=_mm_clmulepi64_si128(u10,v10,0x11);
    uv1^=_mm_clmulepi64_si128(u01,v01,0);
    uv1^=_mm_clmulepi64_si128(u01,v01,0x11);
    uv1^=_mm_clmulepi64_si128(u11,v11,0);
    uv1^=_mm_clmulepi64_si128(u11,v11,0x11);

    u+=32;
    v+=32;
  }
  u0=_mm_loadu_si128((__m128i*)u);
  v0=_mm_loadu_si128((__m128i*)v);

  /* even bytes *x^8 */
  u00=_mm_slli_epi16(u0,8);
  v00=_mm_slli_epi16(v0,8);
  /* odd bytes */
  u01=_mm_srli_epi16(u0,8);
  v01=_mm_srli_epi16(v0,8);

  uv0^=_mm_clmulepi64_si128(u00,v00,0);
  uv0^=_mm_clmulepi64_si128(u00,v00,0x11);
  uv1^=_mm_clmulepi64_si128(u01,v01,0);
  uv1^=_mm_clmulepi64_si128(u01,v01,0x11);

  e=_mm_srli_si128(uv0,8)^_mm_srli_si128(uv1,6);
  REM_CORE_GF256_SSE2(e,S)

  return (uint8_t)_mm_cvtsi128_si32(e);
}

#undef NDIM
#define NDIM NB_VIN

/*! \brief Compute the dot product of u and v over GF(256).
 \details v has to be 16-bit reversed in each 64-bit block because we use the
 carry-less multiplication.
 \param[in] u A vector of v elements.
 \param[in] v A vector of v elements.
 \return u.v.
 \req v=1 mod 32 or v=2 mod 32.
 \req v has to be 16-bit reversed in each 64-bit block.
 \req Zero padding for v (before the 16-bit reverse operation), except for the
 8 last bytes whose choice of values is free (because they are not used).
 \req If v=2 mod 32, then for the last 16-byte block B=B1||B0, switch the bytes
 4 and 7 (after the 16-bit reverse operation), i.e. B0|=B0>>(8*(7-4)); followed
 by B&=0xff00ff00000000;.
 \alloc v+PADDING16(v) bytes for u and v.
 \csttime u, v.
*/
uint8_t dotProductRevOp2_v_gf256_pclmul(const uint8_t *u, const uint8_t *v)
{
  __m128i e,S,uv0,uv1,u0,u1,v0,v1,u00,u10,u01,u11,v00,v10,v01,v11;
  unsigned int i;

  uv0=_mm_setzero_si128();
  uv1=_mm_setzero_si128();
  for(i=0;i<(NDIM>>5);++i)
  {
    u0=_mm_loadu_si128((__m128i*)u);
    u1=_mm_loadu_si128((__m128i*)u+1);
    v0=_mm_loadu_si128((__m128i*)v);
    v1=_mm_loadu_si128((__m128i*)v+1);

    /* even bytes *x^8 */
    u00=_mm_slli_epi16(u0,8);
    u10=_mm_slli_epi16(u1,8);
    v00=_mm_slli_epi16(v0,8);
    v10=_mm_slli_epi16(v1,8);
    /* odd bytes */
    u01=_mm_srli_epi16(u0,8);
    u11=_mm_srli_epi16(u1,8);
    v01=_mm_srli_epi16(v0,8);
    v11=_mm_srli_epi16(v1,8);

    /* each carry-less multiplication performs a dot product */
    uv0^=_mm_clmulepi64_si128(u00,v00,0);
    uv0^=_mm_clmulepi64_si128(u00,v00,0x11);
    uv0^=_mm_clmulepi64_si128(u10,v10,0);
    uv0^=_mm_clmulepi64_si128(u10,v10,0x11);
    uv1^=_mm_clmulepi64_si128(u01,v01,0);
    uv1^=_mm_clmulepi64_si128(u01,v01,0x11);
    uv1^=_mm_clmulepi64_si128(u11,v11,0);
    uv1^=_mm_clmulepi64_si128(u11,v11,0x11);

    u+=32;
    v+=32;
  }
  /* * * * * * * B A */
  u0=_mm_loadu_si128((__m128i*)u);
  /* 0 C 0 D 0 0 0 0 */
  v0=_mm_loadu_si128((__m128i*)v);
  /* 0 * 0 * 0 B 0 A */
  u0=_mm_unpacklo_epi8(u0,_mm_setzero_si128());
  uv1^=_mm_clmulepi64_si128(u0,v0,0);

  e=_mm_srli_si128(uv0,8)^_mm_srli_si128(uv1,6);
  REM_CORE_GF256_SSE2(e,S)

  return (uint8_t)_mm_cvtsi128_si32(e);
}

#undef NDIM

/*! \brief Compute the dot product of u and v over GF(256).
 \details u is 16-bit reversed in each 64-bit block via pshufb because we use
 the carry-less multiplication.
 \param[in] u A vector of n elements.
 \param[in] v A vector of n elements.
 \param[in] n The length of u and v.
 \return u.v.
 \alloc n bytes for u and v.
 \csttime u, v.
*/
uint8_t dotProduct_gf256_pclmul(const uint8_t *u, const uint8_t *v,
                                unsigned int n)
{
  const __m128i mask_rev64_epi16=_mm_set_epi64x((uint64_t)0x9080b0a0d0c0f0e,
                                                (uint64_t)0x100030205040706);
  __m128i e,S,uv0,uv1,u0,u1,v0,v1,u00,u10,u01,u11,v00,v10,v01,v11;
  uint64_t u64,v64;
  unsigned int i;

  uv0=_mm_setzero_si128();
  uv1=_mm_setzero_si128();
  for(i=0;i<(n>>5);++i)
  {
    u0=_mm_loadu_si128((__m128i*)u);
    u1=_mm_loadu_si128((__m128i*)u+1);
    v0=_mm_loadu_si128((__m128i*)v);
    v1=_mm_loadu_si128((__m128i*)v+1);
    /* 7 6 5 4 3 2 1 0 --> 1 0 3 2 5 4 7 6 */
    u0=_mm_shuffle_epi8(u0,mask_rev64_epi16);
    u1=_mm_shuffle_epi8(u1,mask_rev64_epi16);

    /* even bytes *x^8 */
    u00=_mm_slli_epi16(u0,8);
    u10=_mm_slli_epi16(u1,8);
    v00=_mm_slli_epi16(v0,8);
    v10=_mm_slli_epi16(v1,8);
    /* odd bytes */
    u01=_mm_srli_epi16(u0,8);
    u11=_mm_srli_epi16(u1,8);
    v01=_mm_srli_epi16(v0,8);
    v11=_mm_srli_epi16(v1,8);

    /* each carry-less multiplication performs a dot product */
    uv0^=_mm_clmulepi64_si128(u00,v00,0);
    uv0^=_mm_clmulepi64_si128(u00,v00,0x11);
    uv0^=_mm_clmulepi64_si128(u10,v10,0);
    uv0^=_mm_clmulepi64_si128(u10,v10,0x11);
    uv1^=_mm_clmulepi64_si128(u01,v01,0);
    uv1^=_mm_clmulepi64_si128(u01,v01,0x11);
    uv1^=_mm_clmulepi64_si128(u11,v11,0);
    uv1^=_mm_clmulepi64_si128(u11,v11,0x11);

    u+=32;
    v+=32;
  }
  n=n&31;
  if(n>>4)
  {
    u0=_mm_loadu_si128((__m128i*)u);
    v0=_mm_loadu_si128((__m128i*)v);
    /* 7 6 5 4 3 2 1 0 --> 1 0 3 2 5 4 7 6 */
    u0=_mm_shuffle_epi8(u0,mask_rev64_epi16);

    /* even bytes *x^8 */
    u00=_mm_slli_epi16(u0,8);
    v00=_mm_slli_epi16(v0,8);
    /* odd bytes */
    u01=_mm_srli_epi16(u0,8);
    v01=_mm_srli_epi16(v0,8);

    uv0^=_mm_clmulepi64_si128(u00,v00,0);
    uv0^=_mm_clmulepi64_si128(u00,v00,0x11);
    uv1^=_mm_clmulepi64_si128(u01,v01,0);
    uv1^=_mm_clmulepi64_si128(u01,v01,0x11);

    u+=16;
    v+=16;
    n-=16;
  }
  if(n>>3)
  {
    u0=_mm_loadu_si64((__m128i*)u);
    v0=_mm_loadu_si64((__m128i*)v);
    /* 7 6 5 4 3 2 1 0 --> 1 0 3 2 5 4 7 6 */
    u0=_mm_shuffle_epi8(u0,mask_rev64_epi16);

    /* even bytes *x^8 */
    u00=_mm_slli_epi16(u0,8);
    v00=_mm_slli_epi16(v0,8);
    /* odd bytes */
    u01=_mm_srli_epi16(u0,8);
    v01=_mm_srli_epi16(v0,8);

    uv0^=_mm_clmulepi64_si128(u00,v00,0);
    uv1^=_mm_clmulepi64_si128(u01,v01,0);

    u+=8;
    v+=8;
    n-=8;
  }
  if(n)
  {
    u64=*u;
    v64=*v;
    for(i=1;i<n;++i)
    {
      u64|=((uint64_t)(u[i]))<<(i<<3);
      v64|=((uint64_t)(v[i]))<<(i<<3);
    }
    u0=_mm_loadu_si64((__m128i*)&u64);
    v0=_mm_loadu_si64((__m128i*)&v64);
    /* 7 6 5 4 3 2 1 0 --> 1 0 3 2 5 4 7 6 */
    u0=_mm_shuffle_epi8(u0,mask_rev64_epi16);

    /* even bytes *x^8 */
    u00=_mm_slli_epi16(u0,8);
    v00=_mm_slli_epi16(v0,8);
    /* odd bytes */
    u01=_mm_srli_epi16(u0,8);
    v01=_mm_srli_epi16(v0,8);

    uv0^=_mm_clmulepi64_si128(u00,v00,0);
    uv1^=_mm_clmulepi64_si128(u01,v01,0);
  }

  e=_mm_srli_si128(uv0,8)^_mm_srli_si128(uv1,6);
  REM_CORE_GF256_SSE2(e,S)

  return (uint8_t)_mm_cvtsi128_si32(e);
}

