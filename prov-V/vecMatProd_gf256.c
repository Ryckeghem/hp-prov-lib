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
 Vector-matrix product over GF(256).
 \author Jocelyn Ryckeghem
*/
#include "vecMatProd_gf256.h"
#include "params.h"
#include "multab_gf256.h"
#include "arith_gf256.h"
#include "dotProduct_gf256.h"
#include "prov_gf256.h"
#include <emmintrin.h>
#include <tmmintrin.h>
#include <wmmintrin.h>
#include <immintrin.h>


/*************** Variable-time use of the input vector ***************/
#define NROWS NB_VIN
#define NCOLS NB_OIL

/*! \brief Vector-matrix product over GF(256): (1*v)*(v*o)=(1*o).
 \param[out] w A vector over GF(256) of size o, equal to u.B.
 \param[in] u A vector over GF(256) of size v.
 \param[in] B A matrix over GF(256) of size v*o.
 \req ceil(o/32)=4.
 \alloc v bytes for u, v*o+PADDING32(o) bytes for B, o+PADDING32(o) bytes for w.
 \csttime B.
 \vartime u.
*/
void vecMatProd_mulTab_gf256_avx2(uint8_t *w, const uint8_t *u,
                                              const uint8_t *B)
{
  const __m256i mask_0f=_mm256_set1_epi8(15);
  __m256i T,T0,T4,w0,w1,w2,w3,B0,B1,B2,B3;
  unsigned int j;

  w0=_mm256_setzero_si256();
  w1=_mm256_setzero_si256();
  w2=_mm256_setzero_si256();
  w3=_mm256_setzero_si256();
  /* columns of u, rows of B */
  for(j=0;j<NROWS;++j)
  {
    /* T=Tl Th, T0=Tl Tl, T4=Th Th */
    T=_mm256_loadu_si256((__m256i*)multab_x0_x4_gf256+*u);
    B0=_mm256_loadu_si256((__m256i*)B);
    B1=_mm256_loadu_si256((__m256i*)B+1);
    B2=_mm256_loadu_si256((__m256i*)B+2);
    B3=_mm256_loadu_si256((__m256i*)B+3);

    T0=_mm256_permute4x64_epi64(T,0x44);
    T4=_mm256_permute4x64_epi64(T,0xee);

    w0^=_mm256_shuffle_epi8(T0,B0&mask_0f);
    w1^=_mm256_shuffle_epi8(T0,B1&mask_0f);
    w2^=_mm256_shuffle_epi8(T0,B2&mask_0f);
    w3^=_mm256_shuffle_epi8(T0,B3&mask_0f);
    w0^=_mm256_shuffle_epi8(T4,_mm256_srli_epi16(B0,4)&mask_0f);
    w1^=_mm256_shuffle_epi8(T4,_mm256_srli_epi16(B1,4)&mask_0f);
    w2^=_mm256_shuffle_epi8(T4,_mm256_srli_epi16(B2,4)&mask_0f);
    w3^=_mm256_shuffle_epi8(T4,_mm256_srli_epi16(B3,4)&mask_0f);
    ++u;
    B+=NCOLS;
  }
  _mm256_storeu_si256((__m256i*)w,w0);
  _mm256_storeu_si256((__m256i*)w+1,w1);
  _mm256_storeu_si256((__m256i*)w+2,w2);
  _mm256_storeu_si256((__m256i*)w+3,w3);
}

#undef NROWS
#undef NCOLS
#define NDIM NB_OIL
#if NDIM&31
  #define NDIMR (NDIM&31)
#else
  #define NDIMR 32
#endif

/*! \brief Vector-Umatrix product over GF(256): (1*o)*(o*o)=(1*o).
 \param[out] w A vector over GF(256) of size o, equal to u.B.
 \param[in] u A vector over GF(256) of size o.
 \param[in] B An upper triangular matrix over GF(256) of size o*o.
 \req ceil(o/32)=4.
 \alloc o bytes for u, (o*(o+1))/2+PADDING32(o) bytes for B,
        o+PADDING32(o) bytes for w.
 \csttime B.
 \vartime u.
*/
void vecMatProd_Uo_mulTab_gf256_avx2(uint8_t *w, const uint8_t *u,
                                                 const uint8_t *B)
{
  const uint64_t Tmask_U[4]={0x1716151413121110,0x1f1e1d1c1b1a1918,
                             0x2726252423222120,0x2f2e2d2c2b2a2928};
  const __m256i mask_01=_mm256_set1_epi8(1);
  const __m256i mask_0f=_mm256_set1_epi8(15);
  __m256i T,T0,T4,w0,w1,w2,w3,B0,B1,B2,B3,sum,mask_U;
  unsigned int j;

  w0=_mm256_setzero_si256();
  w1=_mm256_setzero_si256();
  w2=_mm256_setzero_si256();
  w3=_mm256_setzero_si256();
  /* columns of u, rows of B */
  mask_U=_mm256_loadu_si256((__m256i*)Tmask_U);
  for(j=0;j<NDIMR;++j)
  {
    /* T=Tl Th, T0=Tl Tl, T4=Th Th */
    T=_mm256_loadu_si256((__m256i*)multab_x0_x4_gf256+*u);
    B0=_mm256_loadu_si256((__m256i*)B);
    B1=_mm256_loadu_si256((__m256i*)B+1);
    B2=_mm256_loadu_si256((__m256i*)B+2);
    B3=_mm256_loadu_si256((__m256i*)B+3);

    T0=_mm256_permute4x64_epi64(T,0x44);
    T4=_mm256_permute4x64_epi64(T,0xee);

    sum=_mm256_shuffle_epi8(T0,B0&mask_0f);
    w1^=_mm256_shuffle_epi8(T0,B1&mask_0f);
    w2^=_mm256_shuffle_epi8(T0,B2&mask_0f);
    w3^=_mm256_shuffle_epi8(T0,B3&mask_0f);
    sum^=_mm256_shuffle_epi8(T4,_mm256_srli_epi16(B0,4)&mask_0f);
    w1^=_mm256_shuffle_epi8(T4,_mm256_srli_epi16(B1,4)&mask_0f);
    w2^=_mm256_shuffle_epi8(T4,_mm256_srli_epi16(B2,4)&mask_0f);
    w3^=_mm256_shuffle_epi8(T4,_mm256_srli_epi16(B3,4)&mask_0f);
    /* here, _mm256_cmpgt_epi8 returns 0xff...ff<<(8*j) */
    w0^=sum&_mm256_cmpgt_epi8(mask_U,mask_0f);

    mask_U=_mm256_sub_epi8(mask_U,mask_01);
    ++u;
    B+=NDIM-1-j;
  }
  _mm256_storeu_si256((__m256i*)w,w0);
  _mm256_storeu_si256((__m256i*)w+1,w1);
  _mm256_storeu_si256((__m256i*)w+2,w2);
  _mm256_storeu_si256((__m256i*)w+3,w3);
  w+=NDIMR;

  B+=NDIMR;
  w1=_mm256_setzero_si256();
  w2=_mm256_setzero_si256();
  w3=_mm256_setzero_si256();
  mask_U=_mm256_loadu_si256((__m256i*)Tmask_U);
  for(;j<(32+NDIMR);++j)
  {
    /* T=Tl Th, T0=Tl Tl, T4=Th Th */
    T=_mm256_loadu_si256((__m256i*)multab_x0_x4_gf256+*u);
    B1=_mm256_loadu_si256((__m256i*)B);
    B2=_mm256_loadu_si256((__m256i*)B+1);
    B3=_mm256_loadu_si256((__m256i*)B+2);

    T0=_mm256_permute4x64_epi64(T,0x44);
    T4=_mm256_permute4x64_epi64(T,0xee);

    sum=_mm256_shuffle_epi8(T0,B1&mask_0f);
    w2^=_mm256_shuffle_epi8(T0,B2&mask_0f);
    w3^=_mm256_shuffle_epi8(T0,B3&mask_0f);
    sum^=_mm256_shuffle_epi8(T4,_mm256_srli_epi16(B1,4)&mask_0f);
    w2^=_mm256_shuffle_epi8(T4,_mm256_srli_epi16(B2,4)&mask_0f);
    w3^=_mm256_shuffle_epi8(T4,_mm256_srli_epi16(B3,4)&mask_0f);
    w1^=sum&_mm256_cmpgt_epi8(mask_U,mask_0f);

    mask_U=_mm256_sub_epi8(mask_U,mask_01);
    ++u;
    B+=NDIM-1-j;
  }
  B+=32;
  mask_U=_mm256_loadu_si256((__m256i*)Tmask_U);
  for(;j<(64+NDIMR);++j)
  {
    /* T=Tl Th, T0=Tl Tl, T4=Th Th */
    T=_mm256_loadu_si256((__m256i*)multab_x0_x4_gf256+*u);
    B2=_mm256_loadu_si256((__m256i*)B);
    B3=_mm256_loadu_si256((__m256i*)B+1);

    T0=_mm256_permute4x64_epi64(T,0x44);
    T4=_mm256_permute4x64_epi64(T,0xee);

    sum=_mm256_shuffle_epi8(T0,B2&mask_0f);
    w3^=_mm256_shuffle_epi8(T0,B3&mask_0f);
    sum^=_mm256_shuffle_epi8(T4,_mm256_srli_epi16(B2,4)&mask_0f);
    w3^=_mm256_shuffle_epi8(T4,_mm256_srli_epi16(B3,4)&mask_0f);
    w2^=sum&_mm256_cmpgt_epi8(mask_U,mask_0f);

    mask_U=_mm256_sub_epi8(mask_U,mask_01);
    ++u;
    B+=NDIM-1-j;
  }
  B+=32;
  mask_U=_mm256_loadu_si256((__m256i*)Tmask_U);
  for(;j<NDIM;++j)
  {
    /* T=Tl Th, T0=Tl Tl, T4=Th Th */
    T=_mm256_loadu_si256((__m256i*)multab_x0_x4_gf256+*u);
    B3=_mm256_loadu_si256((__m256i*)B);

    T0=_mm256_permute4x64_epi64(T,0x44);
    T4=_mm256_permute4x64_epi64(T,0xee);

    sum=_mm256_shuffle_epi8(T0,B3&mask_0f);
    sum^=_mm256_shuffle_epi8(T4,_mm256_srli_epi16(B3,4)&mask_0f);
    w3^=sum&_mm256_cmpgt_epi8(mask_U,mask_0f);

    mask_U=_mm256_sub_epi8(mask_U,mask_01);
    ++u;
    B+=NDIM-1-j;
  }
  w1^=_mm256_loadu_si256((__m256i*)w);
  w2^=_mm256_loadu_si256((__m256i*)w+1);
  w3^=_mm256_loadu_si256((__m256i*)w+2);
  _mm256_storeu_si256((__m256i*)w,w1);
  _mm256_storeu_si256((__m256i*)w+1,w2);
  _mm256_storeu_si256((__m256i*)w+2,w3);
}

#undef NDIM
#undef NDIMR
#define NDIM NB_VIN

/*! \brief Vector-Umatrix product over GF(256): (1*v)*(v*v)=(1*v).
 \param[out] w A vector over GF(256) of size v, equal to u.B.
 \param[in] u A vector over GF(256) of size v.
 \param[in] B An upper triangular matrix over GF(256) of size v*v.
 \req v=162.
 \alloc v bytes for u, (v*(v+1))/2+PADDING32(v) bytes for B, v bytes for w.
 \csttime B.
 \vartime u.
*/
void vecMatProd_Uv_mulTab_gf256_avx2(uint8_t *w, const uint8_t *u,
                                                 const uint8_t *B)
{
  const uint64_t Tmask_U[4]={0x1716151413121110,0x1f1e1d1c1b1a1918,
                             0x2726252423222120,0x2f2e2d2c2b2a2928};
  const __m256i mask_01=_mm256_set1_epi8(1);
  const __m256i mask_0f=_mm256_set1_epi8(15);
  __m256i T,T0,T4,w0,w1,w2,w3,w4,B0,B1,B2,B3,B4,sum,mask_U;
  unsigned int j;

  B+=2;
  /* columns of u, rows of B */
  /* j=0 */
  /* T=Tl Th, T0=Tl Tl, T4=Th Th */
  T=_mm256_loadu_si256((__m256i*)multab_x0_x4_gf256+*u);
  mask_U=_mm256_loadu_si256((__m256i*)(B-2));
  B0=_mm256_loadu_si256((__m256i*)B);
  B1=_mm256_loadu_si256((__m256i*)B+1);
  B2=_mm256_loadu_si256((__m256i*)B+2);
  B3=_mm256_loadu_si256((__m256i*)B+3);
  B4=_mm256_loadu_si256((__m256i*)B+4);

  T0=_mm256_permute4x64_epi64(T,0x44);
  T4=_mm256_permute4x64_epi64(T,0xee);

  sum=_mm256_shuffle_epi8(T0,mask_U&mask_0f);
  w0=_mm256_shuffle_epi8(T0,B0&mask_0f);
  w1=_mm256_shuffle_epi8(T0,B1&mask_0f);
  w2=_mm256_shuffle_epi8(T0,B2&mask_0f);
  w3=_mm256_shuffle_epi8(T0,B3&mask_0f);
  w4=_mm256_shuffle_epi8(T0,B4&mask_0f);
  sum^=_mm256_shuffle_epi8(T4,_mm256_srli_epi16(mask_U,4)&mask_0f);
  w0^=_mm256_shuffle_epi8(T4,_mm256_srli_epi16(B0,4)&mask_0f);
  w1^=_mm256_shuffle_epi8(T4,_mm256_srli_epi16(B1,4)&mask_0f);
  w2^=_mm256_shuffle_epi8(T4,_mm256_srli_epi16(B2,4)&mask_0f);
  w3^=_mm256_shuffle_epi8(T4,_mm256_srli_epi16(B3,4)&mask_0f);
  w4^=_mm256_shuffle_epi8(T4,_mm256_srli_epi16(B4,4)&mask_0f);

  ++u;
  B+=NDIM-1;

  /* j=1 */
  /* T=Tl Th, T0=Tl Tl, T4=Th Th */
  T=_mm256_loadu_si256((__m256i*)multab_x0_x4_gf256+*u);
  mask_U=_mm256_loadu_si256((__m256i*)(B-1));
  B0=_mm256_loadu_si256((__m256i*)B);
  B1=_mm256_loadu_si256((__m256i*)B+1);
  B2=_mm256_loadu_si256((__m256i*)B+2);
  B3=_mm256_loadu_si256((__m256i*)B+3);
  B4=_mm256_loadu_si256((__m256i*)B+4);

  T0=_mm256_permute4x64_epi64(T,0x44);
  T4=_mm256_permute4x64_epi64(T,0xee);

  T=_mm256_shuffle_epi8(T0,mask_U&mask_0f);
  w0^=_mm256_shuffle_epi8(T0,B0&mask_0f);
  w1^=_mm256_shuffle_epi8(T0,B1&mask_0f);
  w2^=_mm256_shuffle_epi8(T0,B2&mask_0f);
  w3^=_mm256_shuffle_epi8(T0,B3&mask_0f);
  w4^=_mm256_shuffle_epi8(T0,B4&mask_0f);
  T^=_mm256_shuffle_epi8(T4,_mm256_srli_epi16(mask_U,4)&mask_0f);
  w0^=_mm256_shuffle_epi8(T4,_mm256_srli_epi16(B0,4)&mask_0f);
  w1^=_mm256_shuffle_epi8(T4,_mm256_srli_epi16(B1,4)&mask_0f);
  w2^=_mm256_shuffle_epi8(T4,_mm256_srli_epi16(B2,4)&mask_0f);
  w3^=_mm256_shuffle_epi8(T4,_mm256_srli_epi16(B3,4)&mask_0f);
  w4^=_mm256_shuffle_epi8(T4,_mm256_srli_epi16(B4,4)&mask_0f);
  sum^=_mm256_slli_epi16(T,8);

  ++u;
  B+=NDIM-2;
  _mm256_storeu_si256((__m256i*)w,sum);
  w+=2;

  mask_U=_mm256_loadu_si256((__m256i*)Tmask_U);
  for(j=0;j<32;++j)
  {
    /* T=Tl Th, T0=Tl Tl, T4=Th Th */
    T=_mm256_loadu_si256((__m256i*)multab_x0_x4_gf256+*u);
    B0=_mm256_loadu_si256((__m256i*)B);
    B1=_mm256_loadu_si256((__m256i*)B+1);
    B2=_mm256_loadu_si256((__m256i*)B+2);
    B3=_mm256_loadu_si256((__m256i*)B+3);
    B4=_mm256_loadu_si256((__m256i*)B+4);

    T0=_mm256_permute4x64_epi64(T,0x44);
    T4=_mm256_permute4x64_epi64(T,0xee);

    sum=_mm256_shuffle_epi8(T0,B0&mask_0f);
    w1^=_mm256_shuffle_epi8(T0,B1&mask_0f);
    w2^=_mm256_shuffle_epi8(T0,B2&mask_0f);
    w3^=_mm256_shuffle_epi8(T0,B3&mask_0f);
    w4^=_mm256_shuffle_epi8(T0,B4&mask_0f);
    sum^=_mm256_shuffle_epi8(T4,_mm256_srli_epi16(B0,4)&mask_0f);
    w1^=_mm256_shuffle_epi8(T4,_mm256_srli_epi16(B1,4)&mask_0f);
    w2^=_mm256_shuffle_epi8(T4,_mm256_srli_epi16(B2,4)&mask_0f);
    w3^=_mm256_shuffle_epi8(T4,_mm256_srli_epi16(B3,4)&mask_0f);
    w4^=_mm256_shuffle_epi8(T4,_mm256_srli_epi16(B4,4)&mask_0f);
    /* here, _mm256_cmpgt_epi8 returns 0xff...ff<<(8*j) */
    w0^=sum&_mm256_cmpgt_epi8(mask_U,mask_0f);

    mask_U=_mm256_sub_epi8(mask_U,mask_01);
    ++u;
    B+=NDIM-3-j;
  }
  _mm256_storeu_si256((__m256i*)w,w0);
  B+=32;
  mask_U=_mm256_loadu_si256((__m256i*)Tmask_U);
  for(j=32;j<64;++j)
  {
    /* T=Tl Th, T0=Tl Tl, T4=Th Th */
    T=_mm256_loadu_si256((__m256i*)multab_x0_x4_gf256+*u);
    B1=_mm256_loadu_si256((__m256i*)B);
    B2=_mm256_loadu_si256((__m256i*)B+1);
    B3=_mm256_loadu_si256((__m256i*)B+2);
    B4=_mm256_loadu_si256((__m256i*)B+3);

    T0=_mm256_permute4x64_epi64(T,0x44);
    T4=_mm256_permute4x64_epi64(T,0xee);

    sum=_mm256_shuffle_epi8(T0,B1&mask_0f);
    w2^=_mm256_shuffle_epi8(T0,B2&mask_0f);
    w3^=_mm256_shuffle_epi8(T0,B3&mask_0f);
    w4^=_mm256_shuffle_epi8(T0,B4&mask_0f);
    sum^=_mm256_shuffle_epi8(T4,_mm256_srli_epi16(B1,4)&mask_0f);
    w2^=_mm256_shuffle_epi8(T4,_mm256_srli_epi16(B2,4)&mask_0f);
    w3^=_mm256_shuffle_epi8(T4,_mm256_srli_epi16(B3,4)&mask_0f);
    w4^=_mm256_shuffle_epi8(T4,_mm256_srli_epi16(B4,4)&mask_0f);
    w1^=sum&_mm256_cmpgt_epi8(mask_U,mask_0f);

    mask_U=_mm256_sub_epi8(mask_U,mask_01);
    ++u;
    B+=NDIM-3-j;
  }
  _mm256_storeu_si256((__m256i*)w+1,w1);
  B+=32;
  mask_U=_mm256_loadu_si256((__m256i*)Tmask_U);
  for(j=64;j<96;++j)
  {
    /* T=Tl Th, T0=Tl Tl, T4=Th Th */
    T=_mm256_loadu_si256((__m256i*)multab_x0_x4_gf256+*u);
    B2=_mm256_loadu_si256((__m256i*)B);
    B3=_mm256_loadu_si256((__m256i*)B+1);
    B4=_mm256_loadu_si256((__m256i*)B+2);

    T0=_mm256_permute4x64_epi64(T,0x44);
    T4=_mm256_permute4x64_epi64(T,0xee);

    sum=_mm256_shuffle_epi8(T0,B2&mask_0f);
    w3^=_mm256_shuffle_epi8(T0,B3&mask_0f);
    w4^=_mm256_shuffle_epi8(T0,B4&mask_0f);
    sum^=_mm256_shuffle_epi8(T4,_mm256_srli_epi16(B2,4)&mask_0f);
    w3^=_mm256_shuffle_epi8(T4,_mm256_srli_epi16(B3,4)&mask_0f);
    w4^=_mm256_shuffle_epi8(T4,_mm256_srli_epi16(B4,4)&mask_0f);
    w2^=sum&_mm256_cmpgt_epi8(mask_U,mask_0f);

    mask_U=_mm256_sub_epi8(mask_U,mask_01);
    ++u;
    B+=NDIM-3-j;
  }
  _mm256_storeu_si256((__m256i*)w+2,w2);
  B+=32;
  mask_U=_mm256_loadu_si256((__m256i*)Tmask_U);
  for(j=96;j<128;++j)
  {
    /* T=Tl Th, T0=Tl Tl, T4=Th Th */
    T=_mm256_loadu_si256((__m256i*)multab_x0_x4_gf256+*u);
    B3=_mm256_loadu_si256((__m256i*)B);
    B4=_mm256_loadu_si256((__m256i*)B+1);

    T0=_mm256_permute4x64_epi64(T,0x44);
    T4=_mm256_permute4x64_epi64(T,0xee);

    sum=_mm256_shuffle_epi8(T0,B3&mask_0f);
    w4^=_mm256_shuffle_epi8(T0,B4&mask_0f);
    sum^=_mm256_shuffle_epi8(T4,_mm256_srli_epi16(B3,4)&mask_0f);
    w4^=_mm256_shuffle_epi8(T4,_mm256_srli_epi16(B4,4)&mask_0f);
    w3^=sum&_mm256_cmpgt_epi8(mask_U,mask_0f);

    mask_U=_mm256_sub_epi8(mask_U,mask_01);
    ++u;
    B+=NDIM-3-j;
  }
  _mm256_storeu_si256((__m256i*)w+3,w3);
  B+=32;
  mask_U=_mm256_loadu_si256((__m256i*)Tmask_U);
  for(j=128;j<160;++j)
  {
    /* T=Tl Th, T0=Tl Tl, T4=Th Th */
    T=_mm256_loadu_si256((__m256i*)multab_x0_x4_gf256+*u);
    B4=_mm256_loadu_si256((__m256i*)B);

    T0=_mm256_permute4x64_epi64(T,0x44);
    T4=_mm256_permute4x64_epi64(T,0xee);

    sum=_mm256_shuffle_epi8(T0,B4&mask_0f);
    sum^=_mm256_shuffle_epi8(T4,_mm256_srli_epi16(B4,4)&mask_0f);
    w4^=sum&_mm256_cmpgt_epi8(mask_U,mask_0f);

    mask_U=_mm256_sub_epi8(mask_U,mask_01);
    ++u;
    B+=NDIM-3-j;
  }
  _mm256_storeu_si256((__m256i*)w+4,w4);
}

#undef NDIM


/*************** Constant-time use of the input vector ***************/
#define NDIM NB_VIN

/*! \brief Constant-time precomputation of multab_x0_x4_gf256[u[i]].
 \param[out] T The lookup table corresponding to u.
 \param[in] u A vector over GF(256) of size v.
 \alloc v+PADDING16(v)*(floor(v/16)%2) bytes for u, 32v bytes for T.
 \csttime u, T.
*/
void vec_to_mulTab_gf256_avx2(uint8_t *T, const uint8_t *u)
{
  const __m256i x0=_mm256_loadu_si256((__m256i*)mulxtab_x0_x4_gf256);
  const __m256i x1=_mm256_loadu_si256((__m256i*)mulxtab_x0_x4_gf256+1);
  const __m256i x2=_mm256_loadu_si256((__m256i*)mulxtab_x0_x4_gf256+2);
  const __m256i x3=_mm256_loadu_si256((__m256i*)mulxtab_x0_x4_gf256+3);
  const __m256i x4=_mm256_loadu_si256((__m256i*)mulxtab_x0_x4_gf256+4);
  const __m256i x5=_mm256_loadu_si256((__m256i*)mulxtab_x0_x4_gf256+5);
  const __m256i x6=_mm256_loadu_si256((__m256i*)mulxtab_x0_x4_gf256+6);
  const __m256i x7=_mm256_loadu_si256((__m256i*)mulxtab_x0_x4_gf256+7);
  const __m256i mask_0f=_mm256_set1_epi8(15);
  const __m256i c256=_mm256_set1_epi16(256);
  __m256i u32,u0,u1,ul,u_x0_x4,u_x0,u_x1,u_x2,u_x3,ml,ml2;
  unsigned int i,k,l;

  for(i=0;i<(NDIM>>4);i+=2)
  {
    u32=_mm256_loadu_si256((__m256i*)u);
    u_x0_x4=_mm256_permute4x64_epi64(u32,0x44);
    u32    =_mm256_permute4x64_epi64(u32,0xee);

    for(k=0;k<2;++k)
    {
      u0=u_x0_x4&mask_0f;
      u1=_mm256_srli_epi16(u_x0_x4,4)&mask_0f;
      u_x0 =_mm256_shuffle_epi8(x0,u0);
      u_x1 =_mm256_shuffle_epi8(x1,u0);
      u_x2 =_mm256_shuffle_epi8(x2,u0);
      u_x3 =_mm256_shuffle_epi8(x3,u0);
      u_x0^=_mm256_shuffle_epi8(x4,u1);
      u_x1^=_mm256_shuffle_epi8(x5,u1);
      u_x2^=_mm256_shuffle_epi8(x6,u1);
      u_x3^=_mm256_shuffle_epi8(x7,u1);

      /* l 0xf0 l 0xf0 l 0xf0 l 0xf0 l 0xf0 l 0xf0 l 0xf0 l 0xf0 */
      ml=_mm256_srli_epi16(mask_0f,4);
      for(l=0;l<16;++l)
      {
        ul =_mm256_shuffle_epi8(u_x0,ml);
        /* l l 0xf0 0xf0 l l 0xf0 0xf0 l l 0xf0 0xf0 l l 0xf0 0xf0 */
        ml2=_mm256_unpacklo_epi8(ml,ml);
        ul^=_mm256_shuffle_epi8(u_x1,ml2);
        /* l l l l 0xf0 0xf0 0xf0 0xf0 l l l l 0xf0 0xf0 0xf0 0xf0 */
        ml2=_mm256_unpacklo_epi16(ml2,ml2);
        ul^=_mm256_shuffle_epi8(u_x2,ml2);
        /* l l l l l l l l 0xf0 0xf0 0xf0 0xf0 0xf0 0xf0 0xf0 0xf0 */
        ml2=_mm256_unpacklo_epi32(ml2,ml2);
        ul^=_mm256_shuffle_epi8(u_x3,ml2);
        _mm256_storeu_si256((__m256i*)T,ul);
        /* l+1 f0 l+1 f0 l+1 f0 l+1 f0 l+1 f0 l+1 f0 l+1 f0 l+1 f0 */
        ml=_mm256_add_epi16(ml,c256);
        T+=32;
      }
      u+=16;
      if((i+1+k)==(NDIM>>4))
      {
        break;
      }
      u_x0_x4=u32;
    }
  }
  ml=mask_0f^_mm256_slli_epi16(mask_0f,4);
  for(i=0;i<(NDIM&15);++i)
  {
    u0=_mm256_set1_epi8(u[i]);
    /* dot product */
    u1=_mm256_andnot_si256(_mm256_shuffle_epi8(ml,u0),x7);
    u1^=_mm256_andnot_si256(_mm256_shuffle_epi8(ml,_mm256_slli_epi16(u0,1)),x6);
    u1^=_mm256_andnot_si256(_mm256_shuffle_epi8(ml,_mm256_slli_epi16(u0,2)),x5);
    u1^=_mm256_andnot_si256(_mm256_shuffle_epi8(ml,_mm256_slli_epi16(u0,3)),x4);
    u1^=_mm256_andnot_si256(_mm256_shuffle_epi8(ml,_mm256_slli_epi16(u0,4)),x3);
    u1^=_mm256_andnot_si256(_mm256_shuffle_epi8(ml,_mm256_slli_epi16(u0,5)),x2);
    u1^=_mm256_andnot_si256(_mm256_shuffle_epi8(ml,_mm256_slli_epi16(u0,6)),x1);
    u1^=_mm256_andnot_si256(_mm256_shuffle_epi8(ml,_mm256_slli_epi16(u0,7)),x0);
    _mm256_storeu_si256((__m256i*)T+i,u1);
  }
}

#undef NDIM
#define NROWS NB_VIN
#define NCOLS NB_OIL

/*! \brief Vector-matrix product over GF(256): (1*v)*(v*o)=(1*o).
 \param[out] w A vector over GF(256) of size o, equal to u.B.
 \param[in] T_u A table, generated by vec_to_mulTab_*(T_u,u).
 \param[in] B A matrix over GF(256) of size v*o.
 \req ceil(o/32)=4.
 \alloc 32v bytes for T_u, v*o+PADDING32(o) bytes for B,
        o+PADDING32(o) bytes for w.
 \csttime T_u, B, w.
*/
void vecMatProd_mulTabVec_gf256_avx2(uint8_t *w, const uint8_t *T_u,
                                                 const uint8_t *B)
{
  const __m256i mask_0f=_mm256_set1_epi8(15);
  __m256i T,T0,T4,w0,w1,w2,w3,B0,B1,B2,B3;
  unsigned int j;

  w0=_mm256_setzero_si256();
  w1=_mm256_setzero_si256();
  w2=_mm256_setzero_si256();
  w3=_mm256_setzero_si256();
  /* columns of u, rows of B */
  for(j=0;j<NROWS;++j)
  {
    /* T=Tl Th, T0=Tl Tl, T4=Th Th */
    T=_mm256_loadu_si256((__m256i*)T_u+j);
    B0=_mm256_loadu_si256((__m256i*)B);
    B1=_mm256_loadu_si256((__m256i*)B+1);
    B2=_mm256_loadu_si256((__m256i*)B+2);
    B3=_mm256_loadu_si256((__m256i*)B+3);

    T0=_mm256_permute4x64_epi64(T,0x44);
    T4=_mm256_permute4x64_epi64(T,0xee);

    w0^=_mm256_shuffle_epi8(T0,B0&mask_0f);
    w1^=_mm256_shuffle_epi8(T0,B1&mask_0f);
    w2^=_mm256_shuffle_epi8(T0,B2&mask_0f);
    w3^=_mm256_shuffle_epi8(T0,B3&mask_0f);
    w0^=_mm256_shuffle_epi8(T4,_mm256_srli_epi16(B0,4)&mask_0f);
    w1^=_mm256_shuffle_epi8(T4,_mm256_srli_epi16(B1,4)&mask_0f);
    w2^=_mm256_shuffle_epi8(T4,_mm256_srli_epi16(B2,4)&mask_0f);
    w3^=_mm256_shuffle_epi8(T4,_mm256_srli_epi16(B3,4)&mask_0f);
    B+=NCOLS;
  }
  _mm256_storeu_si256((__m256i*)w,w0);
  _mm256_storeu_si256((__m256i*)w+1,w1);
  _mm256_storeu_si256((__m256i*)w+2,w2);
  _mm256_storeu_si256((__m256i*)w+3,w3);
}

#undef NROWS
#undef NCOLS
#define NDIM NB_VIN

/*! \brief Vector-Umatrix product over GF(256): (1*v)*(v*v)=(1*v).
 \param[out] w A vector over GF(256) of size v, equal to u.B.
 \param[in] T_u A table, generated by vec_to_mulTab_*(T_u,u).
 \param[in] B An upper triangular matrix over GF(256) of size v*v.
 \req v=162.
 \alloc 32v bytes for T_u, (v*(v+1))/2+PADDING32(v) bytes for B, v bytes for w.
 \csttime T_u, B, w.
*/
void vecMatProd_Uv_mulTabVec_gf256_avx2(uint8_t *w, const uint8_t *T_u,
                                                    const uint8_t *B)
{
  const uint64_t Tmask_U[4]={0x1716151413121110,0x1f1e1d1c1b1a1918,
                             0x2726252423222120,0x2f2e2d2c2b2a2928};
  const __m256i mask_01=_mm256_set1_epi8(1);
  const __m256i mask_0f=_mm256_set1_epi8(15);
  __m256i T,T0,T4,w0,w1,w2,w3,w4,B0,B1,B2,B3,B4,sum,mask_U;
  unsigned int j;

  B+=2;
  /* columns of u, rows of B */
  /* j=0 */
  /* T=Tl Th, T0=Tl Tl, T4=Th Th */
  T=_mm256_loadu_si256((__m256i*)T_u);
  mask_U=_mm256_loadu_si256((__m256i*)(B-2));
  B0=_mm256_loadu_si256((__m256i*)B);
  B1=_mm256_loadu_si256((__m256i*)B+1);
  B2=_mm256_loadu_si256((__m256i*)B+2);
  B3=_mm256_loadu_si256((__m256i*)B+3);
  B4=_mm256_loadu_si256((__m256i*)B+4);

  T0=_mm256_permute4x64_epi64(T,0x44);
  T4=_mm256_permute4x64_epi64(T,0xee);

  sum=_mm256_shuffle_epi8(T0,mask_U&mask_0f);
  w0=_mm256_shuffle_epi8(T0,B0&mask_0f);
  w1=_mm256_shuffle_epi8(T0,B1&mask_0f);
  w2=_mm256_shuffle_epi8(T0,B2&mask_0f);
  w3=_mm256_shuffle_epi8(T0,B3&mask_0f);
  w4=_mm256_shuffle_epi8(T0,B4&mask_0f);
  sum^=_mm256_shuffle_epi8(T4,_mm256_srli_epi16(mask_U,4)&mask_0f);
  w0^=_mm256_shuffle_epi8(T4,_mm256_srli_epi16(B0,4)&mask_0f);
  w1^=_mm256_shuffle_epi8(T4,_mm256_srli_epi16(B1,4)&mask_0f);
  w2^=_mm256_shuffle_epi8(T4,_mm256_srli_epi16(B2,4)&mask_0f);
  w3^=_mm256_shuffle_epi8(T4,_mm256_srli_epi16(B3,4)&mask_0f);
  w4^=_mm256_shuffle_epi8(T4,_mm256_srli_epi16(B4,4)&mask_0f);

  T_u+=32;
  B+=NDIM-1;

  /* j=1 */
  /* T=Tl Th, T0=Tl Tl, T4=Th Th */
  T=_mm256_loadu_si256((__m256i*)T_u);
  mask_U=_mm256_loadu_si256((__m256i*)(B-1));
  B0=_mm256_loadu_si256((__m256i*)B);
  B1=_mm256_loadu_si256((__m256i*)B+1);
  B2=_mm256_loadu_si256((__m256i*)B+2);
  B3=_mm256_loadu_si256((__m256i*)B+3);
  B4=_mm256_loadu_si256((__m256i*)B+4);

  T0=_mm256_permute4x64_epi64(T,0x44);
  T4=_mm256_permute4x64_epi64(T,0xee);

  T=_mm256_shuffle_epi8(T0,mask_U&mask_0f);
  w0^=_mm256_shuffle_epi8(T0,B0&mask_0f);
  w1^=_mm256_shuffle_epi8(T0,B1&mask_0f);
  w2^=_mm256_shuffle_epi8(T0,B2&mask_0f);
  w3^=_mm256_shuffle_epi8(T0,B3&mask_0f);
  w4^=_mm256_shuffle_epi8(T0,B4&mask_0f);
  T^=_mm256_shuffle_epi8(T4,_mm256_srli_epi16(mask_U,4)&mask_0f);
  w0^=_mm256_shuffle_epi8(T4,_mm256_srli_epi16(B0,4)&mask_0f);
  w1^=_mm256_shuffle_epi8(T4,_mm256_srli_epi16(B1,4)&mask_0f);
  w2^=_mm256_shuffle_epi8(T4,_mm256_srli_epi16(B2,4)&mask_0f);
  w3^=_mm256_shuffle_epi8(T4,_mm256_srli_epi16(B3,4)&mask_0f);
  w4^=_mm256_shuffle_epi8(T4,_mm256_srli_epi16(B4,4)&mask_0f);
  sum^=_mm256_slli_epi16(T,8);

  T_u+=32;
  B+=NDIM-2;
  _mm256_storeu_si256((__m256i*)w,sum);
  w+=2;

  mask_U=_mm256_loadu_si256((__m256i*)Tmask_U);
  for(j=0;j<32;++j)
  {
    /* T=Tl Th, T0=Tl Tl, T4=Th Th */
    T=_mm256_loadu_si256((__m256i*)T_u+j);
    B0=_mm256_loadu_si256((__m256i*)B);
    B1=_mm256_loadu_si256((__m256i*)B+1);
    B2=_mm256_loadu_si256((__m256i*)B+2);
    B3=_mm256_loadu_si256((__m256i*)B+3);
    B4=_mm256_loadu_si256((__m256i*)B+4);

    T0=_mm256_permute4x64_epi64(T,0x44);
    T4=_mm256_permute4x64_epi64(T,0xee);

    sum=_mm256_shuffle_epi8(T0,B0&mask_0f);
    w1^=_mm256_shuffle_epi8(T0,B1&mask_0f);
    w2^=_mm256_shuffle_epi8(T0,B2&mask_0f);
    w3^=_mm256_shuffle_epi8(T0,B3&mask_0f);
    w4^=_mm256_shuffle_epi8(T0,B4&mask_0f);

    sum^=_mm256_shuffle_epi8(T4,_mm256_srli_epi16(B0,4)&mask_0f);
    w1^=_mm256_shuffle_epi8(T4,_mm256_srli_epi16(B1,4)&mask_0f);
    w2^=_mm256_shuffle_epi8(T4,_mm256_srli_epi16(B2,4)&mask_0f);
    w3^=_mm256_shuffle_epi8(T4,_mm256_srli_epi16(B3,4)&mask_0f);
    w4^=_mm256_shuffle_epi8(T4,_mm256_srli_epi16(B4,4)&mask_0f);
    /* here, _mm256_cmpgt_epi8 returns 0xff...ff<<(8*j) */
    w0^=sum&_mm256_cmpgt_epi8(mask_U,mask_0f);

    mask_U=_mm256_sub_epi8(mask_U,mask_01);
    B+=NDIM-3-j;
  }
  _mm256_storeu_si256((__m256i*)w,w0);
  B+=32;
  mask_U=_mm256_loadu_si256((__m256i*)Tmask_U);
  for(j=32;j<64;++j)
  {
    /* T=Tl Th, T0=Tl Tl, T4=Th Th */
    T=_mm256_loadu_si256((__m256i*)T_u+j);
    B1=_mm256_loadu_si256((__m256i*)B);
    B2=_mm256_loadu_si256((__m256i*)B+1);
    B3=_mm256_loadu_si256((__m256i*)B+2);
    B4=_mm256_loadu_si256((__m256i*)B+3);

    T0=_mm256_permute4x64_epi64(T,0x44);
    T4=_mm256_permute4x64_epi64(T,0xee);

    sum=_mm256_shuffle_epi8(T0,B1&mask_0f);
    w2^=_mm256_shuffle_epi8(T0,B2&mask_0f);
    w3^=_mm256_shuffle_epi8(T0,B3&mask_0f);
    w4^=_mm256_shuffle_epi8(T0,B4&mask_0f);

    sum^=_mm256_shuffle_epi8(T4,_mm256_srli_epi16(B1,4)&mask_0f);
    w2^=_mm256_shuffle_epi8(T4,_mm256_srli_epi16(B2,4)&mask_0f);
    w3^=_mm256_shuffle_epi8(T4,_mm256_srli_epi16(B3,4)&mask_0f);
    w4^=_mm256_shuffle_epi8(T4,_mm256_srli_epi16(B4,4)&mask_0f);
    w1^=sum&_mm256_cmpgt_epi8(mask_U,mask_0f);

    mask_U=_mm256_sub_epi8(mask_U,mask_01);
    B+=NDIM-3-j;
  }
  _mm256_storeu_si256((__m256i*)w+1,w1);
  B+=32;
  mask_U=_mm256_loadu_si256((__m256i*)Tmask_U);
  for(j=64;j<96;++j)
  {
    /* T=Tl Th, T0=Tl Tl, T4=Th Th */
    T=_mm256_loadu_si256((__m256i*)T_u+j);
    B2=_mm256_loadu_si256((__m256i*)B);
    B3=_mm256_loadu_si256((__m256i*)B+1);
    B4=_mm256_loadu_si256((__m256i*)B+2);

    T0=_mm256_permute4x64_epi64(T,0x44);
    T4=_mm256_permute4x64_epi64(T,0xee);

    sum=_mm256_shuffle_epi8(T0,B2&mask_0f);
    w3^=_mm256_shuffle_epi8(T0,B3&mask_0f);
    w4^=_mm256_shuffle_epi8(T0,B4&mask_0f);

    sum^=_mm256_shuffle_epi8(T4,_mm256_srli_epi16(B2,4)&mask_0f);
    w3^=_mm256_shuffle_epi8(T4,_mm256_srli_epi16(B3,4)&mask_0f);
    w4^=_mm256_shuffle_epi8(T4,_mm256_srli_epi16(B4,4)&mask_0f);
    w2^=sum&_mm256_cmpgt_epi8(mask_U,mask_0f);

    mask_U=_mm256_sub_epi8(mask_U,mask_01);
    B+=NDIM-3-j;
  }
  _mm256_storeu_si256((__m256i*)w+2,w2);
  B+=32;
  mask_U=_mm256_loadu_si256((__m256i*)Tmask_U);
  for(j=96;j<128;++j)
  {
    /* T=Tl Th, T0=Tl Tl, T4=Th Th */
    T=_mm256_loadu_si256((__m256i*)T_u+j);
    B3=_mm256_loadu_si256((__m256i*)B);
    B4=_mm256_loadu_si256((__m256i*)B+1);

    T0=_mm256_permute4x64_epi64(T,0x44);
    T4=_mm256_permute4x64_epi64(T,0xee);

    sum=_mm256_shuffle_epi8(T0,B3&mask_0f);
    w4^=_mm256_shuffle_epi8(T0,B4&mask_0f);

    sum^=_mm256_shuffle_epi8(T4,_mm256_srli_epi16(B3,4)&mask_0f);
    w4^=_mm256_shuffle_epi8(T4,_mm256_srli_epi16(B4,4)&mask_0f);
    w3^=sum&_mm256_cmpgt_epi8(mask_U,mask_0f);

    mask_U=_mm256_sub_epi8(mask_U,mask_01);
    B+=NDIM-3-j;
  }
  _mm256_storeu_si256((__m256i*)w+3,w3);
  B+=32;
  mask_U=_mm256_loadu_si256((__m256i*)Tmask_U);
  for(j=128;j<160;++j)
  {
    /* T=Tl Th, T0=Tl Tl, T4=Th Th */
    T=_mm256_loadu_si256((__m256i*)T_u+j);
    B4=_mm256_loadu_si256((__m256i*)B);

    T0=_mm256_permute4x64_epi64(T,0x44);
    T4=_mm256_permute4x64_epi64(T,0xee);

    sum=_mm256_shuffle_epi8(T0,B4&mask_0f);
    sum^=_mm256_shuffle_epi8(T4,_mm256_srli_epi16(B4,4)&mask_0f);
    w4^=sum&_mm256_cmpgt_epi8(mask_U,mask_0f);

    mask_U=_mm256_sub_epi8(mask_U,mask_01);
    B+=NDIM-3-j;
  }
  _mm256_storeu_si256((__m256i*)w+4,w4);
}

#undef NDIM
#define NROWS NB_VIN
#define NCOLS NB_OIL

/*! \brief Multiply a vector by the transpose of a matrix, then add it to w,
 over GF(256): (1*o)*(o*v)=(1*v).
 \param[out] w A vector over GF(256) of size v, equal to u.(B^T)=B.(u^T).
 \param[in] u A vector over GF(256) of size o.
 \param[in] B A matrix over GF(256) of size v*o.
 \req o=108.
 \alloc o+PADDING16(o) bytes for u, v*o+PADDING16(o) bytes for B, o bytes for w.
 \csttime u, B, w.
*/
void addVecMatTProd_gf256_pclmul(uint8_t *w, const uint8_t *u,
                                             const uint8_t *B)
{
  const __m128i mask_rev64_epi16=_mm_set_epi64x((uint64_t)0x9080b0a0d0c0f0e,
                                                (uint64_t)0x100030205040706);
  __m128i e,S,w0,w1,u00,u10,u20,u30,u40,u50,u60,u01,u11,u21,u31,u41,u51,u61,
          B0,B1,B00,B10,B01,B11;
  unsigned int i;

  B0=_mm_loadu_si128((__m128i*)u);
  B1=_mm_loadu_si128((__m128i*)u+1);
  /* 7 6 5 4 3 2 1 0 --> 1 0 3 2 5 4 7 6 */
  B0=_mm_shuffle_epi8(B0,mask_rev64_epi16);
  B1=_mm_shuffle_epi8(B1,mask_rev64_epi16);
  /* even bytes *x^8 */
  u00=_mm_slli_epi16(B0,8);
  u10=_mm_slli_epi16(B1,8);
  /* odd bytes */
  u01=_mm_srli_epi16(B0,8);
  u11=_mm_srli_epi16(B1,8);

  B0=_mm_loadu_si128((__m128i*)u+2);
  B1=_mm_loadu_si128((__m128i*)u+3);
  /* 7 6 5 4 3 2 1 0 --> 1 0 3 2 5 4 7 6 */
  B0=_mm_shuffle_epi8(B0,mask_rev64_epi16);
  B1=_mm_shuffle_epi8(B1,mask_rev64_epi16);
  /* even bytes *x^8 */
  u20=_mm_slli_epi16(B0,8);
  u30=_mm_slli_epi16(B1,8);
  /* odd bytes */
  u21=_mm_srli_epi16(B0,8);
  u31=_mm_srli_epi16(B1,8);

  B0=_mm_loadu_si128((__m128i*)u+4);
  B1=_mm_loadu_si128((__m128i*)u+5);
  /* 7 6 5 4 3 2 1 0 --> 1 0 3 2 5 4 7 6 */
  B0=_mm_shuffle_epi8(B0,mask_rev64_epi16);
  B1=_mm_shuffle_epi8(B1,mask_rev64_epi16);
  /* even bytes *x^8 */
  u40=_mm_slli_epi16(B0,8);
  u50=_mm_slli_epi16(B1,8);
  /* odd bytes */
  u41=_mm_srli_epi16(B0,8);
  u51=_mm_srli_epi16(B1,8);

  B0=_mm_loadu_si128((__m128i*)u+6);
  /* zero padding */
  B0=_mm_srli_si128(_mm_slli_si128(B0,4),4);
  /* 7 6 5 4 3 2 1 0 --> 1 0 3 2 5 4 7 6 */
  B0=_mm_shuffle_epi8(B0,mask_rev64_epi16);
  /* even bytes *x^8 */
  u60=_mm_slli_epi16(B0,8);
  /* odd bytes */
  u61=_mm_srli_epi16(B0,8);

  for(i=0;i<NROWS;++i)
  {
    B0=_mm_loadu_si128((__m128i*)B);
    B1=_mm_loadu_si128((__m128i*)B+1);

    /* even bytes *x^8 */
    B00=_mm_slli_epi16(B0,8);
    B10=_mm_slli_epi16(B1,8);
    /* odd bytes */
    B01=_mm_srli_epi16(B0,8);
    B11=_mm_srli_epi16(B1,8);

    /* each carry-less multiplication performs a dot product */
    w0 =_mm_clmulepi64_si128(u00,B00,0);
    w0^=_mm_clmulepi64_si128(u00,B00,0x11);
    w0^=_mm_clmulepi64_si128(u10,B10,0);
    w0^=_mm_clmulepi64_si128(u10,B10,0x11);
    w1 =_mm_clmulepi64_si128(u01,B01,0);
    w1^=_mm_clmulepi64_si128(u01,B01,0x11);
    w1^=_mm_clmulepi64_si128(u11,B11,0);
    w1^=_mm_clmulepi64_si128(u11,B11,0x11);

    /* repeat the previous process */
    B0=_mm_loadu_si128((__m128i*)B+2);
    B1=_mm_loadu_si128((__m128i*)B+3);

    /* even bytes *x^8 */
    B00=_mm_slli_epi16(B0,8);
    B10=_mm_slli_epi16(B1,8);
    /* odd bytes */
    B01=_mm_srli_epi16(B0,8);
    B11=_mm_srli_epi16(B1,8);

    /* each carry-less multiplication performs a dot product */
    w0^=_mm_clmulepi64_si128(u20,B00,0);
    w0^=_mm_clmulepi64_si128(u20,B00,0x11);
    w0^=_mm_clmulepi64_si128(u30,B10,0);
    w0^=_mm_clmulepi64_si128(u30,B10,0x11);
    w1^=_mm_clmulepi64_si128(u21,B01,0);
    w1^=_mm_clmulepi64_si128(u21,B01,0x11);
    w1^=_mm_clmulepi64_si128(u31,B11,0);
    w1^=_mm_clmulepi64_si128(u31,B11,0x11);

    /* repeat the previous process */
    B0=_mm_loadu_si128((__m128i*)B+4);
    B1=_mm_loadu_si128((__m128i*)B+5);

    /* even bytes *x^8 */
    B00=_mm_slli_epi16(B0,8);
    B10=_mm_slli_epi16(B1,8);
    /* odd bytes */
    B01=_mm_srli_epi16(B0,8);
    B11=_mm_srli_epi16(B1,8);

    /* each carry-less multiplication performs a dot product */
    w0^=_mm_clmulepi64_si128(u40,B00,0);
    w0^=_mm_clmulepi64_si128(u40,B00,0x11);
    w0^=_mm_clmulepi64_si128(u50,B10,0);
    w0^=_mm_clmulepi64_si128(u50,B10,0x11);
    w1^=_mm_clmulepi64_si128(u41,B01,0);
    w1^=_mm_clmulepi64_si128(u41,B01,0x11);
    w1^=_mm_clmulepi64_si128(u51,B11,0);
    w1^=_mm_clmulepi64_si128(u51,B11,0x11);

    /* repeat the previous process */
    B0=_mm_loadu_si128((__m128i*)B+6);

    /* even bytes *x^8 */
    B00=_mm_slli_epi16(B0,8);
    /* odd bytes */
    B01=_mm_srli_epi16(B0,8);

    /* each carry-less multiplication performs a dot product */
    w0^=_mm_clmulepi64_si128(u60,B00,0);
    w0^=_mm_clmulepi64_si128(u60,B00,0x11);
    w1^=_mm_clmulepi64_si128(u61,B01,0);
    w1^=_mm_clmulepi64_si128(u61,B01,0x11);

    e=_mm_srli_si128(w0,8)^_mm_srli_si128(w1,6);
    REM_CORE_GF256_SSE2(e,S)
    w[i]^=(uint8_t)_mm_cvtsi128_si32(e);

    B+=NCOLS;
  }
}

#undef NROWS
#undef NCOLS
#define NDIM NB_VIN

/*! \brief Vector-matrix product over GF(256): (1*v)*(v*v)=(1*v).
 \details Multiplication of a vector by the transpose of an upper triangular
 matrix.
 \param[out] w A vector over GF(256) of size v, equal to u.(B^T).
 \param[in] u A vector over GF(256) of size v.
 \param[in] B An upper triangular matrix over GF(256) of size v*v.
 \alloc v bytes for u, (v*(v+1))/2 bytes for B, v bytes for w.
 \csttime u, B, w.
*/
void addVecMatProd_Ut_gf256_pclmul(uint8_t *w, const uint8_t *u,
                                               const uint8_t *B)
{
  unsigned int j;

  /* columns of u, rows of B */
  for(j=0;j<NDIM;++j)
  {
    /* dot product of u and the j-th row of B */
    w[j]^=dotProduct_gf256_pclmul(u,B,NDIM-j);
    ++u;
    B+=NDIM-j;
  }
}

#undef NDIM

