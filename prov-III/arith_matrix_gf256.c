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
 Matrix operations over GF(256).
 \author Jocelyn Ryckeghem
*/
#include "arith_matrix_gf256.h"
#include "params.h"
#include "multab_gf256.h"
#include "prov_gf256.h"
#include <emmintrin.h>
#include <tmmintrin.h>
#include <immintrin.h>


/*! \brief Copy a matrix.
 \param[out] C A matrix over GF(256) of size m*(o+1), C=A.
 \param[in] A A matrix over GF(256) of size m*(o+1).
 \req LEN_AV=0 mod 32.
 \alloc LEN_AV bytes for A and C.
 \csttime A, C.
*/
void copyMatrixPad32_gf256_avx2(uint8_t *C, const uint8_t *A)
{
  unsigned int i;

  for(i=0;i<((LEN_AV>>5)-1);i+=2)
  {
    _mm256_storeu_si256((__m256i*)C+i,_mm256_loadu_si256((__m256i*)A+i));
    _mm256_storeu_si256((__m256i*)C+i+1,_mm256_loadu_si256((__m256i*)A+i+1));
  }
  #if LEN_AV&63
    _mm256_storeu_si256((__m256i*)C,_mm256_loadu_si256((__m256i*)A));
  #endif
}

#define NDIM NB_OIL

/*! \brief Add the transpose of the strictly lower triangular part of A to A.
 \param[in,out] A A matrix over GF(256) of size o*o.
 \req o=2 mod 16.
 \alloc o*o bytes for A.
 \csttime A.
*/
static void symmetrizeUpperT_gf256_sse2(uint8_t *A)
{
  __m128i t0,t1,At0,At1,At2,At3,At4,At5,At6,At7,r0,r1,r2,r3,s0,s1,s2,s3;
  uint8_t *A_,*At;
  unsigned int i,j,k;
  uint16_t tmp[16];

  /* for each 16*16 block of the lower triangular part */
  for(i=0;i<(NDIM>>4);++i)
  {
    for(j=0;j<i;++j)
    {
      /* compute the transpose of a block and add it to the corresponding block
         of the upper triangular part */
      At=A+i*(NDIM<<4)+(j<<4);
      A_=A+j*(NDIM<<4)+(i<<4);
      /* take the rows 8 by 8 */
      for(k=0;k<2;++k)
      {
        t0 =_mm_loadu_si128((__m128i*)At);
        At5=_mm_loadu_si128((__m128i*)(At+NDIM));
        At7=_mm_loadu_si128((__m128i*)(At+2*NDIM));
        At0=_mm_loadu_si128((__m128i*)(At+3*NDIM));
        t1 =_mm_loadu_si128((__m128i*)(At+4*NDIM));
        At1=_mm_loadu_si128((__m128i*)(At+5*NDIM));
        At2=_mm_loadu_si128((__m128i*)(At+6*NDIM));
        At4=_mm_loadu_si128((__m128i*)(At+7*NDIM));

        /* 2 */
        At6=_mm_unpacklo_epi8(t0,At5);
        At3=_mm_unpackhi_epi8(t0,At5);
        t0 =_mm_unpacklo_epi8(At7,At0);
        At5=_mm_unpackhi_epi8(At7,At0);
        At7=_mm_unpacklo_epi8(t1,At1);
        At0=_mm_unpackhi_epi8(t1,At1);
        t1 =_mm_unpacklo_epi8(At2,At4);
        At1=_mm_unpackhi_epi8(At2,At4);

        /* 4 */
        At2=_mm_unpacklo_epi16(At6,t0);
        At4=_mm_unpackhi_epi16(At6,t0);
        At6=_mm_unpacklo_epi16(At3,At5);
        t0 =_mm_unpackhi_epi16(At3,At5);
        At3=_mm_unpacklo_epi16(At7,t1);
        At5=_mm_unpackhi_epi16(At7,t1);
        At7=_mm_unpacklo_epi16(At0,At1);
        t1 =_mm_unpackhi_epi16(At0,At1);

        /* 8 */
        At0=_mm_unpacklo_epi32(At2,At3);
        At1=_mm_unpackhi_epi32(At2,At3);
        At2=_mm_unpacklo_epi32(At4,At5);
        At3=_mm_unpackhi_epi32(At4,At5);
        At4=_mm_unpacklo_epi32(At6,At7);
        At5=_mm_unpackhi_epi32(At6,At7);
        At6=_mm_unpacklo_epi32(t0,t1);
        At7=_mm_unpackhi_epi32(t0,t1);

        _mm_storeu_si128((__m128i*)(A_-8),_mm_slli_si128(At0,8)
        ^_mm_loadu_si128((__m128i*)(A_-8)));
        _mm_storeu_si128((__m128i*)(A_+NDIM),_mm_srli_si128(At0,8)
        ^_mm_loadu_si128((__m128i*)(A_+NDIM)));
        _mm_storeu_si128((__m128i*)(A_+2*NDIM-8),_mm_slli_si128(At1,8)
        ^_mm_loadu_si128((__m128i*)(A_+2*NDIM-8)));
        _mm_storeu_si128((__m128i*)(A_+3*NDIM),_mm_srli_si128(At1,8)
        ^_mm_loadu_si128((__m128i*)(A_+3*NDIM)));
        _mm_storeu_si128((__m128i*)(A_+4*NDIM-8),_mm_slli_si128(At2,8)
        ^_mm_loadu_si128((__m128i*)(A_+4*NDIM-8)));
        _mm_storeu_si128((__m128i*)(A_+5*NDIM),_mm_srli_si128(At2,8)
        ^_mm_loadu_si128((__m128i*)(A_+5*NDIM)));
        _mm_storeu_si128((__m128i*)(A_+6*NDIM-8),_mm_slli_si128(At3,8)
        ^_mm_loadu_si128((__m128i*)(A_+6*NDIM-8)));
        _mm_storeu_si128((__m128i*)(A_+7*NDIM),_mm_srli_si128(At3,8)
        ^_mm_loadu_si128((__m128i*)(A_+7*NDIM)));
        _mm_storeu_si128((__m128i*)(A_+8*NDIM-8),_mm_slli_si128(At4,8)
        ^_mm_loadu_si128((__m128i*)(A_+8*NDIM-8)));
        _mm_storeu_si128((__m128i*)(A_+9*NDIM),_mm_srli_si128(At4,8)
        ^_mm_loadu_si128((__m128i*)(A_+9*NDIM)));
        _mm_storeu_si128((__m128i*)(A_+10*NDIM-8),_mm_slli_si128(At5,8)
        ^_mm_loadu_si128((__m128i*)(A_+10*NDIM-8)));
        _mm_storeu_si128((__m128i*)(A_+11*NDIM),_mm_srli_si128(At5,8)
        ^_mm_loadu_si128((__m128i*)(A_+11*NDIM)));
        _mm_storeu_si128((__m128i*)(A_+12*NDIM-8),_mm_slli_si128(At6,8)
        ^_mm_loadu_si128((__m128i*)(A_+12*NDIM-8)));
        _mm_storeu_si128((__m128i*)(A_+13*NDIM),_mm_srli_si128(At6,8)
        ^_mm_loadu_si128((__m128i*)(A_+13*NDIM)));
        _mm_storeu_si128((__m128i*)(A_+14*NDIM-8),_mm_slli_si128(At7,8)
        ^_mm_loadu_si128((__m128i*)(A_+14*NDIM-8)));
        _mm_storeu_si128((__m128i*)(A_+15*NDIM),_mm_srli_si128(At7,8)
        ^_mm_loadu_si128((__m128i*)(A_+15*NDIM)));

        At+=NDIM<<3;
        A_+=8;
      }
    }
    /* diagonal block: At==A_ */
    At=A+i*(NDIM<<4)+(i<<4);
    /* k=1 */
    At+=NDIM<<3;
    At0=_mm_loadu_si128((__m128i*)At);
    At1=_mm_loadu_si128((__m128i*)(At+NDIM));
    At2=_mm_loadu_si128((__m128i*)(At+2*NDIM));
    At3=_mm_loadu_si128((__m128i*)(At+3*NDIM));
    At4=_mm_loadu_si128((__m128i*)(At+4*NDIM));
    At5=_mm_loadu_si128((__m128i*)(At+5*NDIM));
    At6=_mm_loadu_si128((__m128i*)(At+6*NDIM));
    At7=_mm_loadu_si128((__m128i*)(At+7*NDIM));

    /* second square subblock */
    /* 2 */
    r0=_mm_unpackhi_epi8(_mm_setzero_si128(),At1);
    r1=_mm_unpackhi_epi8(At2,At3);
    r2=_mm_unpackhi_epi8(At4,At5);
    r3=_mm_unpackhi_epi8(At6,At7);
    /* 4 */
    r0=_mm_unpacklo_epi16(r0,r1);
    r1=_mm_unpacklo_epi16(r2,r3);
    r2=_mm_unpackhi_epi16(r2,r3);
    /* 8 */
    r3=_mm_unpacklo_epi32(r0,r1);
    r0=_mm_unpackhi_epi32(r0,r1);

    /* update the higher part of At */
    At6^=_mm_slli_si128(_mm_srli_si128(r2,11),15);
    At5^=_mm_slli_si128(_mm_srli_si128(r2,6),14);
    At4^=_mm_slli_si128(_mm_srli_si128(r2,1),13);
    At0^=_mm_slli_si128(r3,8);
    At1^=_mm_slli_si128(_mm_srli_si128(r3,10),10);
    At2^=_mm_slli_si128(_mm_srli_si128(r0,3),11);
    At3^=_mm_slli_si128(_mm_srli_si128(r0,12),12);
    _mm_storeu_si128((__m128i*)At,At0);
    _mm_storeu_si128((__m128i*)(At+NDIM),At1);
    _mm_storeu_si128((__m128i*)(At+2*NDIM),At2);
    _mm_storeu_si128((__m128i*)(At+3*NDIM),At3);
    _mm_storeu_si128((__m128i*)(At+4*NDIM),At4);
    _mm_storeu_si128((__m128i*)(At+5*NDIM),At5);
    _mm_storeu_si128((__m128i*)(At+6*NDIM),At6);

    /* square subblock */
    /* 2 */
    r0=_mm_unpacklo_epi8(At0,At1);
    r1=_mm_unpacklo_epi8(At2,At3);
    r2=_mm_unpacklo_epi8(At4,At5);
    r3=_mm_unpacklo_epi8(At6,At7);
    /* 4 */
    At0=_mm_unpacklo_epi16(r0,r1);
    At1=_mm_unpackhi_epi16(r0,r1);
    At2=_mm_unpacklo_epi16(r2,r3);
    At3=_mm_unpackhi_epi16(r2,r3);
    /* 8 */
    r0=_mm_unpacklo_epi32(At0,At2);
    r1=_mm_unpackhi_epi32(At0,At2);
    r2=_mm_unpacklo_epi32(At1,At3);
    r3=_mm_unpackhi_epi32(At1,At3);

    /* k=0 */
    At-=NDIM<<3;
    At0=_mm_loadu_si128((__m128i*)At);
    At1=_mm_loadu_si128((__m128i*)(At+NDIM));
    At2=_mm_loadu_si128((__m128i*)(At+2*NDIM));
    At3=_mm_loadu_si128((__m128i*)(At+3*NDIM));
    At4=_mm_loadu_si128((__m128i*)(At+4*NDIM));
    At5=_mm_loadu_si128((__m128i*)(At+5*NDIM));
    At6=_mm_loadu_si128((__m128i*)(At+6*NDIM));
    At7=_mm_loadu_si128((__m128i*)(At+7*NDIM));
    At7^=_mm_unpackhi_epi64(_mm_setzero_si128(),r3);
    _mm_storeu_si128((__m128i*)(At+7*NDIM),At7);

    /* first square subblock */
    /* 2 */
    s0=_mm_unpacklo_epi8(_mm_setzero_si128(),At1);
    s1=_mm_unpacklo_epi8(At2,At3);
    s2=_mm_unpacklo_epi8(At4,At5);
    s3=_mm_unpacklo_epi8(At6,At7);
    /* 4 */
    s0=_mm_unpacklo_epi16(s0,s1);
    s1=_mm_unpacklo_epi16(s2,s3);
    s2=_mm_unpackhi_epi16(s2,s3);
    /* 8 */
    s3=_mm_unpacklo_epi32(s0,s1);
    s0=_mm_unpackhi_epi32(s0,s1);

    /* update the lower part of At */
    At6^=_mm_alignr_epi8(r3,_mm_slli_si128(_mm_srli_si128(s2,11),15),8);
    At4^=_mm_alignr_epi8(r2,_mm_slli_si128(_mm_srli_si128(s2,1),13),8);
    At5^=_mm_unpackhi_epi64(_mm_slli_si128(_mm_srli_si128(s2,6),14),r2);
    At0^=_mm_unpacklo_epi64(s3,r0);
    At2^=_mm_unpacklo_epi64(_mm_slli_si128(_mm_srli_si128(s0,3),3),r1);
    At1^=_mm_unpackhi_epi64(_mm_slli_si128(_mm_srli_si128(s3,10),10),r0);
    At3^=_mm_unpackhi_epi64(_mm_slli_si128(_mm_srli_si128(s0,12),12),r1);
    _mm_storeu_si128((__m128i*)At,At0);
    _mm_storeu_si128((__m128i*)(At+NDIM),At1);
    _mm_storeu_si128((__m128i*)(At+2*NDIM),At2);
    _mm_storeu_si128((__m128i*)(At+3*NDIM),At3);
    _mm_storeu_si128((__m128i*)(At+4*NDIM),At4);
    _mm_storeu_si128((__m128i*)(At+5*NDIM),At5);
    _mm_storeu_si128((__m128i*)(At+6*NDIM),At6);
  }

  /* rectangular blocks, contigue on the lower triangular part */
  At=A+(NDIM-(NDIM&15))*NDIM;
  A_=A+NDIM-(NDIM&15);
  for(j=0;j<(NDIM>>4);++j)
  {
    At0=_mm_loadu_si128((__m128i*)At+j);
    At1=_mm_loadu_si128((__m128i*)(At+NDIM)+j);
    _mm_storeu_si128((__m128i*)tmp,_mm_unpacklo_epi8(At0,At1));
    _mm_storeu_si128((__m128i*)tmp+1,_mm_unpackhi_epi8(At0,At1));
    for(i=0;i<16;i+=2)
    {
      ((uint16_t*)A_)[i*(NDIM>>1)]^=tmp[i];
      ((uint16_t*)A_)[(i+1)*(NDIM>>1)]^=tmp[i+1];
    }
    A_+=NDIM<<4;
  }
  /* diagonal block, size 2*2 */
  A_[1]^=A_[NDIM];
}

#if NDIM&31
  #define NDIMR (NDIM&31)
#else
  #define NDIMR 32
#endif

/*! \brief Add the transpose of the strictly lower triangular part of A to A,
 then store the upper triangular part of the result (the diagonal is this of A).
 \param[out] Au An upper triangular matrix over GF(256) of size o*o.
 \param[in] A A matrix over GF(256) of size o*o.
 \req ceil(o/32)=3.
 \alloc o*o bytes for A, (o*(o+1))/2 bytes for Au.
 \csttime A, Au.
*/
void sqrMatrixToUpperT_gf256_avx2(uint8_t *Au, uint8_t *A)
{
  unsigned int i;

  symmetrizeUpperT_gf256_sse2(A);
  for(i=0;i<NDIMR;++i)
  {
    _mm256_storeu_si256((__m256i*)Au,_mm256_loadu_si256((__m256i*)A));
    _mm256_storeu_si256((__m256i*)Au+1,_mm256_loadu_si256((__m256i*)A+1));
    _mm256_storeu_si256((__m256i*)Au+2,_mm256_loadu_si256((__m256i*)A+2));
    A+=NDIM+1;
    Au+=NDIM-i;
  }
  for(;i<(32+NDIMR);++i)
  {
    _mm256_storeu_si256((__m256i*)Au,_mm256_loadu_si256((__m256i*)A));
    _mm256_storeu_si256((__m256i*)Au+1,_mm256_loadu_si256((__m256i*)A+1));
    A+=NDIM+1;
    Au+=NDIM-i;
  }
  for(;i<NDIM-8;++i)
  {
    _mm256_storeu_si256((__m256i*)Au,_mm256_loadu_si256((__m256i*)A));
    A+=NDIM+1;
    Au+=NDIM-i;
  }
  /* optimization for n-byte rows, n<=8 */
  *(uint64_t*)Au=*(uint64_t*)A;
  A+=NDIM+1;
  Au+=8;
  *(uint64_t*)Au=*(uint64_t*)A;
  A+=NDIM+1;
  Au+=7;
  *(uint64_t*)Au=*(uint64_t*)A;
  A+=NDIM+1;
  Au+=6;
  *(uint64_t*)Au=*(uint64_t*)A;
  A+=NDIM+1;
  Au+=5;
  *(uint32_t*)Au=*(uint32_t*)A;
  A+=NDIM+1;
  Au+=4;
  *(uint32_t*)Au=*(uint32_t*)A;
  A+=NDIM+1;
  Au+=3;
  *(uint16_t*)Au=*(uint16_t*)A;
  A+=NDIM+1;
  Au+=2;
  *Au=*A;
}

#undef NDIM
#undef NDIMR


/*************** Variable-time use of the input left matrix ***************/
#define NROWS NB_VIN
#define NCOLS NB_OIL

/*! \brief Multiply an upper triangular matrix by a matrix, then add it to C,
 over GF(256): (v*o)+=(v*v)*(v*o).
 \details C is erasing itself during the algorithm if NCOLS is not a multiple
 of 32. So, we load the first 32 bytes of the next row of C before to store the
 current row.
 \param[in,out] C A matrix over GF(256) of size v*o, the result is C+A*B.
 \param[in] A An upper triangular matrix over GF(256) of size v*v.
 \param[in] B A matrix over GF(256) of size v*o.
 \req ceil(o/32)=3.
 \alloc (v*(v+1))/2 bytes for A, v*o+PADDING32(o) bytes for B,
 v*o+32 bytes for C.
 \csttime B and C.
 \vartime A.
*/
void addMatrixMul_UPP_mulTab_gf256_avx2(uint8_t *C, const uint8_t *A,
                                                    const uint8_t *B)
{
  const __m256i mask_0f=_mm256_set1_epi8(15);
  __m256i T,T0,T4,C0,C1,C2,B0,B1,B2;
  const uint8_t *Bj;
  unsigned int i,j;

  C0=_mm256_loadu_si256((__m256i*)C);
  /* rows of A */
  for(i=0;i<NROWS;++i)
  {
    Bj=B;
    C1=_mm256_loadu_si256((__m256i*)C+1);
    C2=_mm256_loadu_si256((__m256i*)C+2);
    /* columns of A, rows of B */
    for(j=i;j<NROWS;++j)
    {
      /* T=Tl Th, T0=Tl Tl, T4=Th Th */
      T=_mm256_loadu_si256((__m256i*)multab_x0_x4_gf256+(*A));
      B0=_mm256_loadu_si256((__m256i*)Bj);
      B1=_mm256_loadu_si256((__m256i*)Bj+1);
      B2=_mm256_loadu_si256((__m256i*)Bj+2);

      T0=_mm256_permute4x64_epi64(T,0x44);
      T4=_mm256_permute4x64_epi64(T,0xee);

      C0^=_mm256_shuffle_epi8(T0,B0&mask_0f);
      C1^=_mm256_shuffle_epi8(T0,B1&mask_0f);
      C2^=_mm256_shuffle_epi8(T0,B2&mask_0f);
      C0^=_mm256_shuffle_epi8(T4,_mm256_srli_epi16(B0,4)&mask_0f);
      C1^=_mm256_shuffle_epi8(T4,_mm256_srli_epi16(B1,4)&mask_0f);
      C2^=_mm256_shuffle_epi8(T4,_mm256_srli_epi16(B2,4)&mask_0f);
      ++A;
      Bj+=NCOLS;
    }
    _mm256_storeu_si256((__m256i*)C,C0);
    _mm256_storeu_si256((__m256i*)C+1,C1);
    /* load the data before to erase it with the last store */
    C0=_mm256_loadu_si256((__m256i*)(C+NCOLS));
    _mm256_storeu_si256((__m256i*)C+2,C2);
    B+=NCOLS;
    C+=NCOLS;
  }
  _mm256_storeu_si256((__m256i*)C,C0);
}

#undef NROWS
#undef NCOLS
#define NROWS NB_VIN
#define NCOLS NB_OIL

/*! \brief Multiply the transpose of an upper triangular matrix by a matrix,
 then add it to C, over GF(256): (v*o)+=(v*v)*(v*o).
 \details C is erasing itself during the algorithm if NCOLS is not a multiple
 of 32. So, we load the first 32 bytes of the next row of C before to store the
 current row.
 \param[in,out] C A matrix over GF(256) of size v*o, the result is C+(A^T)*B.
 \param[in] A An upper triangular matrix over GF(256) of size v*v.
 \param[in] B A matrix over GF(256) of size v*o.
 \req ceil(o/32)=3.
 \alloc (v*(v+1))/2 bytes for A, v*o+PADDING32(o) bytes for B,
 v*o+32 bytes for C.
 \csttime B and C.
 \vartime A.
*/
void addMatrixMul_UtPP_mulTab_gf256_avx2(uint8_t *C, const uint8_t *A,
                                                     const uint8_t *B)
{
  const __m256i mask_0f=_mm256_set1_epi8(15);
  __m256i T,T0,T4,C0,C1,C2,B0,B1,B2;
  const uint8_t *Aij,*Bj;
  unsigned int i,j;

  C0=_mm256_loadu_si256((__m256i*)C);
  /* rows of A */
  for(i=0;i<NROWS;++i)
  {
    Aij=A;
    Bj=B;
    C1=_mm256_loadu_si256((__m256i*)C+1);
    C2=_mm256_loadu_si256((__m256i*)C+2);
    /* columns of A, rows of B */
    for(j=0;j<=i;++j)
    {
      /* T=Tl Th, T0=Tl Tl, T4=Th Th */
      T=_mm256_loadu_si256((__m256i*)multab_x0_x4_gf256+(*Aij));
      B0=_mm256_loadu_si256((__m256i*)Bj);
      B1=_mm256_loadu_si256((__m256i*)Bj+1);
      B2=_mm256_loadu_si256((__m256i*)Bj+2);

      T0=_mm256_permute4x64_epi64(T,0x44);
      T4=_mm256_permute4x64_epi64(T,0xee);

      C0^=_mm256_shuffle_epi8(T0,B0&mask_0f);
      C1^=_mm256_shuffle_epi8(T0,B1&mask_0f);
      C2^=_mm256_shuffle_epi8(T0,B2&mask_0f);
      C0^=_mm256_shuffle_epi8(T4,_mm256_srli_epi16(B0,4)&mask_0f);
      C1^=_mm256_shuffle_epi8(T4,_mm256_srli_epi16(B1,4)&mask_0f);
      C2^=_mm256_shuffle_epi8(T4,_mm256_srli_epi16(B2,4)&mask_0f);
      Aij+=NROWS-1-j;
      Bj+=NCOLS;
    }
    _mm256_storeu_si256((__m256i*)C,C0);
    _mm256_storeu_si256((__m256i*)C+1,C1);
    /* load the data before to erase it with the last store */
    C0=_mm256_loadu_si256((__m256i*)(C+NCOLS));
    _mm256_storeu_si256((__m256i*)C+2,C2);
    ++A;
    C+=NCOLS;
  }
  _mm256_storeu_si256((__m256i*)C,C0);
}

#undef NROWS
#undef NCOLS


/*************** Constant-time use of the input left matrix ***************/
#define NROWS NB_VIN
#define NCOLS NB_OIL

/*! \brief Constant-time precomputation of multab_x0_x4_gf256[A[j][i]].
 \param[out] T The lookup table corresponding to the transpose of A.
 \param[in] A A matrix over GF(256) of size v*o.
 \alloc v*o bytes for A, 32v*o bytes for T.
 \csttime A, T.
*/
void matT_to_mulTab_gf256_avx2(uint8_t *T, const uint8_t *A)
{
  const __m256i x0=_mm256_loadu_si256((__m256i*)mulxtab_x0_x4_gf256);
  const __m256i x1=_mm256_loadu_si256((__m256i*)mulxtab_x0_x4_gf256+1);
  const __m256i x2=_mm256_loadu_si256((__m256i*)mulxtab_x0_x4_gf256+2);
  const __m256i x3=_mm256_loadu_si256((__m256i*)mulxtab_x0_x4_gf256+3);
  const __m256i x4=_mm256_loadu_si256((__m256i*)mulxtab_x0_x4_gf256+4);
  const __m256i x5=_mm256_loadu_si256((__m256i*)mulxtab_x0_x4_gf256+5);
  const __m256i x6=_mm256_loadu_si256((__m256i*)mulxtab_x0_x4_gf256+6);
  const __m256i x7=_mm256_loadu_si256((__m256i*)mulxtab_x0_x4_gf256+7);
  const __m256i m_0f=_mm256_set1_epi8(15);
  const __m256i m_ff00=_mm256_slli_epi16(~_mm256_setzero_si256(),8);
  const __m256i m_ffff0000=_mm256_slli_epi32(~_mm256_setzero_si256(),16);
  const __m256i m_ffffffff00000000=_mm256_slli_epi64(~_mm256_setzero_si256(),
                                                     32);
  const __m256i mh=_mm256_slli_si256(~_mm256_setzero_si256(),8);
  __m256i A0,A1,r;
  unsigned int i,j;

  for(j=0;j<NCOLS;++j)
  {
    for(i=0;i<NROWS;++i)
    {
      A0=_mm256_set1_epi8(A[i*NCOLS+j]);
      r=_mm256_cmpgt_epi8(_mm256_setzero_si256(),A0)&x7;
      A1=_mm256_srli_epi16(A0,4)&m_0f;
      A0&=m_0f;
      /* dot product */
      r^=_mm256_shuffle_epi8(m_ff00,A0)&x0;
      r^=_mm256_shuffle_epi8(m_ff00,A1)&x4;
      r^=_mm256_shuffle_epi8(m_ffff0000,A0)&x1;
      r^=_mm256_shuffle_epi8(m_ffff0000,A1)&x5;
      r^=_mm256_shuffle_epi8(m_ffffffff00000000,A0)&x2;
      r^=_mm256_shuffle_epi8(m_ffffffff00000000,A1)&x6;
      r^=_mm256_shuffle_epi8(mh,A0)&x3;
      _mm256_storeu_si256((__m256i*)T,r);
      T+=32;
    }
  }
}

#undef NROWS
#undef NCOLS
#define NROWS NB_VIN
#define NCOLS NB_OIL

/*! \brief Multiply two matrices over GF(256): (o*v)*(v*o)=(o*o).
 \param[out] C A square matrix over GF(256) of size o*o, equal to (A^T)*B.
 \param[in] T_A A table, generated by matT_to_mulTab_*(T_A,A).
 \param[in] B A matrix over GF(256) of size v*o.
 \req ceil(o/32)=3.
 \alloc 32v*o bytes for A, v*o+PADDING32(o) bytes for B,
        o*o+PADDING32(o) bytes for C.
 \csttime T_A, B and C.
*/
void matrixMul_mulTabA_gf256_avx2(uint8_t *C, const uint8_t *T_A,
                                              const uint8_t *B)
{
  const __m256i mask_0f=_mm256_set1_epi8(15);
  __m256i T,T0,T4,C0,C1,C2,B0,B1,B2;
  const uint8_t *Bj;
  unsigned int i,j;

  /* rows of A */
  for(i=0;i<NCOLS;++i)
  {
    Bj=B;
    C0=_mm256_setzero_si256();
    C1=_mm256_setzero_si256();
    C2=_mm256_setzero_si256();
    /* columns of A, rows of B */
    for(j=0;j<NROWS;++j)
    {
      /* T=Tl Th, T0=Tl Tl, T4=Th Th */
      T=_mm256_loadu_si256((__m256i*)T_A+j);
      B0=_mm256_loadu_si256((__m256i*)Bj);
      B1=_mm256_loadu_si256((__m256i*)Bj+1);
      B2=_mm256_loadu_si256((__m256i*)Bj+2);

      T0=_mm256_permute4x64_epi64(T,0x44);
      T4=_mm256_permute4x64_epi64(T,0xee);

      C0^=_mm256_shuffle_epi8(T0,B0&mask_0f);
      C1^=_mm256_shuffle_epi8(T0,B1&mask_0f);
      C2^=_mm256_shuffle_epi8(T0,B2&mask_0f);
      C0^=_mm256_shuffle_epi8(T4,_mm256_srli_epi16(B0,4)&mask_0f);
      C1^=_mm256_shuffle_epi8(T4,_mm256_srli_epi16(B1,4)&mask_0f);
      C2^=_mm256_shuffle_epi8(T4,_mm256_srli_epi16(B2,4)&mask_0f);
      Bj+=NCOLS;
    }
    _mm256_storeu_si256((__m256i*)C,C0);
    _mm256_storeu_si256((__m256i*)C+1,C1);
    _mm256_storeu_si256((__m256i*)C+2,C2);
    C+=NCOLS;
    T_A+=NROWS<<5;
  }
}

#undef NROWS
#undef NCOLS

