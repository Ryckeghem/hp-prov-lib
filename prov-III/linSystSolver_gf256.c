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
 Solve a linear system over GF(256).
 \author Jocelyn Ryckeghem
*/
#include "linSystSolver_gf256.h"
#include "params.h"
#include "lookup.h"
#include "invtab_gf256.h"
#include "multab_gf256.h"
#include "dotProduct_gf256.h"
#include "prov_gf256.h"
#include <immintrin.h>


#define NROWS NB_EQ

static void elimStep3_cstTime_gf256_avx2(uint8_t *A, uint8_t *row,
                                                     unsigned int j)
{
  const __m256i mask_0f=_mm256_set1_epi8(15);
  const __m256i x0=_mm256_loadu_si256((__m256i*)mulxtab_x0_x4_gf256);
  const __m256i x1=_mm256_loadu_si256((__m256i*)mulxtab_x0_x4_gf256+1);
  const __m256i x2=_mm256_loadu_si256((__m256i*)mulxtab_x0_x4_gf256+2);
  const __m256i x3=_mm256_loadu_si256((__m256i*)mulxtab_x0_x4_gf256+3);
  const __m256i x4=_mm256_loadu_si256((__m256i*)mulxtab_x0_x4_gf256+4);
  const __m256i x5=_mm256_loadu_si256((__m256i*)mulxtab_x0_x4_gf256+5);
  const __m256i x6=_mm256_loadu_si256((__m256i*)mulxtab_x0_x4_gf256+6);
  const __m256i x7=_mm256_loadu_si256((__m256i*)mulxtab_x0_x4_gf256+7);
  __m256i A0,T,T0,T4,r0,r1,r2,B00,B01,B10,B11,B20,B21;
  unsigned int i;

  /* inverse the pivot row, the pivot is row[j] */
  A0=_mm256_set1_epi8(lookup256_r2_avx2(invtab_gf256,row[j]));
  /* dot product */
  T =_mm256_srai_epi32(A0,31)&x7;
  T^=_mm256_srai_epi32(_mm256_slli_epi32(A0,1),31)&x6;
  T^=_mm256_srai_epi32(_mm256_slli_epi32(A0,2),31)&x5;
  T^=_mm256_srai_epi32(_mm256_slli_epi32(A0,3),31)&x4;
  T^=_mm256_srai_epi32(_mm256_slli_epi32(A0,4),31)&x3;
  T^=_mm256_srai_epi32(_mm256_slli_epi32(A0,5),31)&x2;
  T^=_mm256_srai_epi32(_mm256_slli_epi32(A0,6),31)&x1;
  T^=_mm256_srai_epi32(_mm256_slli_epi32(A0,7),31)&x0;

  B00=_mm256_loadu_si256((__m256i*)row);
  B10=_mm256_loadu_si256((__m256i*)row+1);
  B20=_mm256_loadu_si256((__m256i*)row+2);
  B01=_mm256_srli_epi16(B00,4);
  B11=_mm256_srli_epi16(B10,4);
  B21=_mm256_srli_epi16(B20,4);
  B00&=mask_0f;
  B10&=mask_0f;
  B20&=mask_0f;
  B01&=mask_0f;
  B11&=mask_0f;
  B21&=mask_0f;

  /* T=Tl Th, T0=Tl Tl, T4=Th Th */
  T0=_mm256_permute4x64_epi64(T,0x44);
  T4=_mm256_permute4x64_epi64(T,0xee);

  /* row *= (pivot of row)^(-1) */
  B00=_mm256_shuffle_epi8(T0,B00)^_mm256_shuffle_epi8(T4,B01);
  B10=_mm256_shuffle_epi8(T0,B10)^_mm256_shuffle_epi8(T4,B11);
  B20=_mm256_shuffle_epi8(T0,B20)^_mm256_shuffle_epi8(T4,B21);
  _mm256_storeu_si256((__m256i*)row,B00);
  _mm256_storeu_si256((__m256i*)row+1,B10);
  _mm256_storeu_si256((__m256i*)row+2,B20);
  B01=_mm256_srli_epi16(B00,4);
  B11=_mm256_srli_epi16(B10,4);
  B21=_mm256_srli_epi16(B20,4);
  B00&=mask_0f;
  B10&=mask_0f;
  B20&=mask_0f;
  B01&=mask_0f;
  B11&=mask_0f;
  B21&=mask_0f;
  for(i=0;i<NROWS;++i)
  {
    /* pivot of the i-th row of A */
    A0=_mm256_set1_epi8(A[j]);
    /* dot product */
    T =_mm256_srai_epi32(A0,31)&x7;
    T^=_mm256_srai_epi32(_mm256_slli_epi32(A0,1),31)&x6;
    T^=_mm256_srai_epi32(_mm256_slli_epi32(A0,2),31)&x5;
    T^=_mm256_srai_epi32(_mm256_slli_epi32(A0,3),31)&x4;
    T^=_mm256_srai_epi32(_mm256_slli_epi32(A0,4),31)&x3;
    T^=_mm256_srai_epi32(_mm256_slli_epi32(A0,5),31)&x2;
    T^=_mm256_srai_epi32(_mm256_slli_epi32(A0,6),31)&x1;
    T^=_mm256_srai_epi32(_mm256_slli_epi32(A0,7),31)&x0;

    r0=_mm256_loadu_si256((__m256i*)A);
    r1=_mm256_loadu_si256((__m256i*)A+1);
    r2=_mm256_loadu_si256((__m256i*)A+2);

    /* T=Tl Th, T0=Tl Tl, T4=Th Th */
    T0=_mm256_permute4x64_epi64(T,0x44);
    T4=_mm256_permute4x64_epi64(T,0xee);

    /* i-th row of A ^= pivot of A * row */
    r0^=_mm256_shuffle_epi8(T0,B00);
    r1^=_mm256_shuffle_epi8(T0,B10);
    r2^=_mm256_shuffle_epi8(T0,B20);
    r0^=_mm256_shuffle_epi8(T4,B01);
    r1^=_mm256_shuffle_epi8(T4,B11);
    r2^=_mm256_shuffle_epi8(T4,B21);

    _mm256_storeu_si256((__m256i*)A,r0);
    _mm256_storeu_si256((__m256i*)A+1,r1);
    _mm256_storeu_si256((__m256i*)A+2,r2);

    A+=SIZE_ROW_Av;
  }
}

static void elimStep2_cstTime_gf256_avx2(uint8_t *A, uint8_t *row,
                                                     unsigned int j)
{
  const __m256i mask_0f=_mm256_set1_epi8(15);
  const __m256i x0=_mm256_loadu_si256((__m256i*)mulxtab_x0_x4_gf256);
  const __m256i x1=_mm256_loadu_si256((__m256i*)mulxtab_x0_x4_gf256+1);
  const __m256i x2=_mm256_loadu_si256((__m256i*)mulxtab_x0_x4_gf256+2);
  const __m256i x3=_mm256_loadu_si256((__m256i*)mulxtab_x0_x4_gf256+3);
  const __m256i x4=_mm256_loadu_si256((__m256i*)mulxtab_x0_x4_gf256+4);
  const __m256i x5=_mm256_loadu_si256((__m256i*)mulxtab_x0_x4_gf256+5);
  const __m256i x6=_mm256_loadu_si256((__m256i*)mulxtab_x0_x4_gf256+6);
  const __m256i x7=_mm256_loadu_si256((__m256i*)mulxtab_x0_x4_gf256+7);
  __m256i A0,T,T0,T4,r0,r1,B00,B01,B10,B11;
  unsigned int i;

  /* inverse the pivot row, the pivot is row[j] */
  A0=_mm256_set1_epi8(lookup256_r2_avx2(invtab_gf256,row[j]));
  /* dot product */
  T =_mm256_srai_epi32(A0,31)&x7;
  T^=_mm256_srai_epi32(_mm256_slli_epi32(A0,1),31)&x6;
  T^=_mm256_srai_epi32(_mm256_slli_epi32(A0,2),31)&x5;
  T^=_mm256_srai_epi32(_mm256_slli_epi32(A0,3),31)&x4;
  T^=_mm256_srai_epi32(_mm256_slli_epi32(A0,4),31)&x3;
  T^=_mm256_srai_epi32(_mm256_slli_epi32(A0,5),31)&x2;
  T^=_mm256_srai_epi32(_mm256_slli_epi32(A0,6),31)&x1;
  T^=_mm256_srai_epi32(_mm256_slli_epi32(A0,7),31)&x0;

  B00=_mm256_loadu_si256((__m256i*)row);
  B10=_mm256_loadu_si256((__m256i*)row+1);
  B01=_mm256_srli_epi16(B00,4);
  B11=_mm256_srli_epi16(B10,4);
  B00&=mask_0f;
  B10&=mask_0f;
  B01&=mask_0f;
  B11&=mask_0f;

  /* T=Tl Th, T0=Tl Tl, T4=Th Th */
  T0=_mm256_permute4x64_epi64(T,0x44);
  T4=_mm256_permute4x64_epi64(T,0xee);

  /* row *= (pivot of row)^(-1) */
  B00=_mm256_shuffle_epi8(T0,B00)^_mm256_shuffle_epi8(T4,B01);
  B10=_mm256_shuffle_epi8(T0,B10)^_mm256_shuffle_epi8(T4,B11);
  _mm256_storeu_si256((__m256i*)row,B00);
  _mm256_storeu_si256((__m256i*)row+1,B10);
  B01=_mm256_srli_epi16(B00,4);
  B11=_mm256_srli_epi16(B10,4);
  B00&=mask_0f;
  B10&=mask_0f;
  B01&=mask_0f;
  B11&=mask_0f;
  for(i=0;i<NROWS;++i)
  {
    /* pivot of the i-th row of A */
    A0=_mm256_set1_epi8(A[j]);
    /* dot product */
    T =_mm256_srai_epi32(A0,31)&x7;
    T^=_mm256_srai_epi32(_mm256_slli_epi32(A0,1),31)&x6;
    T^=_mm256_srai_epi32(_mm256_slli_epi32(A0,2),31)&x5;
    T^=_mm256_srai_epi32(_mm256_slli_epi32(A0,3),31)&x4;
    T^=_mm256_srai_epi32(_mm256_slli_epi32(A0,4),31)&x3;
    T^=_mm256_srai_epi32(_mm256_slli_epi32(A0,5),31)&x2;
    T^=_mm256_srai_epi32(_mm256_slli_epi32(A0,6),31)&x1;
    T^=_mm256_srai_epi32(_mm256_slli_epi32(A0,7),31)&x0;

    r0=_mm256_loadu_si256((__m256i*)A);
    r1=_mm256_loadu_si256((__m256i*)A+1);

    /* T=Tl Th, T0=Tl Tl, T4=Th Th */
    T0=_mm256_permute4x64_epi64(T,0x44);
    T4=_mm256_permute4x64_epi64(T,0xee);

    /* i-th row of A ^= pivot of A * row */
    r0^=_mm256_shuffle_epi8(T0,B00);
    r1^=_mm256_shuffle_epi8(T0,B10);
    r0^=_mm256_shuffle_epi8(T4,B01);
    r1^=_mm256_shuffle_epi8(T4,B11);

    _mm256_storeu_si256((__m256i*)A,r0);
    _mm256_storeu_si256((__m256i*)A+1,r1);

    A+=SIZE_ROW_Av;
  }
}

static void elimStep1_cstTime_gf256_avx2(uint8_t *A, uint8_t *row,
                                                     unsigned int j)
{
  const __m256i mask_0f=_mm256_set1_epi8(15);
  const __m256i x0=_mm256_loadu_si256((__m256i*)mulxtab_x0_x4_gf256);
  const __m256i x1=_mm256_loadu_si256((__m256i*)mulxtab_x0_x4_gf256+1);
  const __m256i x2=_mm256_loadu_si256((__m256i*)mulxtab_x0_x4_gf256+2);
  const __m256i x3=_mm256_loadu_si256((__m256i*)mulxtab_x0_x4_gf256+3);
  const __m256i x4=_mm256_loadu_si256((__m256i*)mulxtab_x0_x4_gf256+4);
  const __m256i x5=_mm256_loadu_si256((__m256i*)mulxtab_x0_x4_gf256+5);
  const __m256i x6=_mm256_loadu_si256((__m256i*)mulxtab_x0_x4_gf256+6);
  const __m256i x7=_mm256_loadu_si256((__m256i*)mulxtab_x0_x4_gf256+7);
  __m256i A0,T,T0,T4,r0,B00,B01;
  unsigned int i;

  /* inverse the pivot row, the pivot is row[j] */
  A0=_mm256_set1_epi8(lookup256_r2_avx2(invtab_gf256,row[j]));
  /* dot product */
  T =_mm256_srai_epi32(A0,31)&x7;
  T^=_mm256_srai_epi32(_mm256_slli_epi32(A0,1),31)&x6;
  T^=_mm256_srai_epi32(_mm256_slli_epi32(A0,2),31)&x5;
  T^=_mm256_srai_epi32(_mm256_slli_epi32(A0,3),31)&x4;
  T^=_mm256_srai_epi32(_mm256_slli_epi32(A0,4),31)&x3;
  T^=_mm256_srai_epi32(_mm256_slli_epi32(A0,5),31)&x2;
  T^=_mm256_srai_epi32(_mm256_slli_epi32(A0,6),31)&x1;
  T^=_mm256_srai_epi32(_mm256_slli_epi32(A0,7),31)&x0;

  B00=_mm256_loadu_si256((__m256i*)row);
  B01=_mm256_srli_epi16(B00,4);
  B00&=mask_0f;
  B01&=mask_0f;

  /* T=Tl Th, T0=Tl Tl, T4=Th Th */
  T0=_mm256_permute4x64_epi64(T,0x44);
  T4=_mm256_permute4x64_epi64(T,0xee);

  /* row *= (pivot of row)^(-1) */
  B00=_mm256_shuffle_epi8(T0,B00)^_mm256_shuffle_epi8(T4,B01);
  _mm256_storeu_si256((__m256i*)row,B00);
  B01=_mm256_srli_epi16(B00,4);
  B00&=mask_0f;
  B01&=mask_0f;
  for(i=0;i<NROWS;++i)
  {
    /* pivot of the i-th row of A */
    A0=_mm256_set1_epi8(A[j]);
    /* dot product */
    T =_mm256_srai_epi32(A0,31)&x7;
    T^=_mm256_srai_epi32(_mm256_slli_epi32(A0,1),31)&x6;
    T^=_mm256_srai_epi32(_mm256_slli_epi32(A0,2),31)&x5;
    T^=_mm256_srai_epi32(_mm256_slli_epi32(A0,3),31)&x4;
    T^=_mm256_srai_epi32(_mm256_slli_epi32(A0,4),31)&x3;
    T^=_mm256_srai_epi32(_mm256_slli_epi32(A0,5),31)&x2;
    T^=_mm256_srai_epi32(_mm256_slli_epi32(A0,6),31)&x1;
    T^=_mm256_srai_epi32(_mm256_slli_epi32(A0,7),31)&x0;

    r0=_mm256_loadu_si256((__m256i*)A);

    /* T=Tl Th, T0=Tl Tl, T4=Th Th */
    T0=_mm256_permute4x64_epi64(T,0x44);
    T4=_mm256_permute4x64_epi64(T,0xee);

    /* i-th row of A ^= pivot of A * row */
    r0^=_mm256_shuffle_epi8(T0,B00);
    r0^=_mm256_shuffle_epi8(T4,B01);

    _mm256_storeu_si256((__m256i*)A,r0);

    A+=SIZE_ROW_Av;
  }
}

#define NCOLS NB_OIL
#if NROWS&31
  #define NROWSR (NROWS&31)
#else
  #define NROWSR 32
#endif
#if NCOLS&31
  #define NCOLSR (NCOLS&31)
#else
  #define NCOLSR 32
#endif
#if (NCOLS+1)&31
  #define NCOLSR_Ab ((NCOLS+1)&31)
#else
  #define NCOLSR_Ab 32
#endif

/*! \brief Apply the Gauss-Jordan elimination on the augmented matrix A|b
 over GF(256).
 \param[in,out] A A matrix over GF(256) of size m*o. The output is the reduced
 row echelon form of A.
 \param[in,out] b A vector over GF(256) of size m.
 \return 1 if the system is inconsistent, 0 otherwise.
 \req The inverse of 0 in GF(256) is different from 0.
 \req m<128 because of _mm256_cmpgt_epi8.
 \req ceil((o+1)/32)=3.
 \alloc m*SIZE_ROW_Av bytes for A, m bytes for b.
 \csttime A, b, ret.
*/
int gaussJordanElim_cstTime_gf256_avx2(uint8_t *A, uint8_t *b)
{
  const __m256i mask_01=_mm256_set1_epi8(1);
  __m256i pr0,pr1,pr2,A0,A1,A2,b1,m_pivot,m_nopivot,i8,nb_pivots;
  uint64_t b0;
  unsigned int i,j;
  uint8_t pivot_row[SIZE_ROW_Av];

  /* augmented matrix: A,b --> A|b */
  for(i=0;i<NROWS;++i)
  {
    A[i*SIZE_ROW_Av+NCOLS]=b[i];
    for(j=NCOLS+1;j<SIZE_ROW_Av;++j)
    {
      A[i*SIZE_ROW_Av+j]=0;
    }
  }

  nb_pivots=_mm256_setzero_si256();
  /* for each column, elimination of the column by the pivot */
  for(j=0;j<NCOLSR_Ab;++j)
  {
    /* generate a row with a non-zero pivot, if a non-zero pivot exists */
    /* set to 0 as soon as the pivot row is found */
    m_nopivot=~_mm256_setzero_si256();
    i8=_mm256_setzero_si256();
    pr0=_mm256_setzero_si256();
    pr1=_mm256_setzero_si256();
    pr2=_mm256_setzero_si256();
    for(i=0;i<NROWS;++i)
    {
      A0=_mm256_set1_epi8(A[i*SIZE_ROW_Av+j]);
      A0=_mm256_cmpeq_epi8(A0,_mm256_setzero_si256());
      /* i>=nb_pivots <=> i+1>nb_pivots */
      i8=_mm256_add_epi8(i8,mask_01);
      /* note that i8<=NROWS and nb_pivots<=NROWS */
      m_pivot=_mm256_cmpgt_epi8(i8,nb_pivots);

      //b1=-((i>=nb_pivots)&m_nopivot);
      b1=m_pivot&m_nopivot;
      m_nopivot=_mm256_andnot_si256(_mm256_andnot_si256(A0,m_pivot),m_nopivot);
      pr0^=_mm256_loadu_si256((__m256i*)(A+i*SIZE_ROW_Av))&b1;
      pr1^=_mm256_loadu_si256((__m256i*)(A+i*SIZE_ROW_Av+32))&b1;
      pr2^=_mm256_loadu_si256((__m256i*)(A+i*SIZE_ROW_Av+64))&b1;
    }
    /* we store nb_pivots+1 in this implementation */
    nb_pivots=_mm256_add_epi8(nb_pivots,mask_01);
    _mm256_storeu_si256((__m256i*)pivot_row,pr0);
    _mm256_storeu_si256((__m256i*)pivot_row+1,pr1);
    _mm256_storeu_si256((__m256i*)pivot_row+2,pr2);

    /* elimination of the column by the pivot */
    /* A[i]^=A[i][j]*(pivot_row*pivot_row[j]^(-1)) for each row */
    elimStep3_cstTime_gf256_avx2(A,pivot_row,j);

    /* store the pivot row in A */
    pr0=_mm256_loadu_si256((__m256i*)pivot_row);
    pr1=_mm256_loadu_si256((__m256i*)pivot_row+1);
    pr2=_mm256_loadu_si256((__m256i*)pivot_row+2);
    i8=_mm256_setzero_si256();
    /* note that necessarily, nb_pivots<=j */
    #if (NCOLSR_Ab-1)<NROWS
    for(i=0;i<=j;++i)
    #else
    for(i=0;i<=((j<NROWS)?j:NROWS-1);++i)
    #endif
    {
      /* we store i+1 and nb_pivots+1 in this implementation */
      i8=_mm256_add_epi8(i8,mask_01);
      b1=_mm256_cmpeq_epi8(i8,nb_pivots);

      A0=_mm256_loadu_si256((__m256i*)(A+i*SIZE_ROW_Av));
      A1=_mm256_loadu_si256((__m256i*)(A+i*SIZE_ROW_Av+32));
      A2=_mm256_loadu_si256((__m256i*)(A+i*SIZE_ROW_Av+64));
      /* store  the pivot row if b1!=0 */
      A0^=(A0^pr0)&b1;
      A1^=(A1^pr1)&b1;
      A2^=(A2^pr2)&b1;
      _mm256_storeu_si256((__m256i*)(A+i*SIZE_ROW_Av),A0);
      _mm256_storeu_si256((__m256i*)(A+i*SIZE_ROW_Av+32),A1);
      _mm256_storeu_si256((__m256i*)(A+i*SIZE_ROW_Av+64),A2);
    }
    /* add -1 if no pivot was found */
    nb_pivots=_mm256_add_epi8(nb_pivots,m_nopivot);
  }

  A+=NCOLSR_Ab;
  /* j-=NCOLSR_Ab; */
  for(j=0;j<32;++j)
  {
    /* generate a row with a non-zero pivot, if a non-zero pivot exists */
    /* set to 0 as soon as the pivot row is found */
    m_nopivot=~_mm256_setzero_si256();
    i8=_mm256_setzero_si256();
    pr0=_mm256_setzero_si256();
    pr1=_mm256_setzero_si256();
    for(i=0;i<NROWS;++i)
    {
      A0=_mm256_set1_epi8(A[i*SIZE_ROW_Av+j]);
      A0=_mm256_cmpeq_epi8(A0,_mm256_setzero_si256());
      /* i>=nb_pivots <=> i+1>nb_pivots */
      i8=_mm256_add_epi8(i8,mask_01);
      /* note that i8<=NROWS and nb_pivots<=NROWS */
      m_pivot=_mm256_cmpgt_epi8(i8,nb_pivots);

      //b1=-((i>=nb_pivots)&m_nopivot);
      b1=m_pivot&m_nopivot;
      m_nopivot=_mm256_andnot_si256(_mm256_andnot_si256(A0,m_pivot),m_nopivot);
      pr0^=_mm256_loadu_si256((__m256i*)(A+i*SIZE_ROW_Av))&b1;
      pr1^=_mm256_loadu_si256((__m256i*)(A+i*SIZE_ROW_Av+32))&b1;
    }
    /* we store nb_pivots+1 in this implementation */
    nb_pivots=_mm256_add_epi8(nb_pivots,mask_01);
    _mm256_storeu_si256((__m256i*)pivot_row,pr0);
    _mm256_storeu_si256((__m256i*)pivot_row+1,pr1);

    /* elimination of the column by the pivot */
    /* A[i]^=A[i][j]*(pivot_row*pivot_row[j]^(-1)) for each row */
    elimStep2_cstTime_gf256_avx2(A,pivot_row,j);

    /* store the pivot row in A */
    pr0=_mm256_loadu_si256((__m256i*)pivot_row);
    pr1=_mm256_loadu_si256((__m256i*)pivot_row+1);
    i8=_mm256_setzero_si256();
    /* note that necessarily, nb_pivots<=j */
    #if (31+NCOLSR_Ab)<NROWS
    for(i=0;i<=(j+NCOLSR_Ab);++i)
    #elif NCOLSR_Ab>=NROWS
    for(i=0;i<NROWS;++i)
    #else
    for(i=0;i<=(((j+NCOLSR_Ab)<NROWS)?j+NCOLSR_Ab:NROWS-1);++i)
    #endif
    {
      /* we store i+1 and nb_pivots+1 in this implementation */
      i8=_mm256_add_epi8(i8,mask_01);
      b1=_mm256_cmpeq_epi8(i8,nb_pivots);

      A0=_mm256_loadu_si256((__m256i*)(A+i*SIZE_ROW_Av));
      A1=_mm256_loadu_si256((__m256i*)(A+i*SIZE_ROW_Av+32));
      /* store  the pivot row if b1!=0 */
      A0^=(A0^pr0)&b1;
      A1^=(A1^pr1)&b1;
      _mm256_storeu_si256((__m256i*)(A+i*SIZE_ROW_Av),A0);
      _mm256_storeu_si256((__m256i*)(A+i*SIZE_ROW_Av+32),A1);
    }
    /* add -1 if no pivot was found */
    nb_pivots=_mm256_add_epi8(nb_pivots,m_nopivot);
  }
  A+=32;
  /* j-=32; */
  for(j=0;j<31;++j)
  {
    /* generate a row with a non-zero pivot, if a non-zero pivot exists */
    /* set to 0 as soon as the pivot row is found */
    m_nopivot=~_mm256_setzero_si256();
    i8=_mm256_setzero_si256();
    pr0=_mm256_setzero_si256();
    for(i=0;i<NROWS;++i)
    {
      A0=_mm256_set1_epi8(A[i*SIZE_ROW_Av+j]);
      A0=_mm256_cmpeq_epi8(A0,_mm256_setzero_si256());
      /* i>=nb_pivots <=> i+1>nb_pivots */
      i8=_mm256_add_epi8(i8,mask_01);
      /* note that i8<=NROWS and nb_pivots<=NROWS */
      m_pivot=_mm256_cmpgt_epi8(i8,nb_pivots);

      //b1=-((i>=nb_pivots)&m_nopivot);
      b1=m_pivot&m_nopivot;
      m_nopivot=_mm256_andnot_si256(_mm256_andnot_si256(A0,m_pivot),m_nopivot);
      pr0^=_mm256_loadu_si256((__m256i*)(A+i*SIZE_ROW_Av))&b1;
    }
    /* we store nb_pivots+1 in this implementation */
    nb_pivots=_mm256_add_epi8(nb_pivots,mask_01);
    _mm256_storeu_si256((__m256i*)pivot_row,pr0);

    /* elimination of the column by the pivot */
    /* A[i]^=A[i][j]*(pivot_row*pivot_row[j]^(-1)) for each row */
    elimStep1_cstTime_gf256_avx2(A,pivot_row,j);

    /* store the pivot row in A */
    pr0=_mm256_loadu_si256((__m256i*)pivot_row);
    i8=_mm256_setzero_si256();
    /* note that necessarily, nb_pivots<=j */
    #if (62+NCOLSR_Ab)<NROWS
    for(i=0;i<=(j+32+NCOLSR_Ab);++i)
    #elif (32+NCOLSR_Ab)>=NROWS
    for(i=0;i<NROWS;++i)
    #else
    for(i=0;i<=(((j+32+NCOLSR_Ab)<NROWS)?j+32+NCOLSR_Ab:NROWS-1);++i)
    #endif
    {
      /* we store i+1 and nb_pivots+1 in this implementation */
      i8=_mm256_add_epi8(i8,mask_01);
      b1=_mm256_cmpeq_epi8(i8,nb_pivots);

      A0=_mm256_loadu_si256((__m256i*)(A+i*SIZE_ROW_Av));
      /* store  the pivot row if b1!=0 */
      A0^=(A0^pr0)&b1;
      _mm256_storeu_si256((__m256i*)(A+i*SIZE_ROW_Av),A0);
    }
    /* add -1 if no pivot was found */
    nb_pivots=_mm256_add_epi8(nb_pivots,m_nopivot);
  }
  A-=32+NCOLSR_Ab;

  /* split the augmented matrix: A|b --> A,b */
  for(i=0;i<NROWS;++i)
  {
    b[i]=A[i*SIZE_ROW_Av+NCOLS];
    A[i*SIZE_ROW_Av+NCOLS]=0;
  }

  /* only the bit sign of b0 will be used */
  b0=0;
  for(i=0;i<NCOLSR;++i)
  {
    /* check if (the row is null) and (b[i] is not null) */
    A0=_mm256_loadu_si256((__m256i*)A);
    A0|=_mm256_loadu_si256((__m256i*)A+1);
    A0|=_mm256_loadu_si256((__m256i*)A+2);
    /* note that each non-null row contains at least one time 1 */
    A0=_mm256_slli_epi16(A0,7);
    /* b0|=(-(uint64_t)(_mm256_movemask_epi8(A0)==0))&(-(uint64_t)(b[i]!=0)); */
    b0|=((uint32_t)_mm256_movemask_epi8(A0)-(uint64_t)1)&~(b[i]-(uint64_t)1);
    A+=SIZE_ROW_Av;
  }
  A+=NCOLSR;
  for(;i<(32+NCOLSR);++i)
  {
    /* check if (the row is null) and (b[i] is not null) */
    A0=_mm256_loadu_si256((__m256i*)A);
    A0|=_mm256_loadu_si256((__m256i*)A+1);
    /* note that each non-null row contains at least one time 1 */
    A0=_mm256_slli_epi16(A0,7);
    /* b0|=(-(uint64_t)(_mm256_movemask_epi8(A0)==0))&(-(uint64_t)(b[i]!=0)); */
    b0|=((uint32_t)_mm256_movemask_epi8(A0)-(uint64_t)1)&~(b[i]-(uint64_t)1);
    A+=SIZE_ROW_Av;
  }
  A+=+32;
  for(;i<NROWS;++i)
  {
    /* check if (the row is null) and (b[i] is not null) */
    A0=_mm256_loadu_si256((__m256i*)A);
    /* note that each non-null row contains at least one time 1 */
    A0=_mm256_slli_epi16(A0,7);
    /* b0|=(-(uint64_t)(_mm256_movemask_epi8(A0)==0))&(-(uint64_t)(b[i]!=0)); */
    b0|=((uint32_t)_mm256_movemask_epi8(A0)-(uint64_t)1)&~(b[i]-(uint64_t)1);
    A+=SIZE_ROW_Av;
  }
  return b0>>63;
}

#undef NROWSR
#undef NCOLSR
#undef NCOLSR_Ab

/*! \brief Perform the backward substitution on the linear system Ax=b over
 GF(256), where A is in row echelon form such that each pivot is 1.
 \param[in,out] x A vector over GF(256) of size o, initialized with the chosen
 values for the free variables. In output, the unique solution of Ax=b
 corresponding to the initial choice of free variables.
 \param[in] A A matrix over GF(256) of size m*o in row echelon form such that
 each pivot is 1.
 \param[in] b A vector over GF(256) of size m.
 \req The linear system Ax=b is consistent.
 \req A is in row echelon form such that each pivot is 1.
 \req The 1 of GF(256) corresponds to an odd byte.
 \req The 0 of GF(256) corresponds to the null byte.
 \req NCOLS>7.
 \alloc m*SIZE_ROW_Av bytes for A, m bytes for b, o bytes for x.
 \csttime A, b, x.
*/
void backSubstitution_cstTime_gf256_64(uint8_t *x, const uint8_t *A,
                                                   const uint8_t *b)
{
  uint64_t dp,ml,mr,mask_eq0;
  unsigned int i,j;

  /* If the i-th row of A is not null, b[i] xor the dot product of this row by x
     gives dp=(A[l]*x[l])^y^b[i], where A[l]=1 is the pivot of the row.
     We perform x[l]=x[l]^dp, i.e. x[l]=y^b[i], which is the unique solution of
     Ax=b for the l-th variable. For the other cases, x is not modified because
     dp&ml&mr is null: ml extracts the first non-zero byte if the latter is 1,
     and erases dp (i.e. ml=0) if the bytes are 0, whereas mr erases dp as soon
     as a non-zero byte is both found and used (mr is updated after the
     computation of dp&ml&mr). */
  for(i=NROWS-1;i!=(unsigned int)(-1);--i)
  {
    /* note that the pivot cannot be before A_ii (A is in row echelon form) */
    dp=dotProduct_gf256_pclmul(A+i*SIZE_ROW_Av+i,x+i,NCOLS-i)^b[i];
    /* duplication of dp on each byte */
    dp*=0x101010101010101;
    /* mr=0 as soon as the pivot is both found and used */
    mr=0xffffffffffffffff;
    for(j=i;j<(NCOLS-7);j+=8)
    {
      /* assumption: the first non-zero byte is 1 */
      ml=*(uint64_t*)(A+i*SIZE_ROW_Av+j);
      /* blsi: extract the lowest set bit */
      #ifdef __BMI__
        ml=_blsi_u64(ml);
      #else
        ml&=-ml;
      #endif
      /* -(uint64_t)(ml==0) for ml=0 or ml=2^8k, 0<=k<8 */
      mask_eq0=-((ml-1)>>63);
      /* 8-bit mask with this bit */
      ml=(ml<<8)-ml;
      /* (ml&mr)!=0 only for the pivot */
      *(uint64_t*)(x+j)^=dp&ml&mr;
      /* mr&=-(uint64_t)(ml==0); */
      mr&=mask_eq0;
    }
    if(j!=NCOLS)
    {
      j=NCOLS-8;
      /* assumption: the first non-zero byte is 1 */
      ml=*(uint64_t*)(A+i*SIZE_ROW_Av+j);
      /* blsi: extract the lowest set bit */
      #ifdef __BMI__
        ml=_blsi_u64(ml);
      #else
        ml&=-ml;
      #endif
      /* -(uint64_t)(ml==0) for ml=0 or ml=2^8k, 0<=k<8 */
      mask_eq0=-((ml-1)>>63);
      /* 8-bit mask with this bit */
      ml=(ml<<8)-ml;
      /* (ml&mr)!=0 only for the pivot */
      *(uint64_t*)(x+j)^=dp&ml&mr;
      /* mr&=-(uint64_t)(ml==0); */
      mr&=mask_eq0;
    }
  }
}

#undef NROWS
#undef NCOLS

