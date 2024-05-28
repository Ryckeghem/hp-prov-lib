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
 AES-CTR-based seed expander.
 \author Jocelyn Ryckeghem
*/
#include "aes_ctr.h"
#include <emmintrin.h>
#include <tmmintrin.h>
#include <wmmintrin.h>

#define AES128_FIRST_ROUNDKEY(c,rc,i) \
  kg=_mm_aeskeygenassist_si128(rk,rc);\
  d0=_mm_aesenc_si128(c^rk,_mm_setzero_si128());\
  c=_mm_add_epi64(c,one);\
  d1=_mm_aesenc_si128(c^rk,_mm_setzero_si128());\
  c=_mm_add_epi64(c,one);\
  d2=_mm_aesenc_si128(c^rk,_mm_setzero_si128());\
  c=_mm_add_epi64(c,one);\
  d3=_mm_aesenc_si128(c^rk,_mm_setzero_si128());\
  c=_mm_add_epi64(c,one);\
  rk^=_mm_slli_si128(rk,8);\
  rk^=_mm_slli_si128(rk,4);\
  rk^=_mm_shuffle_epi32(kg,0xff);\
  k##i=rk;

#define AES128_ROUNDKEY(rc,i) \
  kg=_mm_aeskeygenassist_si128(rk,rc);\
  d0^=rk;\
  d1^=rk;\
  d2^=rk;\
  d3^=rk;\
  d0=_mm_aesenc_si128(d0,_mm_setzero_si128());\
  d1=_mm_aesenc_si128(d1,_mm_setzero_si128());\
  d2=_mm_aesenc_si128(d2,_mm_setzero_si128());\
  d3=_mm_aesenc_si128(d3,_mm_setzero_si128());\
  rk^=_mm_slli_si128(rk,8);\
  rk^=_mm_slli_si128(rk,4);\
  rk^=_mm_shuffle_epi32(kg,0xff);\
  k##i=rk;

#define AES128_LAST_ROUNDKEY(rc) \
  kg=_mm_aeskeygenassist_si128(rk,rc);\
  d0^=rk;\
  d1^=rk;\
  d2^=rk;\
  d3^=rk;\
  d0=_mm_aesenclast_si128(d0,_mm_setzero_si128());\
  d1=_mm_aesenclast_si128(d1,_mm_setzero_si128());\
  d2=_mm_aesenclast_si128(d2,_mm_setzero_si128());\
  d3=_mm_aesenclast_si128(d3,_mm_setzero_si128());\
  rk^=_mm_slli_si128(rk,8);\
  rk^=_mm_slli_si128(rk,4);\
  rk^=_mm_shuffle_epi32(kg,0xff);\
  _mm_storeu_si128((__m128i*)T,d0^rk);\
  _mm_storeu_si128((__m128i*)T+1,d1^rk);\
  _mm_storeu_si128((__m128i*)T+2,d2^rk);\
  _mm_storeu_si128((__m128i*)T+3,d3^rk);


#define AES192_FIRST_2ROUNDKEYS(c,rc,rc_,i,i1,i2) \
  kg=_mm_aeskeygenassist_si128(rk_,rc);\
  k##i=rk_;\
  rk^=_mm_slli_si128(rk,8);\
  rk^=_mm_slli_si128(rk,4);\
  rk_^=_mm_srli_si128(rk,12);\
  rk_^=_mm_slli_epi64(rk_,32);\
  sh=_mm_shuffle_epi32(kg,0x55);\
  rk^=sh;\
  rk_^=sh;\
  kg=_mm_aeskeygenassist_si128(rk_,rc_);\
  k##i=_mm_unpacklo_epi64(k##i,rk);\
  k##i1=_mm_alignr_epi8(rk_,rk,8);\
  rk^=_mm_slli_si128(rk,8);\
  rk^=_mm_slli_si128(rk,4);\
  rk_^=_mm_srli_si128(rk,12);\
  rk_^=_mm_slli_epi64(rk_,32);\
  sh=_mm_shuffle_epi32(kg,0x55);\
  rk^=sh;\
  rk_^=sh;\
  k##i2=rk;

#define AES192_2ROUNDKEYS(c,rc,rc_,i,i1,i2) \
  kg=_mm_aeskeygenassist_si128(rk_,rc);\
  k##i=rk_;\
  rk^=_mm_slli_si128(rk,8);\
  rk^=_mm_slli_si128(rk,4);\
  rk_^=_mm_srli_si128(rk,12);\
  rk_^=_mm_slli_epi64(rk_,32);\
  sh=_mm_shuffle_epi32(kg,0x55);\
  rk^=sh;\
  rk_^=sh;\
  kg=_mm_aeskeygenassist_si128(rk_,rc_);\
  k##i=_mm_unpacklo_epi64(k##i,rk);\
  k##i1=_mm_alignr_epi8(rk_,rk,8);\
  rk^=_mm_slli_si128(rk,8);\
  rk^=_mm_slli_si128(rk,4);\
  rk_^=_mm_srli_si128(rk,12);\
  rk_^=_mm_slli_epi64(rk_,32);\
  sh=_mm_shuffle_epi32(kg,0x55);\
  rk^=sh;\
  rk_^=sh;\
  k##i2=rk;

#define AES192_LAST_2ROUNDKEYS(c,rc,rc_,i,i1) \
  kg=_mm_aeskeygenassist_si128(rk_,rc);\
  k##i=rk_;\
  rk^=_mm_slli_si128(rk,8);\
  rk^=_mm_slli_si128(rk,4);\
  rk_^=_mm_srli_si128(rk,12);\
  rk_^=_mm_slli_epi64(rk_,32);\
  sh=_mm_shuffle_epi32(kg,0x55);\
  rk^=sh;\
  rk_^=sh;\
  kg=_mm_aeskeygenassist_si128(rk_,rc_);\
  k##i=_mm_unpacklo_epi64(k##i,rk);\
  k##i1=_mm_alignr_epi8(rk_,rk,8);\
  rk^=_mm_slli_si128(rk,8);\
  rk^=_mm_slli_si128(rk,4);\
  rk^=_mm_shuffle_epi32(kg,0x55);

#define AES192_LAST_ROUNDKEY_8BYTES(c,rc) \
  kg=_mm_aeskeygenassist_si128(rk_,rc);\
  rk^=_mm_slli_epi64(rk,32);\
  rk^=_mm_shuffle_epi32(kg,0x55);\
  rk=_mm_unpacklo_epi64(rk_,rk);


#define AES256_FIRST_2ROUNDKEYS(c,rc,i,i1) \
  kg=_mm_aeskeygenassist_si128(rk_,rc);\
  d0=_mm_aesenc_si128(c^rk,rk_);\
  c=_mm_add_epi64(c,one);\
  d1=_mm_aesenc_si128(c^rk,rk_);\
  c=_mm_add_epi64(c,one);\
  d2=_mm_aesenc_si128(c^rk,rk_);\
  c=_mm_add_epi64(c,one);\
  d3=_mm_aesenc_si128(c^rk,rk_);\
  c=_mm_add_epi64(c,one);\
  rk^=_mm_slli_si128(rk,8);\
  rk^=_mm_slli_si128(rk,4);\
  rk^=_mm_shuffle_epi32(kg,0xff);\
  k##i=rk;\
  kg=_mm_aeskeygenassist_si128(rk,rc);\
  d0=_mm_aesenc_si128(d0,rk);\
  d1=_mm_aesenc_si128(d1,rk);\
  d2=_mm_aesenc_si128(d2,rk);\
  d3=_mm_aesenc_si128(d3,rk);\
  rk_^=_mm_slli_si128(rk_,8);\
  rk_^=_mm_slli_si128(rk_,4);\
  rk_^=_mm_shuffle_epi32(kg,0xaa);\
  k##i1=rk_;

#define AES256_2ROUNDKEYS(rc,i,i1) \
  kg=_mm_aeskeygenassist_si128(rk_,rc);\
  d0=_mm_aesenc_si128(d0,rk_);\
  d1=_mm_aesenc_si128(d1,rk_);\
  d2=_mm_aesenc_si128(d2,rk_);\
  d3=_mm_aesenc_si128(d3,rk_);\
  rk^=_mm_slli_si128(rk,8);\
  rk^=_mm_slli_si128(rk,4);\
  rk^=_mm_shuffle_epi32(kg,0xff);\
  k##i=rk;\
  kg=_mm_aeskeygenassist_si128(rk,rc);\
  d0=_mm_aesenc_si128(d0,rk);\
  d1=_mm_aesenc_si128(d1,rk);\
  d2=_mm_aesenc_si128(d2,rk);\
  d3=_mm_aesenc_si128(d3,rk);\
  rk_^=_mm_slli_si128(rk_,8);\
  rk_^=_mm_slli_si128(rk_,4);\
  rk_^=_mm_shuffle_epi32(kg,0xaa);\
  k##i1=rk_;

#define AES256_LAST_ROUNDKEY(rc) \
  kg=_mm_aeskeygenassist_si128(rk_,rc);\
  d0=_mm_aesenc_si128(d0,rk_);\
  d1=_mm_aesenc_si128(d1,rk_);\
  d2=_mm_aesenc_si128(d2,rk_);\
  d3=_mm_aesenc_si128(d3,rk_);\
  rk^=_mm_slli_si128(rk,8);\
  rk^=_mm_slli_si128(rk,4);\
  rk^=_mm_shuffle_epi32(kg,0xff);\
  d0=_mm_aesenclast_si128(d0,rk);\
  d1=_mm_aesenclast_si128(d1,rk);\
  d2=_mm_aesenclast_si128(d2,rk);\
  d3=_mm_aesenclast_si128(d3,rk);\
  _mm_storeu_si128((__m128i*)T,d0);\
  _mm_storeu_si128((__m128i*)T+1,d1);\
  _mm_storeu_si128((__m128i*)T+2,d2);\
  _mm_storeu_si128((__m128i*)T+3,d3);


#define AES_ENC_x4(k) \
    d0=_mm_aesenc_si128(d0,k);\
    d1=_mm_aesenc_si128(d1,k);\
    d2=_mm_aesenc_si128(d2,k);\
    d3=_mm_aesenc_si128(d3,k);

#define AES_ENCLAST_x4(k,T) \
    d0=_mm_aesenclast_si128(d0,k);\
    d1=_mm_aesenclast_si128(d1,k);\
    d2=_mm_aesenclast_si128(d2,k);\
    d3=_mm_aesenclast_si128(d3,k);\
    _mm_storeu_si128((__m128i*)T,d0);\
    _mm_storeu_si128((__m128i*)T+1,d1);\
    _mm_storeu_si128((__m128i*)T+2,d2);\
    _mm_storeu_si128((__m128i*)T+3,d3);


/*! \brief Perform the AES128 encryption in CTR mode on a zero-byte message.
 \details The encrytion key is the seed. The counter is initialized to zero,
 then we set the least significant byte of its higher 64 bits to hprefix.
 \param[out] T The encryption of a zero-byte message of size outByteLen bytes.
 \param[in] outByteLen The length of the zero-byte message in bytes.
 \param[in] seed The encryption key of size 16 bytes.
 \param[in] hprefix A prefix for generating different data for a same seed.
 \req outByteLen>63.
 \alloc 16 bytes for seed, outByteLen bytes for T.
 \csttime seed, T.
*/
void expandSeedAES128_CTR(unsigned char* T, unsigned int outByteLen,
                          const unsigned char* seed, unsigned char hprefix)
{
  const __m128i one=_mm_set_epi64x(0,1);
  __m128i c,d0,d1,d2,d3,k0,k1,k2,k3,k4,k5,k6,k7,k8,k9,rk,kg;
  unsigned int i;

  c=_mm_srli_si128(_mm_slli_si128(_mm_set1_epi8(hprefix),15),7);

  k0=_mm_loadu_si128((__m128i*)seed);
  rk=k0;
  AES128_FIRST_ROUNDKEY(c,1,1)
  AES128_ROUNDKEY(2,2)
  AES128_ROUNDKEY(4,3)
  AES128_ROUNDKEY(8,4)
  AES128_ROUNDKEY(0x10,5)
  AES128_ROUNDKEY(0x20,6)
  AES128_ROUNDKEY(0x40,7)
  AES128_ROUNDKEY(0x80,8)
  AES128_ROUNDKEY(0x1b,9)
  AES128_LAST_ROUNDKEY(0x36)
  T+=64;

  for(i=64;i<(outByteLen-63);i+=64)
  {
    d0=_mm_aesenc_si128(c^k0,k1);
    c=_mm_add_epi64(c,one);
    d1=_mm_aesenc_si128(c^k0,k1);
    c=_mm_add_epi64(c,one);
    d2=_mm_aesenc_si128(c^k0,k1);
    c=_mm_add_epi64(c,one);
    d3=_mm_aesenc_si128(c^k0,k1);
    c=_mm_add_epi64(c,one);

    AES_ENC_x4(k2)
    AES_ENC_x4(k3)
    AES_ENC_x4(k4)
    AES_ENC_x4(k5)
    AES_ENC_x4(k6)
    AES_ENC_x4(k7)
    AES_ENC_x4(k8)
    AES_ENC_x4(k9)
    AES_ENCLAST_x4(rk,T)
    T+=64;
  }
  outByteLen&=63;
  if(outByteLen)
  {
    unsigned char tmp[64];
    d0=_mm_aesenc_si128(c^k0,k1);
    c=_mm_add_epi64(c,one);
    d1=_mm_aesenc_si128(c^k0,k1);
    c=_mm_add_epi64(c,one);
    d2=_mm_aesenc_si128(c^k0,k1);
    c=_mm_add_epi64(c,one);
    d3=_mm_aesenc_si128(c^k0,k1);

    AES_ENC_x4(k2)
    AES_ENC_x4(k3)
    AES_ENC_x4(k4)
    AES_ENC_x4(k5)
    AES_ENC_x4(k6)
    AES_ENC_x4(k7)
    AES_ENC_x4(k8)
    AES_ENC_x4(k9)
    AES_ENCLAST_x4(rk,tmp)

    for(i=0;i<outByteLen;++i)
    {
      T[i]=tmp[i];
    }
  }
}

/*! \brief Perform the AES128 encryption in CTR mode, reduced to 4 rounds,
 on a zero-byte message.
 \details The encrytion key is the seed. The counter is initialized to zero,
 then we set the least significant byte of its higher 64 bits to hprefix.
 The first 4 rounds of AES are performed, but the last round is performed
 without the mixcolumns operation.
 \param[out] T The encryption of a zero-byte message of size outByteLen bytes.
 \param[in] outByteLen The length of the zero-byte message in bytes.
 \param[in] seed The encryption key of size 16 bytes.
 \param[in] hprefix A prefix for generating different data for a same seed.
 \req outByteLen>63.
 \alloc 16 bytes for seed, outByteLen bytes for T.
 \csttime seed, T.
*/
void expandSeedAES128_4rounds_CTR(unsigned char* T, unsigned int outByteLen,
                                  const unsigned char* seed,
                                  unsigned char hprefix)
{
  const __m128i one=_mm_set_epi64x(0,1);
  __m128i c,d0,d1,d2,d3,k0,k1,k2,k3,rk,kg;
  unsigned int i;

  c=_mm_srli_si128(_mm_slli_si128(_mm_set1_epi8(hprefix),15),7);

  k0=_mm_loadu_si128((__m128i*)seed);
  rk=k0;
  AES128_FIRST_ROUNDKEY(c,1,1)
  AES128_ROUNDKEY(2,2)
  AES128_ROUNDKEY(4,3)
  AES128_LAST_ROUNDKEY(8)
  T+=64;

  for(i=64;i<(outByteLen-63);i+=64)
  {
    d0=_mm_aesenc_si128(c^k0,k1);
    c=_mm_add_epi64(c,one);
    d1=_mm_aesenc_si128(c^k0,k1);
    c=_mm_add_epi64(c,one);
    d2=_mm_aesenc_si128(c^k0,k1);
    c=_mm_add_epi64(c,one);
    d3=_mm_aesenc_si128(c^k0,k1);
    c=_mm_add_epi64(c,one);

    AES_ENC_x4(k2)
    AES_ENC_x4(k3)
    AES_ENCLAST_x4(rk,T)
    T+=64;
  }
  outByteLen&=63;
  if(outByteLen)
  {
    unsigned char tmp[64];
    d0=_mm_aesenc_si128(c^k0,k1);
    c=_mm_add_epi64(c,one);
    d1=_mm_aesenc_si128(c^k0,k1);
    c=_mm_add_epi64(c,one);
    d2=_mm_aesenc_si128(c^k0,k1);
    c=_mm_add_epi64(c,one);
    d3=_mm_aesenc_si128(c^k0,k1);

    AES_ENC_x4(k2)
    AES_ENC_x4(k3)
    AES_ENCLAST_x4(rk,tmp)

    for(i=0;i<outByteLen;++i)
    {
      T[i]=tmp[i];
    }
  }
}

/*! \brief Perform the AES192 encryption in CTR mode on a zero-byte message.
 \details The encrytion key is the seed. The counter is initialized to zero,
 then we set the least significant byte of its higher 64 bits to hprefix.
 \param[out] T The encryption of a zero-byte message of size outByteLen bytes.
 \param[in] outByteLen The length of the zero-byte message in bytes.
 \param[in] seed The encryption key of size 24 bytes.
 \param[in] hprefix A prefix for generating different data for a same seed.
 \alloc 24 bytes for seed, outByteLen bytes for T.
 \csttime seed, T.
*/
void expandSeedAES192_CTR(unsigned char* T, unsigned int outByteLen,
                          const unsigned char* seed, unsigned char hprefix)
{
  const __m128i one=_mm_set_epi64x(0,1);
  __m128i c,d0,d1,d2,d3,k0,k1,k2,k3,k4,k5,k6,k7,k8,k9,k10,k11,rk,rk_,kg,sh;
  unsigned int i;

  c=_mm_srli_si128(_mm_slli_si128(_mm_set1_epi8(hprefix),15),7);

  k0=_mm_loadu_si128((__m128i*)seed);
  rk_=_mm_srli_si128(_mm_loadu_si128((__m128i*)(seed+8)),8);
  rk=k0;
  AES192_FIRST_2ROUNDKEYS(c,1,2,1,2,3)
  AES192_2ROUNDKEYS(c,4,8,4,5,6)
  AES192_2ROUNDKEYS(c,0x10,0x20,7,8,9)
  AES192_LAST_2ROUNDKEYS(c,0x40,0x80,10,11)

  for(i=0;i<(outByteLen-(outByteLen&63));i+=64)
  {
    d0=_mm_aesenc_si128(c^k0,k1);
    c=_mm_add_epi64(c,one);
    d1=_mm_aesenc_si128(c^k0,k1);
    c=_mm_add_epi64(c,one);
    d2=_mm_aesenc_si128(c^k0,k1);
    c=_mm_add_epi64(c,one);
    d3=_mm_aesenc_si128(c^k0,k1);
    c=_mm_add_epi64(c,one);

    AES_ENC_x4(k2)
    AES_ENC_x4(k3)
    AES_ENC_x4(k4)
    AES_ENC_x4(k5)
    AES_ENC_x4(k6)
    AES_ENC_x4(k7)
    AES_ENC_x4(k8)
    AES_ENC_x4(k9)
    AES_ENC_x4(k10)
    AES_ENC_x4(k11)
    AES_ENCLAST_x4(rk,T)
    T+=64;
  }
  outByteLen&=63;
  if(outByteLen)
  {
    unsigned char tmp[64];
    d0=_mm_aesenc_si128(c^k0,k1);
    c=_mm_add_epi64(c,one);
    d1=_mm_aesenc_si128(c^k0,k1);
    c=_mm_add_epi64(c,one);
    d2=_mm_aesenc_si128(c^k0,k1);
    c=_mm_add_epi64(c,one);
    d3=_mm_aesenc_si128(c^k0,k1);

    AES_ENC_x4(k2)
    AES_ENC_x4(k3)
    AES_ENC_x4(k4)
    AES_ENC_x4(k5)
    AES_ENC_x4(k6)
    AES_ENC_x4(k7)
    AES_ENC_x4(k8)
    AES_ENC_x4(k9)
    AES_ENC_x4(k10)
    AES_ENC_x4(k11)
    AES_ENCLAST_x4(rk,tmp)

    for(i=0;i<outByteLen;++i)
    {
      T[i]=tmp[i];
    }
  }
}

/*! \brief Perform the AES192 encryption in CTR mode, reduced to 4 rounds,
 on a zero-byte message.
 \details The encrytion key is the seed. The counter is initialized to zero,
 then we set the least significant byte of its higher 64 bits to hprefix.
 The first 4 rounds of AES are performed, but the last round is performed
 without the mixcolumns operation.
 \param[out] T The encryption of a zero-byte message of size outByteLen bytes.
 \param[in] outByteLen The length of the zero-byte message in bytes.
 \param[in] seed The encryption key of size 24 bytes.
 \param[in] hprefix A prefix for generating different data for a same seed.
 \alloc 24 bytes for seed, outByteLen bytes for T.
 \csttime seed, T.
*/
void expandSeedAES192_4rounds_CTR(unsigned char* T, unsigned int outByteLen,
                                  const unsigned char* seed,
                                  unsigned char hprefix)
{
  const __m128i one=_mm_set_epi64x(0,1);
  __m128i c,d0,d1,d2,d3,k0,k1,k2,k3,rk,rk_,kg,sh;
  unsigned int i;

  c=_mm_srli_si128(_mm_slli_si128(_mm_set1_epi8(hprefix),15),7);

  k0=_mm_loadu_si128((__m128i*)seed);
  rk_=_mm_srli_si128(_mm_loadu_si128((__m128i*)(seed+8)),8);
  rk=k0;
  AES192_FIRST_2ROUNDKEYS(c,1,2,1,2,3)
  AES192_LAST_ROUNDKEY_8BYTES(c,4)

  for(i=0;i<(outByteLen-(outByteLen&63));i+=64)
  {
    d0=_mm_aesenc_si128(c^k0,k1);
    c=_mm_add_epi64(c,one);
    d1=_mm_aesenc_si128(c^k0,k1);
    c=_mm_add_epi64(c,one);
    d2=_mm_aesenc_si128(c^k0,k1);
    c=_mm_add_epi64(c,one);
    d3=_mm_aesenc_si128(c^k0,k1);
    c=_mm_add_epi64(c,one);

    AES_ENC_x4(k2)
    AES_ENC_x4(k3)
    AES_ENCLAST_x4(rk,T)
    T+=64;
  }
  outByteLen&=63;
  if(outByteLen)
  {
    unsigned char tmp[64];
    d0=_mm_aesenc_si128(c^k0,k1);
    c=_mm_add_epi64(c,one);
    d1=_mm_aesenc_si128(c^k0,k1);
    c=_mm_add_epi64(c,one);
    d2=_mm_aesenc_si128(c^k0,k1);
    c=_mm_add_epi64(c,one);
    d3=_mm_aesenc_si128(c^k0,k1);

    AES_ENC_x4(k2)
    AES_ENC_x4(k3)
    AES_ENCLAST_x4(rk,tmp)

    for(i=0;i<outByteLen;++i)
    {
      T[i]=tmp[i];
    }
  }
}

/*! \brief Perform the AES256 encryption in CTR mode on a zero-byte message.
 \details The encrytion key is the seed. The counter is initialized to zero,
 then we set the least significant byte of its higher 64 bits to hprefix.
 \param[out] T The encryption of a zero-byte message of size outByteLen bytes.
 \param[in] outByteLen The length of the zero-byte message in bytes.
 \param[in] seed The encryption key of size 32 bytes.
 \param[in] hprefix A prefix for generating different data for a same seed.
 \req outByteLen>63.
 \alloc 32 bytes for seed, outByteLen bytes for T.
 \csttime seed, T.
*/
void expandSeedAES256_CTR(unsigned char* T, unsigned int outByteLen,
                          const unsigned char* seed, unsigned char hprefix)
{
  const __m128i one=_mm_set_epi64x(0,1);
  __m128i c,d0,d1,d2,d3,k0,k1,k2,k3,k4,k5,k6,k7,k8,k9,k10,k11,k12,k13,rk,rk_,kg;
  unsigned int i;

  c=_mm_srli_si128(_mm_slli_si128(_mm_set1_epi8(hprefix),15),7);

  k0=_mm_loadu_si128((__m128i*)seed);
  k1=_mm_loadu_si128((__m128i*)seed+1);
  rk=k0;
  rk_=k1;
  AES256_FIRST_2ROUNDKEYS(c,1,2,3)
  AES256_2ROUNDKEYS(2,4,5)
  AES256_2ROUNDKEYS(4,6,7)
  AES256_2ROUNDKEYS(8,8,9)
  AES256_2ROUNDKEYS(0x10,10,11)
  AES256_2ROUNDKEYS(0x20,12,13)
  AES256_LAST_ROUNDKEY(0x40)
  T+=64;

  for(i=64;i<(outByteLen-63);i+=64)
  {
    d0=_mm_aesenc_si128(c^k0,k1);
    c=_mm_add_epi64(c,one);
    d1=_mm_aesenc_si128(c^k0,k1);
    c=_mm_add_epi64(c,one);
    d2=_mm_aesenc_si128(c^k0,k1);
    c=_mm_add_epi64(c,one);
    d3=_mm_aesenc_si128(c^k0,k1);
    c=_mm_add_epi64(c,one);

    AES_ENC_x4(k2)
    AES_ENC_x4(k3)
    AES_ENC_x4(k4)
    AES_ENC_x4(k5)
    AES_ENC_x4(k6)
    AES_ENC_x4(k7)
    AES_ENC_x4(k8)
    AES_ENC_x4(k9)
    AES_ENC_x4(k10)
    AES_ENC_x4(k11)
    AES_ENC_x4(k12)
    AES_ENC_x4(k13)
    AES_ENCLAST_x4(rk,T)
    T+=64;
  }
  outByteLen&=63;
  if(outByteLen)
  {
    unsigned char tmp[64];
    d0=_mm_aesenc_si128(c^k0,k1);
    c=_mm_add_epi64(c,one);
    d1=_mm_aesenc_si128(c^k0,k1);
    c=_mm_add_epi64(c,one);
    d2=_mm_aesenc_si128(c^k0,k1);
    c=_mm_add_epi64(c,one);
    d3=_mm_aesenc_si128(c^k0,k1);

    AES_ENC_x4(k2)
    AES_ENC_x4(k3)
    AES_ENC_x4(k4)
    AES_ENC_x4(k5)
    AES_ENC_x4(k6)
    AES_ENC_x4(k7)
    AES_ENC_x4(k8)
    AES_ENC_x4(k9)
    AES_ENC_x4(k10)
    AES_ENC_x4(k11)
    AES_ENC_x4(k12)
    AES_ENC_x4(k13)
    AES_ENCLAST_x4(rk,tmp)

    for(i=0;i<outByteLen;++i)
    {
      T[i]=tmp[i];
    }
  }
}

/*! \brief Perform the AES256 encryption in CTR mode, reduced to 4 rounds,
 on a zero-byte message.
 \details The encrytion key is the seed. The counter is initialized to zero,
 then we set the least significant byte of its higher 64 bits to hprefix.
 The first 4 rounds of AES are performed, but the last round is performed
 without the mixcolumns operation.
 \param[out] T The encryption of a zero-byte message of size outByteLen bytes.
 \param[in] outByteLen The length of the zero-byte message in bytes.
 \param[in] seed The encryption key of size 32 bytes.
 \param[in] hprefix A prefix for generating different data for a same seed.
 \req outByteLen>63.
 \alloc 32 bytes for seed, outByteLen bytes for T.
 \csttime seed, T.
*/
void expandSeedAES256_4rounds_CTR(unsigned char* T, unsigned int outByteLen,
                                  const unsigned char* seed,
                                  unsigned char hprefix)
{
  const __m128i one=_mm_set_epi64x(0,1);
  __m128i c,d0,d1,d2,d3,k0,k1,k2,k3,rk,rk_,kg;
  unsigned int i;

  c=_mm_srli_si128(_mm_slli_si128(_mm_set1_epi8(hprefix),15),7);

  k0=_mm_loadu_si128((__m128i*)seed);
  k1=_mm_loadu_si128((__m128i*)seed+1);
  rk=k0;
  rk_=k1;
  AES256_FIRST_2ROUNDKEYS(c,1,2,3)
  AES256_LAST_ROUNDKEY(2)
  T+=64;

  for(i=64;i<(outByteLen-63);i+=64)
  {
    d0=_mm_aesenc_si128(c^k0,k1);
    c=_mm_add_epi64(c,one);
    d1=_mm_aesenc_si128(c^k0,k1);
    c=_mm_add_epi64(c,one);
    d2=_mm_aesenc_si128(c^k0,k1);
    c=_mm_add_epi64(c,one);
    d3=_mm_aesenc_si128(c^k0,k1);
    c=_mm_add_epi64(c,one);

    AES_ENC_x4(k2)
    AES_ENC_x4(k3)
    AES_ENCLAST_x4(rk,T)
    T+=64;
  }
  outByteLen&=63;
  if(outByteLen)
  {
    unsigned char tmp[64];
    d0=_mm_aesenc_si128(c^k0,k1);
    c=_mm_add_epi64(c,one);
    d1=_mm_aesenc_si128(c^k0,k1);
    c=_mm_add_epi64(c,one);
    d2=_mm_aesenc_si128(c^k0,k1);
    c=_mm_add_epi64(c,one);
    d3=_mm_aesenc_si128(c^k0,k1);

    AES_ENC_x4(k2)
    AES_ENC_x4(k3)
    AES_ENCLAST_x4(rk,tmp)

    for(i=0;i<outByteLen;++i)
    {
      T[i]=tmp[i];
    }
  }
}

