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
 Cryptographic operations of PROV.
 \author Jocelyn Ryckeghem
*/
#include "prov_sign.h"
#include "ret.h"
#include "hprefix.h"
#include "params.h"
#include "randombytes.h"
#include "aes_ctr.h"
#include "xkcp_or_keccak.h"
#include "dotProduct_gf256.h"
#include "vecMatProd_gf256.h"
#include "arith_matrix_gf256.h"
#include "linSystSolver_gf256.h"
#include <stdlib.h>


/*! \brief Generate the complete public-key from the public seed.
 \param[out] P1 m public matrices over GF(256) of size v*v.
 \param[out] P2 m public matrices over GF(256) of size v*o.
 \param[in] seed_pk The public seed.
 \alloc m*((v*(v+1))/2) bytes for P1, m*v*o bytes for P2.
*/
void prov_expand_pk(uint8_t *P1, uint8_t *P2, const unsigned char *seed_pk)
{
  expandPublicSeed(P1,NB_EQ*LEN_P1,seed_pk,HPREFIX_P1);
  expandPublicSeed(P2,NB_EQ*LEN_P2,seed_pk,HPREFIX_P2);
}

/*! \brief Generate P1 and O from seed_pk and seed_sk.
 \param[out] P1 m public matrices over GF(256) of size v*v.
 \param[out] O A secret matrix over GF(256) of size v*o.
 \param[in] seed_pk The public seed.
 \param[in] seed_sk The secret seed.
 \alloc m*((v*(v+1))/2) bytes for P1, v*o bytes for O.
*/
void prov_expand_P1_O(uint8_t *P1, uint8_t *O, const unsigned char *seed_pk,
                                               const unsigned char *seed_sk)
{
  expandPublicSeed(P1,NB_EQ*LEN_P1,seed_pk,HPREFIX_P1);
  expandSecretSeed(O,LEN_O,seed_sk,HPREFIX_O);
}

/*! \brief Keypair generation of PROV.
 \param[out] pk The public-key of PROV, pk=P3,seed_pk,hpk.
 \param[out] sk The secret-key of PROV, sk=SHAKE256(6||pk),seed_sk.
 \param[out] O A secret matrix over GF(256) of size v*o.
 \param[out] P1 m public matrices over GF(256) of size v*v.
 \param[out] P2 m public matrices over GF(256) of size v*o,
 corresponding to P1_i*O+P2_i.
 \param[in] mem A buffer of NB_OIL*NB_OIL+PADDING32(NB_OIL)+32*LEN_O bytes,
 for storing P3_i as a square matrix and the lookup table of O^T.
 \alloc LEN_PK for pk, LEN_SK for sk, LEN_O+PADDING32(NB_OIL) for O,
 NB_EQ*LEN_P1 for P1, NB_EQ*LEN_P2+32 for P2,
 NB_OIL*NB_OIL+PADDING32(NB_OIL)+(LEN_O<<5) for mem.
 \csttime sk, O, P2.
 \vartime P1.
*/
static void prov_sign_keypair_core(unsigned char *pk, unsigned char *sk,
                                   uint8_t *O, uint8_t *P1, uint8_t *P2,
                                   uint8_t *mem)
{
  Keccak_HashInstance hashInstance;
  uint8_t *P3_i_sqrMat,*T_Ot;
  unsigned int i;
  unsigned char prefix;

  P3_i_sqrMat=mem;
  T_Ot=P3_i_sqrMat+NB_OIL*NB_OIL+PADDING32(NB_OIL);

  sk+=LEN_HPK;
  /* random generation of the secret seed */
  randombytes(sk,LEN_SEED_SK);

  /* seed_pk <-- SHAKE256(0||seed_sk) */
  prefix=HPREFIX_SEED_PK;
  Keccak_HashInitialize_SHAKE256(&hashInstance);
  /* length of the input in bits! */
  Keccak_HashUpdate(&hashInstance,&prefix,8);
  Keccak_HashUpdate(&hashInstance,sk,LEN_SEED_SK<<3);
  Keccak_HashFinal(&hashInstance,NULL);
  pk+=NB_EQ*LEN_P3;
  /* length of the output in bits! */
  Keccak_HashSqueeze(&hashInstance,pk,LEN_SEED_PK<<3);

  /* O <-- SHAKE256(3||seed_sk) is a SECRET data */
  expandSecretSeed(O,LEN_O,sk,HPREFIX_O);

  /* P1 <-- SHAKE256(1||seed_pk) is a PUBLIC data */
  expandPublicSeed(P1,NB_EQ*LEN_P1,pk,HPREFIX_P1);
  /* P2 <-- SHAKE256(2||seed_pk) is a PUBLIC data */
  expandPublicSeed(P2,NB_EQ*LEN_P2,pk,HPREFIX_P2);
  pk-=NB_EQ*LEN_P3;

  /* computation of P3 (PUBLIC data) */
  matT_to_mulTab_gf256_avx2(T_Ot,O);
  for(i=0;i<NB_EQ;++i)
  {
    /* P2_i+=P1_i*O */
    addMatrixMul_UPP_mulTab_gf256_avx2(P2+i*LEN_P2,P1+i*LEN_P1,O);
    /* P3_i_sqrMat=Ot*(P1_i*O+P2_i) */
    matrixMul_mulTabA_gf256_avx2(P3_i_sqrMat,T_Ot,P2+i*LEN_P2);
    /* P3_i=Sym(P3_i_sqrMat) */
    sqrMatrixToUpperT_gf256_avx2(pk+i*LEN_P3,P3_i_sqrMat);
  }

  sk-=LEN_HPK;
  /* hpk=SHAKE256(6||pk) is a PUBLIC data */
  prefix=HPREFIX_HPK;
  Keccak_HashInitialize_SHAKE256(&hashInstance);
  Keccak_HashUpdate(&hashInstance,&prefix,8);
  Keccak_HashUpdate(&hashInstance,pk,(NB_EQ*LEN_P3+LEN_SEED_PK)<<3);
  Keccak_HashFinal(&hashInstance,NULL);
  Keccak_HashSqueeze(&hashInstance,sk,LEN_HPK<<3);

  #if HPK_IN_PK
    /* copy of hpk in pk */
    pk+=NB_EQ*LEN_P3+LEN_SEED_PK;
    for(i=0;i<(LEN_HPK>>3);++i)
    {
      ((uint64_t*)pk)[i]=((uint64_t*)sk)[i];
    }
    #if (LEN_HPK)&7
    for(i<<=3;i<LEN_HPK;++i)
    {
      pk[i]=sk[i];
    }
    #endif
  #endif
}

/*! \brief Keypair generation of PROV.
 \param[out] pk The public-key of PROV, pk=P3,seed_pk,hpk.
 \param[out] sk The secret-key of PROV, sk=SHAKE256(6||pk),seed_sk.
 \return 0 or MALLOC_FAIL.
 \alloc LEN_PK for pk, LEN_SK for sk.
 \csttime sk.
*/
int prov_sign_keypair(unsigned char *pk, unsigned char *sk)
{
  uint8_t *O;

  /* memory allocation: O, P1, P2, P3_i stored as a square matrix, tab(O^T) */
  O=(uint8_t*)malloc((LEN_O+NB_EQ*(LEN_P1+LEN_P2)+NB_OIL*NB_OIL
                      +PADDING32(NB_OIL)+(LEN_O<<5))*sizeof(uint8_t));
  if(!O)
  {
    return MALLOC_FAIL;
  }
  prov_sign_keypair_core(pk,sk,O,O+LEN_O,O+LEN_O+NB_EQ*LEN_P1,
                         O+LEN_O+NB_EQ*(LEN_P1+LEN_P2));
  free(O);
  return 0;
}

/*! \brief Keypair generation of PROV, with a precomputed secret.
 \param[out] pk The public-key of PROV, pk=P3,seed_pk,hpk.
 \param[out] esk The secret-key of PROV, sk=S,seed_pk,SHAKE256(6||pk),seed_sk.
 \return 0 or MALLOC_FAIL.
 \req (LEN_ESK-NB_EQ*LEN_S)>31.
 \alloc LEN_PK for pk, LEN_ESK for esk.
 \csttime esk.
*/
int prov_sign_keypair_esk(unsigned char *pk, unsigned char *esk)
{
  uint8_t *O;
  unsigned int i;

  /* memory allocation: O, P1, P3_i stored as a square matrix, tab(O^T) */
  /* P2 is allocated in the expanded secret key */
  O=(uint8_t*)malloc((LEN_O+NB_EQ*LEN_P1+NB_OIL*NB_OIL+PADDING32(NB_OIL)
                      +(LEN_O<<5))*sizeof(uint8_t));
  if(!O)
  {
    return MALLOC_FAIL;
  }
  #if (LEN_ESK-NB_EQ*LEN_S)<32
    /* we assume that S is the first data of the secret-key */
    #error "The required memory for esk+i*LEN_P2 is LEN_P2+32."
  #endif
  prov_sign_keypair_core(pk,esk+NB_EQ*LEN_S+LEN_SEED_PK,O,O+LEN_O,esk,
                         O+LEN_O+NB_EQ*LEN_P1);

  /* S=(P1+P1^T)*O+P2 is a SECRET data */
  /* prov_sign_keypair_core computed P2+=P1*O */
  for(i=0;i<NB_EQ;++i)
  {
    /* S_i=P1_i^T*O+(P1_i*O+P2_i) */
    addMatrixMul_UtPP_mulTab_gf256_avx2(esk+i*LEN_P2,O+LEN_O+i*LEN_P1,O);
  }

  free(O);

  /* copy of the public seed in sk */
  esk+=NB_EQ*LEN_S;
  pk+=NB_EQ*LEN_P3;
  for(i=0;i<(LEN_SEED_PK>>3);++i)
  {
    ((uint64_t*)esk)[i]=((uint64_t*)pk)[i];
  }
  #if (LEN_SEED_PK)&7
    for(i<<=3;i<(LEN_SEED_PK);++i)
    {
      esk[i]=pk[i];
    }
  #endif

  return 0;
}

/*! \brief Signature generation of PROV.
 \param[out] sm The signature of m.
 \param[in] m A message to sign with the secret-key sk.
 \param[in] mlen The length of m in bytes.
 \param[in] sk The secret-key of PROV, sk=SHAKE256(6||pk),seed_sk.
 \return 0 or MALLOC_FAIL.
 \req (LEN_SIGN-LEN_SALT)>=PADDING16(NB_OIL).
 \alloc LEN_SIGN for sm, mlen bytes for m, LEN_SK for sk.
 \csttime sk (except the variable number of tried salts).
*/
int prov_sign(unsigned char *sm, const unsigned char *m,
              unsigned long long mlen, const unsigned char *sk)
{
  Keccak_HashInstance hashInstance,hI;
  uint64_t b;
  unsigned char *salt;
  uint8_t *s,*v,*o,*P1,*P2,*O,*Tv,*Tv2,*Av,*Av_;
  unsigned int i,j;
  unsigned char prefix;
  uint8_t h[NB_EQ+PADDING8(NB_EQ)],P1v[NB_EQ+PADDING8(NB_EQ)],
          vec_v[NB_VIN+PADDING32(NB_VIN)],vec_v2[NB_VIN+PADDING32(NB_VIN)],
          vec_o[NB_OIL+PADDING32(NB_OIL)],v_rev[NB_VIN+PADDING16(NB_VIN)];
  unsigned char seed_pk[LEN_SEED_PK];

  /* memory allocation */
  #if LEN_AV<NB_EQ*LEN_P1
    /* P1 is the largest */
    Tv=(uint8_t*)malloc(((NB_VIN<<6)+NB_EQ*(LEN_P1+LEN_P2)+LEN_AV)
                        *sizeof(uint8_t));
  #else
    /* Av is the largest */
    Tv=(uint8_t*)malloc((NB_VIN<<6)+NB_EQ*LEN_P2+(LEN_AV<<1))*sizeof(uint8_t));
  #endif
  if(!Tv)
  {
    return MALLOC_FAIL;
  }
  Tv2=Tv+(NB_VIN<<5);
  P1=Tv2+(NB_VIN<<5);
  P2=P1+NB_EQ*LEN_P1;
  Av=P2+NB_EQ*LEN_P2;
  /* shared memory */
  Av_=P1;
  O=P2;

  s=sm;
  salt=sm+NB_VAR;

  v=s;
  o=v+NB_VIN;

  sk+=LEN_HPK;
  /* seed_pk <-- SHAKE256(0||seed_sk) */
  prefix=HPREFIX_SEED_PK;
  Keccak_HashInitialize_SHAKE256(&hashInstance);
  Keccak_HashUpdate(&hashInstance,&prefix,8);
  Keccak_HashUpdate(&hashInstance,sk,LEN_SEED_SK<<3);
  Keccak_HashFinal(&hashInstance,NULL);
  Keccak_HashSqueeze(&hashInstance,seed_pk,LEN_SEED_PK<<3);
  /* P1 <-- SHAKE256(1||seed_pk) is a PUBLIC data */
  expandPublicSeed(P1,NB_EQ*LEN_P1,seed_pk,HPREFIX_P1);
  /* P2 <-- SHAKE256(2||seed_pk) is a PUBLIC data */
  expandPublicSeed(P2,NB_EQ*LEN_P2,seed_pk,HPREFIX_P2);

  /* v,o  <-- SHAKE256(4||seed_sk||m) is a SECRET data */
  prefix=HPREFIX_v;
  Keccak_HashInitialize_SHAKE256(&hashInstance);
  Keccak_HashUpdate(&hashInstance,&prefix,8);
  Keccak_HashUpdate(&hashInstance,sk,LEN_SEED_SK<<3);
  Keccak_HashUpdate(&hashInstance,m,mlen<<3);
  Keccak_HashFinal(&hashInstance,NULL);
  Keccak_HashSqueeze(&hashInstance,s,NB_VAR<<3);

  /* precomputations about v */
  /* 7 6 5 4 3 2 1 0 --> 1 0 3 2 5 4 7 6 */
  for(i=0;i<(NB_VIN>>3);++i)
  {
    b=((uint64_t*)v)[i];
    /* 7 6 5 4 3 2 1 0 --> 3 2 1 0 7 6 5 4 */
    b=(b<<32)|(b>>32);
    /* 3 2 1 0 7 6 5 4 --> 1 0 3 2 5 4 7 6 */
    ((uint64_t*)v_rev)[i]=((b<<16)&(uint64_t)0xffff0000ffff0000)
                         |((b>>16)&(uint64_t)0xffff0000ffff);
  }
  #if NB_VIN<8
    #error "Value of NB_VIN not supported!"
  #endif
  #if (NB_VIN&31)==2
    /* special format */
    b=(*((uint64_t*)(v+NB_VIN-8)));
    /* 7 6 * * * * * * --> 0 6 0 7 0 0 0 0 */
    ((uint64_t*)v_rev)[i]=  (b&(uint64_t)0xff000000000000)
                         |((b>>24)&(uint64_t)0xff00000000);
  #elif NB_VIN&7
    /* zero padding */
    b=(*((uint64_t*)(v+NB_VIN-8)))>>((8-(NB_VIN&7))<<3);
    /* 7 6 5 4 3 2 1 0 --> 3 2 1 0 7 6 5 4 */
    b=(b<<32)|(b>>32);
    /* 3 2 1 0 7 6 5 4 --> 1 0 3 2 5 4 7 6 */
    ((uint64_t*)v_rev)[i]=((b<<16)&(uint64_t)0xffff0000ffff0000)
                         |((b>>16)&(uint64_t)0xffff0000ffff);
    #if (NB_VIN&15)<8
      /* zero padding */
      ((uint64_t*)v_rev)[i+1]=0;
    #endif
  #elif (NB_VIN&15)==8
    /* zero padding */
    ((uint64_t*)v_rev)[i]=0;
  #endif
  vec_to_mulTab_gf256_avx2(Tv,v);
  for(i=0;i<(NB_EQ-(NB_EQ&1));i+=2)
  {
    /* v^T *P2_i */
    vecMatx2Prod_mulTabVec_gf256_avx2(Av+i*SIZE_ROW_Av,Av+(i+1)*SIZE_ROW_Av,
                                      Tv,P2+i*LEN_P2,P2+(i+1)*LEN_P2);
  }
  #if NB_EQ&1
  /* v^T *P2_i */
  vecMatProd_mulTabVec_gf256_avx2(Av+i*SIZE_ROW_Av,Tv,P2+i*LEN_P2);
  #endif

  /* P2 is no longer used, O=P2 is available */
  /* O <-- SHAKE256(3||seed_sk) is a SECRET data */
  expandSecretSeed(O,LEN_O,sk,HPREFIX_O);
  sk-=LEN_HPK;
  /* P1v = -v^T *P1* v */
  for(i=0;i<(NB_EQ-(NB_EQ&1));i+=2)
  {
    /* v^T * ((P1_i+P1_i^T)*O+P2_i) */
    /* v^T *P1 */
    vecMatx2Prod_Uv_mulTabVec_gf256_avx2(vec_v,vec_v2,
                                         Tv,P1+i*LEN_P1,P1+(i+1)*LEN_P1);
    /* (v^T *P1)* v */
    P1v[i]=dotProductRevOp2_v_gf256_pclmul(vec_v,v_rev);
    P1v[i+1]=dotProductRevOp2_v_gf256_pclmul(vec_v2,v_rev);

    /* v^T *P1_i^T */
    addVecMatProd_Ut_gf256_pclmul(vec_v,v,P1+i*LEN_P1);
    addVecMatProd_Ut_gf256_pclmul(vec_v2,v,P1+(i+1)*LEN_P1);

    /* (v^T * (P1_i+P1_i^T))*O */
    /* precomputation about vec_v */
    vec_to_mulTab_gf256_avx2(Tv2,vec_v);
    vecMatProd_mulTabVec_gf256_avx2(vec_o,Tv2,O);
    for(j=0;j<((NB_OIL+7)>>3);++j)
    {
      ((uint64_t*)(Av+i*SIZE_ROW_Av))[j]^=((uint64_t*)vec_o)[j];
    }
    vec_to_mulTab_gf256_avx2(Tv2,vec_v2);
    vecMatProd_mulTabVec_gf256_avx2(vec_o,Tv2,O);
    for(j=0;j<((NB_OIL+7)>>3);++j)
    {
      ((uint64_t*)(Av+(i+1)*SIZE_ROW_Av))[j]^=((uint64_t*)vec_o)[j];
    }
  }
  #if NB_EQ&1
  /* v^T * ((P1_i+P1_i^T)*O+P2_i) */
  /* v^T *P1 */
  vecMatProd_Uv_mulTabVec_gf256_avx2(vec_v,Tv,P1+i*LEN_P1);
  /* (v^T *P1)* v */
  P1v[i]=dotProductRevOp2_v_gf256_pclmul(vec_v,v_rev);

  /* v^T *P1_i^T */
  addVecMatProd_Ut_gf256_pclmul(vec_v,v,P1+i*LEN_P1);

  /* (v^T * (P1_i+P1_i^T))*O */
  /* precomputation about vec_v */
  vec_to_mulTab_gf256_avx2(Tv2,vec_v);
  vecMatProd_mulTabVec_gf256_avx2(vec_o,Tv2,O);
  for(j=0;j<((NB_OIL+7)>>3);++j)
  {
    ((uint64_t*)(Av+i*SIZE_ROW_Av))[j]^=((uint64_t*)vec_o)[j];
  }
  #endif

  /* P1 is no longer used, Av_=P1 is available */
  prefix=HPREFIX_h;
  do
  {
    /* generation of the salt */
    Keccak_HashSqueeze(&hashInstance,salt,LEN_SALT<<3);

    /* h=SHAKE256(5||hpk||m||salt) */
    Keccak_HashInitialize_SHAKE256(&hI);
    Keccak_HashUpdate(&hI,&prefix,8);
    Keccak_HashUpdate(&hI,sk,LEN_HPK<<3);
    Keccak_HashUpdate(&hI,m,mlen<<3);
    Keccak_HashUpdate(&hI,salt,LEN_SALT<<3);
    Keccak_HashFinal(&hI,NULL);
    Keccak_HashSqueeze(&hI,h,NB_EQ<<3);

    for(i=0;i<((NB_EQ+7)>>3);++i)
    {
      ((uint64_t*)h)[i]^=((uint64_t*)P1v)[i];
    }
    /* copy of Av because the Gaussian elimination modifies this matrix */
    copyMatrixPad32_gf256_avx2(Av_,Av);

    /* solve the linear system Av*o=t */
  } while(gaussJordanElim_cstTime_gf256_avx2(Av_,h));
  backSubstitution_cstTime_gf256_64(o,Av_,h);

  #if (LEN_SIGN-LEN_SALT)<PADDING16(NB_OIL)
    /* we assume that v||o is the first data of the signature */
    #error "The required memory for sm+NB_VIN is NB_OIL+PADDING16(NB_OIL)."
  #endif

  /* v+=O*o */
  addVecMatTProd_gf256_pclmul(v,o,O);

  free(Tv);
  return 0;
}

/*! \brief Signature generation of PROV, with the knowledge of P1, P2 and O.
 \param[out] sm The signature of m.
 \param[in] m A message to sign with the secret-key sk.
 \param[in] mlen The length of m in bytes.
 \param[in] sk The secret-key of PROV, sk=SHAKE256(6||pk),seed_sk.
 \param[in] P1 m public matrices over GF(256) of size v*v.
 \param[in] P2 m public matrices over GF(256) of size v*o.
 \param[out] O A secret matrix over GF(256) of size v*o.
 \return 0 or MALLOC_FAIL.
 \req (LEN_SIGN-LEN_SALT)>=PADDING16(NB_OIL).
 \alloc LEN_SIGN for sm, mlen bytes for m, LEN_SK for sk,
 NB_EQ*LEN_P1+PADDING32(NB_VIN) for P1, NB_EQ*LEN_P2+PADDING32(NB_OIL) for P2,
 LEN_O+PADDING32(NB_OIL) bytes for O.
 \csttime sk (except the variable number of tried salts), O.
*/
int prov_sign_epk_O(unsigned char *sm, const unsigned char *m,
                    unsigned long long mlen, const unsigned char *sk,
                    const uint8_t *P1, const uint8_t *P2, const uint8_t *O)
{
  Keccak_HashInstance hashInstance,hI;
  uint64_t b;
  unsigned char *salt;
  uint8_t *s,*v,*o,*Tv,*Tv2,*Av,*Av_;
  unsigned int i,j;
  unsigned char prefix;
  uint8_t h[NB_EQ+PADDING8(NB_EQ)],P1v[NB_EQ+PADDING8(NB_EQ)],
          vec_v[NB_VIN+PADDING32(NB_VIN)],vec_v2[NB_VIN+PADDING32(NB_VIN)],
          vec_o[NB_OIL+PADDING32(NB_OIL)],v_rev[NB_VIN+PADDING16(NB_VIN)];

  /* memory allocation */
  Tv=(uint8_t*)malloc(((NB_VIN<<6)+(LEN_AV<<1))*sizeof(uint8_t));
  if(!Tv)
  {
    return MALLOC_FAIL;
  }
  Tv2=Tv+(NB_VIN<<5);
  Av=Tv2+(NB_VIN<<5);
  Av_=Av+LEN_AV;

  s=sm;
  salt=sm+NB_VAR;

  v=s;
  o=v+NB_VIN;

  sk+=LEN_HPK;
  /* v,o  <-- SHAKE256(4||seed_sk||m) is a SECRET data */
  prefix=HPREFIX_v;
  Keccak_HashInitialize_SHAKE256(&hashInstance);
  Keccak_HashUpdate(&hashInstance,&prefix,8);
  Keccak_HashUpdate(&hashInstance,sk,LEN_SEED_SK<<3);
  Keccak_HashUpdate(&hashInstance,m,mlen<<3);
  Keccak_HashFinal(&hashInstance,NULL);
  Keccak_HashSqueeze(&hashInstance,s,NB_VAR<<3);
  sk-=LEN_HPK;

  /* precomputations about v */
  /* 7 6 5 4 3 2 1 0 --> 1 0 3 2 5 4 7 6 */
  for(i=0;i<(NB_VIN>>3);++i)
  {
    b=((uint64_t*)v)[i];
    /* 7 6 5 4 3 2 1 0 --> 3 2 1 0 7 6 5 4 */
    b=(b<<32)|(b>>32);
    /* 3 2 1 0 7 6 5 4 --> 1 0 3 2 5 4 7 6 */
    ((uint64_t*)v_rev)[i]=((b<<16)&(uint64_t)0xffff0000ffff0000)
                         |((b>>16)&(uint64_t)0xffff0000ffff);
  }
  #if NB_VIN<8
    #error "Value of NB_VIN not supported!"
  #endif
  #if (NB_VIN&31)==2
    /* special format */
    b=(*((uint64_t*)(v+NB_VIN-8)));
    /* 7 6 * * * * * * --> 0 6 0 7 0 0 0 0 */
    ((uint64_t*)v_rev)[i]=  (b&(uint64_t)0xff000000000000)
                         |((b>>24)&(uint64_t)0xff00000000);
  #elif NB_VIN&7
    /* zero padding */
    b=(*((uint64_t*)(v+NB_VIN-8)))>>((8-(NB_VIN&7))<<3);
    /* 7 6 5 4 3 2 1 0 --> 3 2 1 0 7 6 5 4 */
    b=(b<<32)|(b>>32);
    /* 3 2 1 0 7 6 5 4 --> 1 0 3 2 5 4 7 6 */
    ((uint64_t*)v_rev)[i]=((b<<16)&(uint64_t)0xffff0000ffff0000)
                         |((b>>16)&(uint64_t)0xffff0000ffff);
    #if (NB_VIN&15)<8
      /* zero padding */
      ((uint64_t*)v_rev)[i+1]=0;
    #endif
  #elif (NB_VIN&15)==8
    /* zero padding */
    ((uint64_t*)v_rev)[i]=0;
  #endif
  vec_to_mulTab_gf256_avx2(Tv,v);
  for(i=0;i<(NB_EQ-(NB_EQ&1));i+=2)
  {
    /* v^T *P2_i */
    vecMatx2Prod_mulTabVec_gf256_avx2(Av+i*SIZE_ROW_Av,Av+(i+1)*SIZE_ROW_Av,
                                      Tv,P2+i*LEN_P2,P2+(i+1)*LEN_P2);
  }
  #if NB_EQ&1
  /* v^T *P2_i */
  vecMatProd_mulTabVec_gf256_avx2(Av+i*SIZE_ROW_Av,Tv,P2+i*LEN_P2);
  #endif

  /* P1v = -v^T *P1* v */
  for(i=0;i<(NB_EQ-(NB_EQ&1));i+=2)
  {
    /* v^T * ((P1_i+P1_i^T)*O+P2_i) */
    /* v^T *P1 */
    vecMatx2Prod_Uv_mulTabVec_gf256_avx2(vec_v,vec_v2,
                                         Tv,P1+i*LEN_P1,P1+(i+1)*LEN_P1);
    /* (v^T *P1)* v */
    P1v[i]=dotProductRevOp2_v_gf256_pclmul(vec_v,v_rev);
    P1v[i+1]=dotProductRevOp2_v_gf256_pclmul(vec_v2,v_rev);

    /* v^T *P1_i^T */
    addVecMatProd_Ut_gf256_pclmul(vec_v,v,P1+i*LEN_P1);
    addVecMatProd_Ut_gf256_pclmul(vec_v2,v,P1+(i+1)*LEN_P1);

    /* (v^T * (P1_i+P1_i^T))*O */
    /* precomputation about vec_v */
    vec_to_mulTab_gf256_avx2(Tv2,vec_v);
    vecMatProd_mulTabVec_gf256_avx2(vec_o,Tv2,O);
    for(j=0;j<((NB_OIL+7)>>3);++j)
    {
      ((uint64_t*)(Av+i*SIZE_ROW_Av))[j]^=((uint64_t*)vec_o)[j];
    }
    vec_to_mulTab_gf256_avx2(Tv2,vec_v2);
    vecMatProd_mulTabVec_gf256_avx2(vec_o,Tv2,O);
    for(j=0;j<((NB_OIL+7)>>3);++j)
    {
      ((uint64_t*)(Av+(i+1)*SIZE_ROW_Av))[j]^=((uint64_t*)vec_o)[j];
    }
  }
  #if NB_EQ&1
  /* v^T * ((P1_i+P1_i^T)*O+P2_i) */
  /* v^T *P1 */
  vecMatProd_Uv_mulTabVec_gf256_avx2(vec_v,Tv,P1+i*LEN_P1);
  /* (v^T *P1)* v */
  P1v[i]=dotProductRevOp2_v_gf256_pclmul(vec_v,v_rev);

  /* v^T *P1_i^T */
  addVecMatProd_Ut_gf256_pclmul(vec_v,v,P1+i*LEN_P1);

  /* (v^T * (P1_i+P1_i^T))*O */
  /* precomputation about vec_v */
  vec_to_mulTab_gf256_avx2(Tv2,vec_v);
  vecMatProd_mulTabVec_gf256_avx2(vec_o,Tv2,O);
  for(j=0;j<((NB_OIL+7)>>3);++j)
  {
    ((uint64_t*)(Av+i*SIZE_ROW_Av))[j]^=((uint64_t*)vec_o)[j];
  }
  #endif

  prefix=HPREFIX_h;
  do
  {
    /* generation of the salt */
    Keccak_HashSqueeze(&hashInstance,salt,LEN_SALT<<3);

    /* h=SHAKE256(5||hpk||m||salt) */
    Keccak_HashInitialize_SHAKE256(&hI);
    Keccak_HashUpdate(&hI,&prefix,8);
    Keccak_HashUpdate(&hI,sk,LEN_HPK<<3);
    Keccak_HashUpdate(&hI,m,mlen<<3);
    Keccak_HashUpdate(&hI,salt,LEN_SALT<<3);
    Keccak_HashFinal(&hI,NULL);
    Keccak_HashSqueeze(&hI,h,NB_EQ<<3);

    for(i=0;i<((NB_EQ+7)>>3);++i)
    {
      ((uint64_t*)h)[i]^=((uint64_t*)P1v)[i];
    }
    /* copy of Av because the Gaussian elimination modifies this matrix */
    copyMatrixPad32_gf256_avx2(Av_,Av);

    /* solve the linear system Av*o=t */
  } while(gaussJordanElim_cstTime_gf256_avx2(Av_,h));
  backSubstitution_cstTime_gf256_64(o,Av_,h);

  #if (LEN_SIGN-LEN_SALT)<PADDING16(NB_OIL)
    /* we assume that v||o is the first data of the signature */
    #error "The required memory for sm+NB_VIN is NB_OIL+PADDING16(NB_OIL)."
  #endif

  /* v+=O*o */
  addVecMatTProd_gf256_pclmul(v,o,O);

  free(Tv);
  return 0;
}

/*! \brief Signature generation of PROV.
 \param[out] sm The signature of m.
 \param[in] m A message to sign with the secret-key esk.
 \param[in] mlen The length of m in bytes.
 \param[in] esk The secret-key of PROV, esk=S,seed_pk,SHAKE256(6||pk),seed_sk.
 \return 0 or MALLOC_FAIL.
 \req (LEN_SIGN-LEN_SALT)>=PADDING16(NB_OIL).
 \req (LEN_ESK-NB_EQ*LEN_S)>=PADDING32(NB_OIL).
 \alloc LEN_SIGN for sm, mlen bytes for m, LEN_ESK for esk.
 \csttime esk (except the variable number of tried salts).
*/
int prov_sign_esk(unsigned char *sm, const unsigned char *m,
                  unsigned long long mlen, const unsigned char *esk)
{
  Keccak_HashInstance hashInstance,hI;
  uint64_t b;
  const uint8_t *S;
  unsigned char *salt;
  uint8_t *s,*v,*o,*P1,*O,*Tv,*Av,*Av_;
  unsigned int i;
  unsigned char prefix;
  uint8_t h[NB_EQ+PADDING8(NB_EQ)],P1v[NB_EQ+PADDING8(NB_EQ)],
          vec_v[NB_VIN+PADDING32(NB_VIN)],vec_v2[NB_VIN+PADDING32(NB_VIN)],
          v_rev[NB_VIN+PADDING16(NB_VIN)];

  /* memory allocation */
  #if (LEN_AV<<1)<(NB_EQ*LEN_P1+PADDING32(NB_VIN))
    /* P1 is the largest */
    Tv=(uint8_t*)malloc(((NB_VIN<<5)+NB_EQ*LEN_P1+PADDING32(NB_VIN))
                        *sizeof(uint8_t));
  #elif (LEN_O+PADDING16(NB_OIL))<(LEN_AV<<1)
    /* 2Av is the largest */
    Tv=(uint8_t*)malloc(((NB_VIN<<5)+(LEN_AV<<1))*sizeof(uint8_t));
  #else
    /* O is the largest */
    Tv=(uint8_t*)malloc(((NB_VIN<<5)+LEN_O+PADDING16(NB_OIL))*sizeof(uint8_t));
  #endif
  if(!Tv)
  {
    return MALLOC_FAIL;
  }
  /* shared memory */
  P1=Tv+(NB_VIN<<5);
  Av=P1;
  Av_=P1+LEN_AV;
  O=P1;

  s=sm;
  salt=sm+NB_VAR;

  v=s;
  o=v+NB_VIN;
  S=esk;
  esk+=NB_EQ*LEN_S;

  /* P1 <-- SHAKE256(1||seed_pk) is PUBLIC data */
  expandPublicSeed(P1,NB_EQ*LEN_P1,esk,HPREFIX_P1);

  esk+=LEN_SEED_PK+LEN_HPK;
  /* v,o  <-- SHAKE256(4||seed_sk||m) is a SECRET data */
  prefix=HPREFIX_v;
  Keccak_HashInitialize_SHAKE256(&hashInstance);
  Keccak_HashUpdate(&hashInstance,&prefix,8);
  Keccak_HashUpdate(&hashInstance,esk,LEN_SEED_SK<<3);
  Keccak_HashUpdate(&hashInstance,m,mlen<<3);
  Keccak_HashFinal(&hashInstance,NULL);
  Keccak_HashSqueeze(&hashInstance,s,NB_VAR<<3);
  esk-=LEN_HPK;

  /* precomputations about v */
  /* 7 6 5 4 3 2 1 0 --> 1 0 3 2 5 4 7 6 */
  for(i=0;i<(NB_VIN>>3);++i)
  {
    b=((uint64_t*)v)[i];
    /* 7 6 5 4 3 2 1 0 --> 3 2 1 0 7 6 5 4 */
    b=(b<<32)|(b>>32);
    /* 3 2 1 0 7 6 5 4 --> 1 0 3 2 5 4 7 6 */
    ((uint64_t*)v_rev)[i]=((b<<16)&(uint64_t)0xffff0000ffff0000)
                         |((b>>16)&(uint64_t)0xffff0000ffff);
  }
  #if NB_VIN<8
    #error "Value of NB_VIN not supported!"
  #endif
  #if (NB_VIN&31)==2
    /* special format */
    b=(*((uint64_t*)(v+NB_VIN-8)));
    /* 7 6 * * * * * * --> 0 6 0 7 0 0 0 0 */
    ((uint64_t*)v_rev)[i]=  (b&(uint64_t)0xff000000000000)
                         |((b>>24)&(uint64_t)0xff00000000);
  #elif NB_VIN&7
    /* zero padding */
    b=(*((uint64_t*)(v+NB_VIN-8)))>>((8-(NB_VIN&7))<<3);
    /* 7 6 5 4 3 2 1 0 --> 3 2 1 0 7 6 5 4 */
    b=(b<<32)|(b>>32);
    /* 3 2 1 0 7 6 5 4 --> 1 0 3 2 5 4 7 6 */
    ((uint64_t*)v_rev)[i]=((b<<16)&(uint64_t)0xffff0000ffff0000)
                         |((b>>16)&(uint64_t)0xffff0000ffff);
    #if (NB_VIN&15)<8
      /* zero padding */
      ((uint64_t*)v_rev)[i+1]=0;
    #endif
  #elif (NB_VIN&15)==8
    /* zero padding */
    ((uint64_t*)v_rev)[i]=0;
  #endif
  vec_to_mulTab_gf256_avx2(Tv,v);
  /* P1v = -v^T *P1* v */
  for(i=0;i<(NB_EQ-(NB_EQ&1));i+=2)
  {
    /* v^T *P1 */
    vecMatx2Prod_Uv_mulTabVec_gf256_avx2(vec_v,vec_v2,
                                         Tv,P1+i*LEN_P1,P1+(i+1)*LEN_P1);
    /* (v^T *P1)* v */
    P1v[i]=dotProductRevOp2_v_gf256_pclmul(vec_v,v_rev);
    P1v[i+1]=dotProductRevOp2_v_gf256_pclmul(vec_v2,v_rev);
  }
  #if NB_EQ&1
  /* v^T *P1 */
  vecMatProd_Uv_mulTabVec_gf256_avx2(vec_v,Tv,P1+i*LEN_P1);
  /* (v^T *P1)* v */
  P1v[i]=dotProductRevOp2_v_gf256_pclmul(vec_v,v_rev);
  #endif

  /* P1 is no longer used, Av=P1 is available */
  for(i=0;i<(NB_EQ-(NB_EQ&1));i+=2)
  {
    #if (LEN_ESK-NB_EQ*LEN_S)<PADDING32(NB_OIL)
      /* we assume that S is the first data of the secret-key */
      #error "The required memory for esk+i*LEN_S is LEN_S+PADDING32(NB_OIL)."
    #endif
    /* v^T * ((P1_i+P1_i^T)*O+P2_i) */
    vecMatx2Prod_mulTabVec_gf256_avx2(Av+i*SIZE_ROW_Av,Av+(i+1)*SIZE_ROW_Av,
                                      Tv,S+i*LEN_S,S+(i+1)*LEN_S);
  }
  #if NB_EQ&1
  /* v^T * ((P1_i+P1_i^T)*O+P2_i) */
  vecMatProd_mulTabVec_gf256_avx2(Av+i*SIZE_ROW_Av,Tv,S+i*LEN_S);
  #endif

  prefix=HPREFIX_h;
  do
  {
    /* generation of the salt */
    Keccak_HashSqueeze(&hashInstance,salt,LEN_SALT<<3);

    /* h=SHAKE256(5||hpk||m||salt) */
    Keccak_HashInitialize_SHAKE256(&hI);
    Keccak_HashUpdate(&hI,&prefix,8);
    Keccak_HashUpdate(&hI,esk,LEN_HPK<<3);
    Keccak_HashUpdate(&hI,m,mlen<<3);
    Keccak_HashUpdate(&hI,salt,LEN_SALT<<3);
    Keccak_HashFinal(&hI,NULL);
    Keccak_HashSqueeze(&hI,h,NB_EQ<<3);

    for(i=0;i<((NB_EQ+7)>>3);++i)
    {
      ((uint64_t*)h)[i]^=((uint64_t*)P1v)[i];
    }
    /* copy of Av because the Gaussian elimination modifies this matrix */
    copyMatrixPad32_gf256_avx2(Av_,Av);
    /* solve the linear system Av*o=t */
  } while(gaussJordanElim_cstTime_gf256_avx2(Av_,h));
  backSubstitution_cstTime_gf256_64(o,Av_,h);

  #if (LEN_SIGN-LEN_SALT)<PADDING16(NB_OIL)
    /* we assume that v||o is the first data of the signature */
    #error "The required memory for sm+NB_VIN is NB_OIL+PADDING16(NB_OIL)."
  #endif

  /* Av is no longer used, O=P1 is available */
  /* O <-- SHAKE256(3||seed_sk) is a SECRET data */
  expandSecretSeed(O,LEN_O,esk+LEN_HPK,HPREFIX_O);
  /* v+=O*o */
  addVecMatTProd_gf256_pclmul(v,o,O);

  free(Tv);
  return 0;
}

/*! \brief Signature generation of PROV, with the knowledge of P1 and O.
 \param[out] sm The signature of m.
 \param[in] m A message to sign with the secret-key esk.
 \param[in] mlen The length of m in bytes.
 \param[in] esk The secret-key of PROV, esk=S,seed_pk,SHAKE256(6||pk),seed_sk.
 \param[in] P1 m public matrices over GF(256) of size v*v.
 \param[in] O A secret matrix over GF(256) of size v*o.
 \return 0 or MALLOC_FAIL.
 \req (LEN_SIGN-LEN_SALT)>=PADDING16(NB_OIL).
 \req (LEN_ESK-NB_EQ*LEN_S)>=PADDING32(NB_OIL).
 \alloc LEN_SIGN for sm, mlen bytes for m, LEN_ESK for esk,
 NB_EQ*LEN_P1+PADDING32(NB_VIN) for P1, LEN_O+PADDING16(NB_OIL) bytes for O.
 \csttime esk (except the variable number of tried salts), O.
*/
int prov_sign_esk_P1_O(unsigned char *sm, const unsigned char *m,
                       unsigned long long mlen, const unsigned char *esk,
                       const uint8_t *P1, const uint8_t *O)
{
  Keccak_HashInstance hashInstance,hI;
  uint64_t b;
  const uint8_t *S;
  unsigned char *salt;
  uint8_t *s,*v,*o,*Tv,*Av,*Av_;
  unsigned int i;
  unsigned char prefix;
  uint8_t h[NB_EQ+PADDING8(NB_EQ)],P1v[NB_EQ+PADDING8(NB_EQ)],
          vec_v[NB_VIN+PADDING32(NB_VIN)],vec_v2[NB_VIN+PADDING32(NB_VIN)],
          v_rev[NB_VIN+PADDING16(NB_VIN)];

  /* memory allocation */
  Tv=(uint8_t*)malloc(((NB_VIN<<5)+(LEN_AV<<1))*sizeof(uint8_t));
  if(!Tv)
  {
    return MALLOC_FAIL;
  }
  Av=Tv+(NB_VIN<<5);
  Av_=Av+LEN_AV;

  s=sm;
  salt=sm+NB_VAR;

  v=s;
  o=v+NB_VIN;
  S=esk;
  esk+=NB_EQ*LEN_S+LEN_SEED_PK+LEN_HPK;

  /* v,o  <-- SHAKE256(4||seed_sk||m) is a SECRET data */
  prefix=HPREFIX_v;
  Keccak_HashInitialize_SHAKE256(&hashInstance);
  Keccak_HashUpdate(&hashInstance,&prefix,8);
  Keccak_HashUpdate(&hashInstance,esk,LEN_SEED_SK<<3);
  Keccak_HashUpdate(&hashInstance,m,mlen<<3);
  Keccak_HashFinal(&hashInstance,NULL);
  Keccak_HashSqueeze(&hashInstance,s,NB_VAR<<3);
  esk-=LEN_HPK;

  /* precomputations about v */
  /* 7 6 5 4 3 2 1 0 --> 1 0 3 2 5 4 7 6 */
  for(i=0;i<(NB_VIN>>3);++i)
  {
    b=((uint64_t*)v)[i];
    /* 7 6 5 4 3 2 1 0 --> 3 2 1 0 7 6 5 4 */
    b=(b<<32)|(b>>32);
    /* 3 2 1 0 7 6 5 4 --> 1 0 3 2 5 4 7 6 */
    ((uint64_t*)v_rev)[i]=((b<<16)&(uint64_t)0xffff0000ffff0000)
                         |((b>>16)&(uint64_t)0xffff0000ffff);
  }
  #if NB_VIN<8
    #error "Value of NB_VIN not supported!"
  #endif
  #if (NB_VIN&31)==2
    /* special format */
    b=(*((uint64_t*)(v+NB_VIN-8)));
    /* 7 6 * * * * * * --> 0 6 0 7 0 0 0 0 */
    ((uint64_t*)v_rev)[i]=  (b&(uint64_t)0xff000000000000)
                         |((b>>24)&(uint64_t)0xff00000000);
  #elif NB_VIN&7
    /* zero padding */
    b=(*((uint64_t*)(v+NB_VIN-8)))>>((8-(NB_VIN&7))<<3);
    /* 7 6 5 4 3 2 1 0 --> 3 2 1 0 7 6 5 4 */
    b=(b<<32)|(b>>32);
    /* 3 2 1 0 7 6 5 4 --> 1 0 3 2 5 4 7 6 */
    ((uint64_t*)v_rev)[i]=((b<<16)&(uint64_t)0xffff0000ffff0000)
                         |((b>>16)&(uint64_t)0xffff0000ffff);
    #if (NB_VIN&15)<8
      /* zero padding */
      ((uint64_t*)v_rev)[i+1]=0;
    #endif
  #elif (NB_VIN&15)==8
    /* zero padding */
    ((uint64_t*)v_rev)[i]=0;
  #endif
  vec_to_mulTab_gf256_avx2(Tv,v);
  /* P1v = -v^T *P1* v */
  for(i=0;i<(NB_EQ-(NB_EQ&1));i+=2)
  {
    /* v^T *P1 */
    vecMatx2Prod_Uv_mulTabVec_gf256_avx2(vec_v,vec_v2,
                                         Tv,P1+i*LEN_P1,P1+(i+1)*LEN_P1);
    /* (v^T *P1)* v */
    P1v[i]=dotProductRevOp2_v_gf256_pclmul(vec_v,v_rev);
    P1v[i+1]=dotProductRevOp2_v_gf256_pclmul(vec_v2,v_rev);
  }
  #if NB_EQ&1
  /* v^T *P1 */
  vecMatProd_Uv_mulTabVec_gf256_avx2(vec_v,Tv,P1+i*LEN_P1);
  /* (v^T *P1)* v */
  P1v[i]=dotProductRevOp2_v_gf256_pclmul(vec_v,v_rev);
  #endif

  for(i=0;i<(NB_EQ-(NB_EQ&1));i+=2)
  {
    #if (LEN_ESK-NB_EQ*LEN_S)<PADDING32(NB_OIL)
      /* we assume that S is the first data of the secret-key */
      #error "The required memory for esk+i*LEN_S is LEN_S+PADDING32(NB_OIL)."
    #endif
    /* v^T * ((P1_i+P1_i^T)*O+P2_i) */
    vecMatx2Prod_mulTabVec_gf256_avx2(Av+i*SIZE_ROW_Av,Av+(i+1)*SIZE_ROW_Av,
                                      Tv,S+i*LEN_S,S+(i+1)*LEN_S);
  }
  #if NB_EQ&1
  /* v^T * ((P1_i+P1_i^T)*O+P2_i) */
  vecMatProd_mulTabVec_gf256_avx2(Av+i*SIZE_ROW_Av,Tv,S+i*LEN_S);
  #endif

  prefix=HPREFIX_h;
  do
  {
    /* generation of the salt */
    Keccak_HashSqueeze(&hashInstance,salt,LEN_SALT<<3);

    /* h=SHAKE256(5||hpk||m||salt) */
    Keccak_HashInitialize_SHAKE256(&hI);
    Keccak_HashUpdate(&hI,&prefix,8);
    Keccak_HashUpdate(&hI,esk,LEN_HPK<<3);
    Keccak_HashUpdate(&hI,m,mlen<<3);
    Keccak_HashUpdate(&hI,salt,LEN_SALT<<3);
    Keccak_HashFinal(&hI,NULL);
    Keccak_HashSqueeze(&hI,h,NB_EQ<<3);

    for(i=0;i<((NB_EQ+7)>>3);++i)
    {
      ((uint64_t*)h)[i]^=((uint64_t*)P1v)[i];
    }
    /* copy of Av because the Gaussian elimination modifies this matrix */
    copyMatrixPad32_gf256_avx2(Av_,Av);
    /* solve the linear system Av*o=t */
  } while(gaussJordanElim_cstTime_gf256_avx2(Av_,h));
  backSubstitution_cstTime_gf256_64(o,Av_,h);

  #if (LEN_SIGN-LEN_SALT)<PADDING16(NB_OIL)
    /* we assume that v||o is the first data of the signature */
    #error "The required memory for sm+NB_VIN is NB_OIL+PADDING16(NB_OIL)."
  #endif

  /* v+=O*o */
  addVecMatTProd_gf256_pclmul(v,o,O);

  free(Tv);
  return 0;
}

/*! \brief Signature verification of PROV.
 \param[in] m A message.
 \param[in] mlen The length of m in bytes.
 \param[in] sm The signature of m.
 \param[in] hpk SHAKE256(6||pk).
 \param[in] P1 m public matrices over GF(256) of size v*v.
 \param[in] P2 m public matrices over GF(256) of size v*o.
 \param[in] P3 m public matrices over GF(256) of size o*o.
 \return -1 for an incorrect signature, 0 else.
 \req NB_VAR=6 mod 8.
 \req NB_VAR>7.
 \req NB_EQ>7.
 \alloc mlen bytes for m, LEN_SIGN for sm, LEN_HPK for hpk,
 NB_EQ*LEN_P1+PADDING32(NB_VIN) for P1, NB_EQ*LEN_P2+PADDING32(NB_OIL) for P2,
 NB_EQ*LEN_P3+PADDING32(NB_OIL).
 \vartime s, the returned value.
*/
int prov_sign_open_epk(const unsigned char *m, unsigned long long mlen,
                       const unsigned char *sm, const unsigned char *hpk,
                       const uint8_t *P1, const uint8_t *P2, const uint8_t *P3)
{
  Keccak_HashInstance hashInstance;
  uint64_t b;
  const unsigned char *salt;
  const uint8_t *s;
  unsigned int i,j;
  unsigned char prefix;
  uint8_t t[NB_EQ],vec_o[NB_OIL+PADDING32(NB_OIL)],
          vec_o2[NB_OIL+PADDING32(NB_OIL)],s_rev[NB_VAR+PADDING16(NB_VAR)];
  #if PADDING32(NB_OIL)>=PADDING16(NB_VAR)
    uint8_t vec_n[NB_VAR+PADDING32(NB_OIL)],vec_n2[NB_VAR+PADDING32(NB_OIL)];
  #else
    uint8_t vec_n[NB_VAR+PADDING16(NB_VAR)],vec_n2[NB_VAR+PADDING16(NB_VAR)];
  #endif

  s=sm;
  salt=sm+NB_VAR;

  /* 7 6 5 4 3 2 1 0 --> 1 0 3 2 5 4 7 6 */
  for(i=0;i<(NB_VAR>>3);++i)
  {
    b=((uint64_t*)s)[i];
    /* 7 6 5 4 3 2 1 0 --> 3 2 1 0 7 6 5 4 */
    b=(b<<32)|(b>>32);
    /* 3 2 1 0 7 6 5 4 --> 1 0 3 2 5 4 7 6 */
    ((uint64_t*)s_rev)[i]=((b<<16)&(uint64_t)0xffff0000ffff0000)
                         |((b>>16)&(uint64_t)0xffff0000ffff);
  }
  #if NB_VAR<8
    #error "Value of NB_VAR not supported!"
  #endif
  #if NB_VAR&7
    /* zero padding */
    b=(*((uint64_t*)(s+NB_VAR-8)))>>((8-(NB_VAR&7))<<3);
    /* 7 6 5 4 3 2 1 0 --> 3 2 1 0 7 6 5 4 */
    b=(b<<32)|(b>>32);
    /* 3 2 1 0 7 6 5 4 --> 1 0 3 2 5 4 7 6 */
    ((uint64_t*)s_rev)[i]=((b<<16)&(uint64_t)0xffff0000ffff0000)
                         |((b>>16)&(uint64_t)0xffff0000ffff);
    #if (NB_VAR&15)<8
      /* zero padding */
      ((uint64_t*)s_rev)[i+1]=0;
    #endif
  #elif (NB_VAR&15)==8
    /* zero padding */
    ((uint64_t*)s_rev)[i]=0;
  #endif

  /* evaluation of the quadratic form */
  for(i=0;i<(NB_EQ-(NB_EQ&1));i+=2)
  {
    /* v^T *P1 */
    vecMatx2Prod_Uv_mulTab_gf256_avx2(vec_n,vec_n2,
                                      s,P1+i*LEN_P1,P1+(i+1)*LEN_P1);
    /* v^T *P2 */
    vecMatx2Prod_mulTab_gf256_avx2(vec_n+NB_VIN,vec_n2+NB_VIN,
                                   s,P2+i*LEN_P2,P2+(i+1)*LEN_P2);
    /* o^T *P3 */
    vecMatx2Prod_Uo_mulTab_gf256_avx2(vec_o,vec_o2,
                                      s+NB_VIN,P3+i*LEN_P3,P3+(i+1)*LEN_P3);
    /* (v^T *P1)*v+(v^T *P2 +o^T *P3)*s */
    for(j=0;j<((NB_OIL+7)>>3);++j)
    {
      ((uint64_t*)(vec_n+NB_VIN))[j]^=((uint64_t*)vec_o)[j];
      ((uint64_t*)(vec_n2+NB_VIN))[j]^=((uint64_t*)vec_o2)[j];
    }
    t[i]=dotProductRevOp2_nbvar_gf256_pclmul(vec_n,s_rev);
    t[i+1]=dotProductRevOp2_nbvar_gf256_pclmul(vec_n2,s_rev);
  }
  #if NB_EQ&1
  /* v^T *P1 || v^T *P2 */
  vecMatProd_UvP_mulTab_gf256_avx2(vec_n,s,P1+i*LEN_P1,P2+i*LEN_P2);
  /* o^T *P3 */
  vecMatProd_Uo_mulTab_gf256_avx2(vec_o,s+NB_VIN,P3+i*LEN_P3);
  /* (v^T *P1)*v+(v^T *P2 +o^T *P3)*s */
  for(j=0;j<((NB_OIL+7)>>3);++j)
  {
    ((uint64_t*)(vec_n+NB_VIN))[j]^=((uint64_t*)vec_o)[j];
  }
  t[i]=dotProductRevOp2_nbvar_gf256_pclmul(vec_n,s_rev);
  #endif

  /* h=SHAKE256(5||hpk||m||salt) */
  prefix=HPREFIX_h;
  Keccak_HashInitialize_SHAKE256(&hashInstance);
  Keccak_HashUpdate(&hashInstance,&prefix,8);
  Keccak_HashUpdate(&hashInstance,hpk,LEN_HPK<<3);
  Keccak_HashUpdate(&hashInstance,m,mlen<<3);
  Keccak_HashUpdate(&hashInstance,salt,LEN_SALT<<3);
  Keccak_HashFinal(&hashInstance,NULL);
  Keccak_HashSqueeze(&hashInstance,vec_o,NB_EQ<<3);

  /* perform t!=h */
  b=0;
  for(i=0;i<(NB_EQ>>3);++i)
  {
    b|=((uint64_t*)t)[i]^((uint64_t*)vec_o)[i];
  }
  #if NB_EQ<8
    #error "Value of NB_EQ not supported!"
  #endif
  #if NB_EQ&7
    b|=(*(uint64_t*)(t+NB_EQ-8))^(*(uint64_t*)(vec_o+NB_EQ-8));
  #endif
  return -(b!=0);
}

/*! \brief Signature verification of PROV.
 \param[in] m A message.
 \param[in] mlen The length of m in bytes.
 \param[in] sm The signature of m.
 \param[in] pk The public-key of PROV, pk=P3,seed_pk,SHAKE256(6||pk).
 \return MALLOC_FAIL for a malloc failure, -1 for an incorrect signature,
 0 else.
 \req (LEN_PK-NB_EQ*LEN_P3)>=PADDING32(NB_OIL).
 \req NB_VAR=6 mod 8.
 \req NB_VAR>7.
 \req NB_EQ>7.
 \alloc mlen bytes for m, LEN_SIGN for sm, LEN_PK for pk.
 \vartime s, the returned value.
*/
int prov_sign_open(const unsigned char *m, unsigned long long mlen,
                   const unsigned char *sm, const unsigned char *pk)
{
  uint8_t *P1,*P2;
  int ret;

  #if HPK_IN_PK
    P1=(uint8_t*)malloc((NB_EQ*(LEN_P1+LEN_P2)+PADDING32(NB_OIL))
                        *sizeof(uint8_t));
  #elif LEN_HPK>=PADDING32(NB_OIL)
    /* LEN_HPK is the largest */
    P1=(uint8_t*)malloc((NB_EQ*(LEN_P1+LEN_P2)+LEN_HPK)*sizeof(uint8_t));
  #else
    /* PADDING32(NB_OIL) is the largest */
    P1=(uint8_t*)malloc((NB_EQ*(LEN_P1+LEN_P2)+PADDING32(NB_OIL))
                        *sizeof(uint8_t));
  #endif
  if(!P1)
  {
    return MALLOC_FAIL;
  }
  P2=P1+NB_EQ*LEN_P1;

  #if (LEN_PK-NB_EQ*LEN_P3)<PADDING32(NB_OIL)
    /* we assume that P3 is the first data of the public-key */
    #error "The required memory for P3 is NB_EQ*LEN_P3+PADDING32(NB_OIL)."
  #endif

  expandPublicSeed(P1,NB_EQ*LEN_P1,pk+NB_EQ*LEN_P3,HPREFIX_P1);
  expandPublicSeed(P2,NB_EQ*LEN_P2,pk+NB_EQ*LEN_P3,HPREFIX_P2);
  #if HPK_IN_PK
  ret=prov_sign_open_epk(m,mlen,sm,pk+NB_EQ*LEN_P3+LEN_SEED_PK,P1,P2,pk);
  #else
  Keccak_HashInstance hashInstance;
  unsigned char prefix;
  /* hpk=SHAKE256(6||pk) is a PUBLIC data */
  prefix=HPREFIX_HPK;
  Keccak_HashInitialize_SHAKE256(&hashInstance);
  Keccak_HashUpdate(&hashInstance,&prefix,8);
  Keccak_HashUpdate(&hashInstance,pk,(NB_EQ*LEN_P3+LEN_SEED_PK)<<3);
  Keccak_HashFinal(&hashInstance,NULL);
  Keccak_HashSqueeze(&hashInstance,P1+NB_EQ*(LEN_P1+LEN_P2),LEN_HPK<<3);
  ret=prov_sign_open_epk(m,mlen,sm,P1+NB_EQ*(LEN_P1+LEN_P2),P1,P2,pk);
  #endif
  free(P1);
  return ret;
}

