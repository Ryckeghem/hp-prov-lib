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
 Performance measurements of PROV.
 \author Jocelyn Ryckeghem
*/
#include "params.h"
#include "api.h"
#include "prov_sign.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

/*! Length of the messages in bytes. */
#define LEN_MSG 32
#if LEN_MSG<4
  #error "Not large enough for accurate benchmarks."
#endif

#if ESK_ON
  #define LEN_SELECTED_SK LEN_ESK
#else
  #define LEN_SELECTED_SK LEN_SK
#endif

#define BENCHL(f,i,lim,c1,c2,c21,cc1,cc2,cc21,INIT_LIM) \
  lim=INIT_LIM;\
  do\
  {\
    c1=clock();\
    cc1=cpucycles();\
    for(i=0;i<lim;++i)\
    {\
      f;\
    }\
    cc2=cpucycles();\
    c2=clock();\
    c21=((double)c2-(double)c1)/(double)CLOCKS_PER_SEC;\
    lim<<=1;\
  } while(c21<(double)2);\
  if(c21<(double)4)\
  {\
    c1=clock();\
    cc1=cpucycles();\
    for(i=0;i<lim;++i)\
    {\
      f;\
    }\
    cc2=cpucycles();\
    c2=clock();\
    c21=((double)c2-(double)c1)/(double)CLOCKS_PER_SEC;\
  } else\
  {\
    lim>>=1;\
  }\
  c21/=(double)lim;\
  cc21=(cc2-cc1)/(double)lim;

#define BENCH(f,i,lim,c1,c2,c21,cc1,cc2,cc21) \
        BENCHL(f,i,lim,c1,c2,c21,cc1,cc2,cc21,256)

#define PRINT_BENCH(f,c21,cc21) \
  printf("%s:\nNumber of cycles: %.2lf\nNumber of seconds: %.9lf\n",\
         #f,cc21,c21);

inline static long long cpucycles(void)
{
  unsigned long long result;
  asm volatile(".byte 15;.byte 49;shlq $32,%%rdx;orq %%rdx,%%rax"
    : "=a" (result) ::  "%rdx");
  return result;
}


/*! \brief Benchmark the cryptographic operations of PROV from HP-PROV.
 \details This function also tests the correctness of the implementation.
 \return An integer different from 0 for the errors, 0 otherwise.
*/
int main()
{
  clock_t c1,c2;
  double c21,cc21;
  long long i,lim,cc1,cc2;
  int ret,fail;

  #if LEN_SELECTED_SK<=1024
    unsigned char sk[LEN_SELECTED_SK];
  #else
    unsigned char *sk;
    sk=(unsigned char*)malloc(LEN_SELECTED_SK*sizeof(unsigned char));
  #endif

  #if LEN_PK<=1024
    unsigned char pk[LEN_PK];
  #else
    unsigned char *pk;
    pk=(unsigned char*)malloc(LEN_PK*sizeof(unsigned char));
  #endif

  uint8_t *epk;
  epk=(uint8_t*)malloc((LEN_O+NB_EQ*(LEN_P1+LEN_P2)+PADDING32(NB_OIL))
                       *sizeof(uint8_t));
  uint8_t *const O=epk, *const P1=epk+LEN_O, *const P2=epk+LEN_O+NB_EQ*LEN_P1,
          *const hpk=sk+CRYPTO_SECRETKEYBYTES-LEN_SK;

  #if LEN_MSG<=1024
    unsigned char msg[LEN_MSG];
  #else
    unsigned char *msg;
    msg=(unsigned char*)malloc(LEN_MSG*sizeof(unsigned char));
  #endif

  #if LEN_SIGN<=1024
    unsigned char sm[LEN_SIGN];
  #else
    unsigned char *sm;
    sm=(unsigned char*)malloc(LEN_SIGN*sizeof(unsigned char));
  #endif

  unsigned char *sm256;
  sm256=(unsigned char*)malloc((LEN_SIGN<<8)*sizeof(unsigned char));

  printf("Running benchmarks for %s.\n\n",CRYPTO_ALGNAME);
  printf("Secret-key size: %u bytes.\n",CRYPTO_SECRETKEYBYTES);
  printf("Public-key size: %u bytes.\n",CRYPTO_PUBLICKEYBYTES);
  printf("Signature size: %u bytes.\n\n",CRYPTO_BYTES);
  printf("Message size: %u bytes.\n\n",LEN_MSG);

  /* initialization of the message */
  for(i=0;i<LEN_MSG;++i)
  {
    msg[i]='A';
  }

  ret=0;
  BENCH(ret|=prov_sign_keypair(pk,sk),i,lim,c1,c2,c21,cc1,cc2,cc21)
  PRINT_BENCH(prov_sign_keypair,c21,cc21)
  if(ret)
  {
    puts("prov_sign_keypair failure!");
  }
  fail=ret;

  #if ESK_ON
  ret=0;
  BENCH(ret|=prov_sign_keypair_esk(pk,sk),i,lim,c1,c2,c21,cc1,cc2,cc21)
  PRINT_BENCH(prov_sign_keypair_esk,c21,cc21)
  if(ret)
  {
    puts("prov_sign_keypair_esk failure!");
  }
  #endif
  fail|=ret;

  /* we change the document for each signature */
  ret=0;
  #if ESK_ON
  BENCH({++*(uint32_t*)msg;ret|=prov_sign_esk(sm,msg,LEN_MSG,sk);},
        i,lim,c1,c2,c21,cc1,cc2,cc21)
  PRINT_BENCH(prov_sign_esk,c21,cc21)
  #else
  BENCH({++*(uint32_t*)msg;ret|=prov_sign(sm,msg,LEN_MSG,sk);},
        i,lim,c1,c2,c21,cc1,cc2,cc21)
  PRINT_BENCH(prov_sign,c21,cc21)
  #endif
  if(ret)
  {
    puts("prov_sign failure!");
  }
  fail|=ret;

  /* we change the document for each signature */
  #if ESK_ON
  BENCH(prov_expand_P1_O(P1,O,sk+NB_EQ*LEN_S,sk+NB_EQ*LEN_S+LEN_SEED_PK
                         +LEN_HPK),i,lim,c1,c2,c21,cc1,cc2,cc21)
  PRINT_BENCH(prov_expand_P1_O,c21,cc21)
  ret=0;
  BENCH({++*(uint32_t*)msg;ret|=prov_sign_esk_P1_O(sm,msg,LEN_MSG,sk,P1,O);},
        i,lim,c1,c2,c21,cc1,cc2,cc21)
  PRINT_BENCH(prov_sign_esk_P1_O,c21,cc21)
  if(ret)
  {
    puts("prov_sign_all_expanded failure!");
  }
  fail|=ret;
  #else
  prov_expand_pk(P1,P2,pk+NB_EQ*LEN_P3);
  prov_expand_P1_O(P1,O,pk+NB_EQ*LEN_P3,sk+LEN_HPK);
  ret=0;
  BENCH({++*(uint32_t*)msg;ret|=prov_sign_epk_O(sm,msg,LEN_MSG,sk,P1,P2,O);},
        i,lim,c1,c2,c21,cc1,cc2,cc21)
  PRINT_BENCH(prov_sign_epk_O,c21,cc21)
  if(ret)
  {
    puts("prov_sign_all_expanded failure!");
  }
  fail|=ret;
  #endif

  /* we generate 256 signatures */
  prov_expand_pk(P1,P2,pk+NB_EQ*LEN_P3);
  ret=0;
  for(i=0;i<256;++i)
  {
    ++*msg;
    #if ESK_ON
    ret|=prov_sign_esk_P1_O(sm256+LEN_SIGN*i,msg,LEN_MSG,sk,P1,O);
    #else
    ret|=prov_sign_epk_O(sm256+LEN_SIGN*i,msg,LEN_MSG,sk,P1,P2,O);
    #endif
    ret|=prov_sign_open_epk(msg,LEN_MSG,sm256+LEN_SIGN*i,hpk,P1,P2,pk);
  }
  if(ret)
  {
    puts("The verification of 256 signatures fails!");
  }
  fail|=ret;

  /* we repeat the verification of 256 signatures */
  ret=0;
  BENCHL({++*msg;ret|=prov_sign_open(msg,LEN_MSG,sm256+LEN_SIGN*(i&255),pk);},
         i,lim,c1,c2,c21,cc1,cc2,cc21,256)
  PRINT_BENCH(prov_sign_open,c21,cc21)
  if(ret)
  {
    puts("prov_sign_open failure!");
  }
  fail|=ret;

  BENCH(prov_expand_pk(P1,P2,pk+NB_EQ*LEN_P3),i,lim,c1,c2,c21,cc1,cc2,cc21)
  PRINT_BENCH(prov_expand_pk,c21,cc21)
  /* we repeat the verification of 256 signatures */
  ret=0;
  BENCHL({++*msg;ret|=prov_sign_open_epk(msg,LEN_MSG,sm256+LEN_SIGN*(i&255),hpk,
        P1,P2,pk);},i,lim,c1,c2,c21,cc1,cc2,cc21,256)
  PRINT_BENCH(prov_sign_open_epk,c21,cc21)
  if(ret)
  {
    puts("prov_sign_open_epk failure!");
  }
  fail|=ret;

  if(fail)
  {
    puts("A problem of correctness of the implementation was detected!");
  }

  #if LEN_SELECTED_SK>1024
    free(sk);
  #endif

  #if LEN_PK>1024
    free(pk);
  #endif

  free(epk);

  #if LEN_MSG>1024
    free(msg);
  #endif

  #if LEN_SIGN>1024
    free(sm);
  #endif

  free(sm256);

  return fail;
}

