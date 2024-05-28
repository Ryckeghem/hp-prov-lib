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

#ifndef _API_H
#define _API_H
/*! \file
 Implementation of the NIST API, defining CRYPTO_SECRETKEYBYTES,
 CRYPTO_PUBLICKEYBYTES, CRYPTO_BYTES, crypto_sign_keypair, crypto_sign,
 crypto_sign_open.
 \author Jocelyn Ryckeghem
*/

#include "params.h"

/*! Set to a non-zero value for an use in SUPERCOP. */
#define SUPERCOP 0

/*! Length of the secret-key in bytes. */
#if ESK_ON
  #define CRYPTO_SECRETKEYBYTES LEN_ESK
#else
  #define CRYPTO_SECRETKEYBYTES LEN_SK
#endif
/*! Length of the public-key in bytes. */
#define CRYPTO_PUBLICKEYBYTES LEN_PK
/*! Maximum length of the signature in bytes. */
#define CRYPTO_BYTES LEN_SIGN

int crypto_sign_keypair(unsigned char *pk, unsigned char *sk);
int crypto_sign(unsigned char *sm, unsigned long long *smlen,
                const unsigned char *m, unsigned long long mlen,
                const unsigned char *sk);
int crypto_sign_open(unsigned char *m, unsigned long long *mlen,
                     const unsigned char *sm, unsigned long long smlen,
                     const unsigned char *pk);

/*! Name of the cryptosystem. */
#if SECU_LEVEL==143
  #define CRYPTO_ALGNAME "prov1"
#elif SECU_LEVEL==207
  #define CRYPTO_ALGNAME "prov3"
#elif SECU_LEVEL==272
  #define CRYPTO_ALGNAME "prov5"
#else
  #define CRYPTO_ALGNAME "prov"
#endif

#endif

