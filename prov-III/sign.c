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
 Implementation of the NIST API, defining crypto_sign_keypair, crypto_sign,
 crypto_sign_open.
 \author Jocelyn Ryckeghem
*/
#include "api.h"
#if SUPERCOP
  #include "crypto_sign.h"
#endif
#include "prov_sign.h"
#include <string.h>


/*! \brief Keypair generation.
 \param[out] pk The public-key corresponding to sk.
 \param[out] sk The secret-key corresponding to pk.
 \return 0 or MALLOC_FAIL.
 \alloc CRYPTO_PUBLICKEYBYTES bytes for pk, CRYPTO_SECRETKEYBYTES bytes for sk.
 \csttime sk.
 \vartime pk.
*/
int crypto_sign_keypair(unsigned char *pk, unsigned char *sk)
{
#if ESK_ON
  return prov_sign_keypair_esk(pk,sk);
#else
  return prov_sign_keypair(pk,sk);
#endif
}

/*! \brief Signature generation.
 \param[out] sm A copy of the message m and its signature obtained with the
 secret-key sk.
 \param[out] smlen The length of sm in bytes.
 \param[in] m A message to sign with the secret-key sk.
 \param[in] mlen The length of m in bytes.
 \param[in] sk A secret-key.
 \return 0 or MALLOC_FAIL.
 \alloc mlen bytes for m, CRYPTO_SECRETKEYBYTES bytes for sk,
 mlen+CRYPTO_BYTES bytes for sm.
 \csttime m, sk (except the variable number of tried salts).
*/
int crypto_sign(unsigned char *sm, unsigned long long *smlen,
                const unsigned char *m, unsigned long long mlen,
                const unsigned char *sk)
{
  *smlen=mlen+CRYPTO_BYTES;
  memcpy(sm,m,(size_t)mlen);
#if ESK_ON
  return prov_sign_esk(sm+mlen,m,mlen,sk);
#else
  return prov_sign(sm+mlen,m,mlen,sk);
#endif
}

/*! \brief Signature verification.
 \param[out] m The message stored in sm.
 \param[out] mlen The length of m in bytes.
 \param[in] sm A copy of the message m and its signature.
 \param[in] smlen The length of sm in bytes.
 \param[in] pk A public-key.
 \return MALLOC_FAIL for a malloc failure, -1 for an incorrect signature, i.e.
 the signature of the message does not correspond to pk, 0 else.
 \alloc smlen bytes for sm, CRYPTO_PUBLICKEYBYTES bytes for pk,
 smlen-CRYPTO_BYTES bytes for m.
 \csttime m, pk.
 \vartime sm.
*/
int crypto_sign_open(unsigned char *m, unsigned long long *mlen,
                     const unsigned char *sm, unsigned long long smlen,
                     const unsigned char *pk)
{
  int ret;
  *mlen=smlen-CRYPTO_BYTES;
  ret=prov_sign_open(sm,*mlen,sm+*mlen,pk);
  /* SUPERCOP requires that memcpy is performed only after sign_open */
  memcpy(m,sm,(size_t)(*mlen));
  return ret;
}

