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

#ifndef _PROV_SIGN_H
#define _PROV_SIGN_H
/*! \file
 Cryptographic operations of PROV.
 \author Jocelyn Ryckeghem
*/

#include <stdint.h>

void prov_expand_pk(uint8_t *P1, uint8_t *P2, const unsigned char *seed_pk);
void prov_expand_P1_O(uint8_t *P1, uint8_t *O, const unsigned char *seed_pk,
                                               const unsigned char *seed_sk);
int prov_sign_keypair(unsigned char *pk, unsigned char *sk);
int prov_sign_keypair_esk(unsigned char *pk, unsigned char *esk);
int prov_sign(unsigned char *sm, const unsigned char *m,
              unsigned long long mlen, const unsigned char *sk);
int prov_sign_epk_O(unsigned char *sm, const unsigned char *m,
                    unsigned long long mlen, const unsigned char *sk,
                    const uint8_t *P1, const uint8_t *P2, const uint8_t *O);
int prov_sign_esk(unsigned char *sm, const unsigned char *m,
                  unsigned long long mlen, const unsigned char *esk);
int prov_sign_esk_P1_O(unsigned char *sm, const unsigned char *m,
                       unsigned long long mlen, const unsigned char *esk,
                       const uint8_t *P1, const uint8_t *O);
int prov_sign_open_epk(const unsigned char *m, unsigned long long mlen,
                       const unsigned char *sm, const unsigned char *hpk,
                       const uint8_t *P1, const uint8_t *P2, const uint8_t *P3);
int prov_sign_open(const unsigned char *m, unsigned long long mlen,
                   const unsigned char *sm, const unsigned char *pk);

#endif

