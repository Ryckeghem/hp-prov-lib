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

#ifndef _AES_CTR_H
#define _AES_CTR_H
/*! \file
 AES-CTR-based seed expander.
 \author Jocelyn Ryckeghem
*/

void expandSeedAES128_CTR(unsigned char* T, unsigned int outByteLen,
                          const unsigned char* seed, unsigned char hprefix);
void expandSeedAES128_4rounds_CTR(unsigned char* T, unsigned int outByteLen,
                                  const unsigned char* seed,
                                  unsigned char hprefix);
void expandSeedAES192_CTR(unsigned char* T, unsigned int outByteLen,
                          const unsigned char* seed, unsigned char hprefix);
void expandSeedAES192_4rounds_CTR(unsigned char* T, unsigned int outByteLen,
                                  const unsigned char* seed,
                                  unsigned char hprefix);
void expandSeedAES256_CTR(unsigned char* T, unsigned int outByteLen,
                          const unsigned char* seed, unsigned char hprefix);
void expandSeedAES256_4rounds_CTR(unsigned char* T, unsigned int outByteLen,
                                  const unsigned char* seed,
                                  unsigned char hprefix);

#endif

