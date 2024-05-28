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

#ifndef _RANDOMBYTES_H
#define _RANDOMBYTES_H
/*! \file
 An interface for the selection of the random bytes generator.
 Any user is allowed to set EXTERN_RAND to 0 or 1.
 \author Jocelyn Ryckeghem
*/

#define EXTERN_RAND 0

#if EXTERN_RAND
  extern int randombytes(unsigned char *x, unsigned long long xlen);
#else
  #include <openssl/rand.h>
  #define randombytes(x,xlen) RAND_bytes(x,(int)xlen)
#endif

#undef EXTERN_RAND

#endif

