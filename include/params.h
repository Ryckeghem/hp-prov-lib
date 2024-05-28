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

#ifndef _PARAMS_H
#define _PARAMS_H
/*! \file
 Parameters of PROV.
 \author Jocelyn Ryckeghem
*/

#include "secLevel.h"

/*! Set to a non-zero value for an expanded secret-key. */
#define ESK_ON 0
/*! Set to a non-zero value for storing hpk in pk (at the end of pk). */
#define HPK_IN_PK 1

#if SECU_LEVEL==143
  #define SECU_REF 128
  /*! Number of equations in the public-key. */
  #define NB_EQ 49
  /*! Number of variables in the public-key. */
  #define NB_VAR 142
  /*! Difference between the number of oil variables and NB_EQ. */
  #define DELTA 8
#elif SECU_LEVEL==207
  #define SECU_REF 192
  /*! Number of equations in the public-key. */
  #define NB_EQ 74
  /*! Number of variables in the public-key. */
  #define NB_VAR 206
  /*! Difference between the number of oil variables and NB_EQ. */
  #define DELTA 8
#elif SECU_LEVEL==272
  #define SECU_REF 256
  /*! Number of equations in the public-key. */
  #define NB_EQ 100
  /*! Number of variables in the public-key. */
  #define NB_VAR 270
  /*! Difference between the number of oil variables and NB_EQ. */
  #define DELTA 8
#else
  #error "Unknown security level."
#endif

/*! Number of oil variables. */
#define NB_OIL (NB_EQ+DELTA)
/*! Number of vinegar variables. */
#define NB_VIN (NB_VAR-NB_OIL)

/*! Minimum non-negative value x to add to a for obtaining a+x=0 mod 8. */
#define PADDING8(a) ((8-((a)&7))&7)
/*! Minimum non-negative value x to add to a for obtaining a+x=0 mod 16. */
#define PADDING16(a) ((16-((a)&15))&15)
/*! Minimum non-negative value x to add to a for obtaining a+x=0 mod 32. */
#define PADDING32(a) ((32-((a)&31))&31)

/* Length in bytes. */
#define LEN_SEED_SK (SECU_REF>>3)
#define LEN_SEED_PK (SECU_REF>>3)
#define LEN_SALT ((SECU_REF>>3)+8)
#define LEN_SIGN (NB_VAR+LEN_SALT)
#define LEN_HPK (SECU_REF>>2)

#define LEN_O (NB_VIN*NB_OIL)
#define LEN_P1 ((NB_VIN*(NB_VIN+1))>>1)
#define LEN_P2 LEN_O
#define LEN_P3 ((NB_OIL*(NB_OIL+1))>>1)
#define LEN_S LEN_P2
/* size of Av in our implementation */
#define SIZE_ROW_Av (NB_OIL+1+PADDING32(NB_OIL+1))
#define LEN_AV (NB_EQ*SIZE_ROW_Av)

#if HPK_IN_PK
  #define LEN_PK (NB_EQ*LEN_P3+LEN_SEED_PK+LEN_HPK)
#else
  #define LEN_PK (NB_EQ*LEN_P3+LEN_SEED_PK)
#endif
#define LEN_SK (LEN_HPK+LEN_SEED_SK)
#define LEN_ESK (NB_EQ*LEN_S+LEN_SEED_PK+LEN_SK)


#if LEN_SEED_SK==16
  #define expandSecretSeed expandSeedAES128_CTR
#elif LEN_SEED_SK==24
  #define expandSecretSeed expandSeedAES192_CTR
#elif LEN_SEED_SK==32
  #define expandSecretSeed expandSeedAES256_CTR
#endif

#if LEN_SEED_PK==16
  #define expandPublicSeed expandSeedAES128_4rounds_CTR
#elif LEN_SEED_PK==24
  #define expandPublicSeed expandSeedAES192_4rounds_CTR
#elif LEN_SEED_PK==32
  #define expandPublicSeed expandSeedAES256_4rounds_CTR
#endif

#endif

