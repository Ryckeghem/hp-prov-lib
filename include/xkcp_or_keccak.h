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

#ifndef _XKCP_OR_KECCAK_H
#define _XKCP_OR_KECCAK_H
/*! \file
 An interface allowing the backward compatibility with libkeccak.
 \author Jocelyn Ryckeghem
*/

/*! Set to a non-zero value for using libXKCP.a.headers, and set to zero for
 using libkeccak.a.headers */
#define LIB_XKCP 1

#if LIB_XKCP
  #include <libXKCP.a.headers/KeccakHash.h>
#else
  #include <libkeccak.a.headers/KeccakHash.h>
#endif

#endif

