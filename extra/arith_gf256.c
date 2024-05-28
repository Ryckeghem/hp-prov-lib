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
 32-bit mode to perform operations in GF(256).
*/
#include "arith_gf256.h"


/*! \brief Modular inverse via the extended Euclid-Stevin algorithm.
 \details Constant-time implementation, the field polynomial is 0x11b.
 \param[in] A An element of GF(256).
 \return 0 if A=0, the inverse of A otherwise.
 \req The field polynomial is an irreducible degree-8 element of GF(2)[x].
 \csttime A.
*/
uint8_t inv_xes_cb_gf256_32(uint8_t A)
{
  uint32_t au,bq,delta,sw,mask_lc_a;
  uint8_t i;

  /* maxdeg(A)-deg(b) */
  delta=-1;
  /* (a<<1)||(u<<1), u=1, we multiply a and u by x^-delta, i.e. x */
  au=(((uint32_t)A)<<17)|2;
  /* b||q, q=0 */
  bq=((uint32_t)0x11b)<<16;

  /* maxdeg(A)+deg(b)=15 */
  for(i=0;i<15;++i)
  {
    mask_lc_a=-(au>>24);
    /* if lc(A)==1 and delta<0, we swap operands (before the elimination) */
    sw=mask_lc_a&(-(delta>>31));
    delta^=(delta^(-delta))&sw;
    --delta;

    /* elimination of the degree-8 term of a: new_a=a*b_8+b*a_8 */
    /* necessary, b_8=1, and the missing swap does not impact the result */
    au^=bq&mask_lc_a;
    /* achieve the conditional swap between a and b */
    /* note that sw!=0 implies mask_lc_a!=0 */
    bq^=au&sw;
    /* multiplication by x (alignment of leading terms) */
    au<<=1;
  }

  /* division by x^8, where 8 is max(deg(A),deg(0x11b)) */
  return (uint8_t)(bq>>8);
}

