# HP-PROV library
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

Welcome to the High-Performance PROV (HP-PROV) library!

LIBRARY TESTED ON LINUX ONLY

This library provides an efficient implementation of the PROV cryptosystem
submitted to the call for additional digital signature schemes for the
post-quantum cryptography standardization process, process initiated by the
National Institute of Standards and Technology (NIST).

HP-PROV is a C programming language implementation. It targets the 64-bit
processors whose AES-NI instruction set, AVX2 instruction set and PCLMULQDQ
instruction are available. You cannot run HP-PROV without these instructions.

HP-PROV is based on the HPFA library, which performs an efficient arithmetic
over finite fields. Both libraries are implemented by the author;
Jocelyn Ryckeghem; but the HPFA library is not yet available. However, certain
part of code of HPFA are used in HP-PROV.

HP-PROV is licensed under the GNU Lesser General Public License, version 3 or
later. The use of code from HPFA by the author does not impact this license.

HP-PROV has to be linked with the OpenSSL library and the eXtended Keccak Code
Package (XKCP). OpenSSL and XKCP have to be installed on your device,
at /usr/local (or modify the LOCAL variable or the XKCP_PATH variable in the
Makefile to choose another path). If you are using libkeccak instead of XKCP,
then you have to set the XKCP_LFLAG variable to -lkeccak in the Makefile,
and set the LIB_XKCP variable to 0 in the include/xkcp_or_keccak.h file.
Note that you can avoid the use of OpenSSL if you set the EXTERN_RAND variable
to 0 in the include/randombytes.h file. In this case, you have to make available
an implementation of the randombytes function.


DOCUMENTATION GENERATION.
Go in the doc directory. Then, use
$ make refman.pdf
for generating refman.pdf and html/index.html.
Both are generated thanks to doxygen.
If you only want html/index.html, then use
$ make doc
Use
$ make clean
for removing the html and latex directories generated by the previous commands.
Note that refman.pdf will not be removed.


HOW TO USE THIS PACKAGE.

HP-PROV provides a Makefile.

First, you can generate the static libraries libHP-PROV-I.a, libHP-PROV-III.a
and libHP-PROV-V.a. For example,
$ make libHP-PROV-I.a
generates libHP-PROV-I.a.

Second, you can generate executable files for benchmarking HP-PROV:
Bench-PROV-I, Bench-PROV-III and Bench-PROV-V. For example,
$ Bench-PROV-I
generates libHP-PROV-I.a then Bench-PROV-I.

Third, you can run all benchmarks just with the following command:
$ make runBench-PROV
This command generates the three static libraries and their corresponding
Bench-PROV executable.
Note that benchmarks also verify the correctness of the implementation.
In particular, 256 messages are signed then verified.

Fourth, you can delete the three static libraries with the following command:
$ make clean_lib
You can delete the other files (.o and Bench-PROV) with the following command:
$ make clean

Optionally, you can try to change the CC variable for using another compiler
than gcc. You also can try to change the OPT variable for using the other flags
of the compiler. OPT is initialized to -O2 -march=native.

