# Copyright 2024 Jocelyn Ryckeghem\
\
This file is part of the HP-PROV library, based on the HPFA library.\
\
HP-PROV is free software: you can redistribute it and/or modify it under\
the terms of the GNU Lesser General Public License as published by\
the Free Software Foundation, either version 3 of the License,\
or (at your option) any later version.\
\
HP-PROV is distributed in the hope that it will be useful, but WITHOUT ANY\
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A\
PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.\
\
You should have received a copy of the GNU Lesser General Public License\
along with HP-PROV. If not, see <https://www.gnu.org/licenses/>.

LOCAL=/usr/local

XKCP_PATH=$(LOCAL)
#XKCP_LFLAG=-lkeccak
XKCP_LFLAG=-lXKCP

HEAD=$(wildcard include/*.h)
SRC1=$(wildcard prov-I/*.c)
SRC3=$(wildcard prov-III/*.c)
SRC5=$(wildcard prov-V/*.c)
OBJS1=$(SRC1:.c=.o)
OBJS3=$(SRC3:.c=.o)
OBJS5=$(SRC5:.c=.o)

CC=gcc
OPT=-O2 -march=native
#OPT=-O2 -maes -mavx2 -mpclmul -mbmi -mtune=native
CFLAGS=$(OPT) -Wall -static -fPIC -Iinclude -I$(XKCP_PATH) -I$(LOCAL)/include
LDFLAGS=-L$(XKCP_PATH) -L$(LOCAL)/lib $(XKCP_LFLAG) -lcrypto -ldl -lpthread

COPYRIGHT="\n\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#$\
\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\n$\
\# HP-PROV Copyright 2024 Jocelyn Ryckeghem \#\n$\
\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#$\
\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\n\n$\
HP-PROV is free software: you can redistribute it and/or modify it under\n$\
the terms of the GNU Lesser General Public License as published by\n$\
the Free Software Foundation, either version 3 of the License,\n$\
or (at your option) any later version.\n\n$\
HP-PROV is distributed in the hope that it will be useful, but WITHOUT ANY\n$\
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS $\
FOR A\nPARTICULAR PURPOSE. See the GNU Lesser General Public License for $\
more details.\n\nYou should have received a copy of the GNU Lesser General $\
Public License\nalong with HP-PROV. $\
If not, see <https://www.gnu.org/licenses/>.\n"


all: runBench-PROV Bench-PROV-I Bench-PROV-III Bench-PROV-V libHP-PROV-I.a libHP-PROV-III.a libHP-PROV-V.a

runBench-PROV: Bench-PROV-I Bench-PROV-III Bench-PROV-V
	./Bench-PROV-I
	./Bench-PROV-III
	./Bench-PROV-V

Bench-PROV-I: bench.c libHP-PROV-I.a
	$(CC) $(CFLAGS) -Iinclude/prov-I -Iinclude -o $@ $< -L. -lHP-PROV-I $(LDFLAGS)

Bench-PROV-III: bench.c libHP-PROV-III.a
	$(CC) $(CFLAGS) -Iinclude/prov-III -Iinclude -o $@ $< -L. -lHP-PROV-III $(LDFLAGS)

Bench-PROV-V: bench.c libHP-PROV-V.a
	$(CC) $(CFLAGS) -Iinclude/prov-V -Iinclude -o $@ $< -L. -lHP-PROV-V $(LDFLAGS)

libHP-PROV-I.a: $(OBJS1)
	ar rcs $@ $^
	@echo $(COPYRIGHT)

libHP-PROV-III.a: $(OBJS3)
	ar rcs $@ $^
	@echo $(COPYRIGHT)

libHP-PROV-V.a: $(OBJS5)
	ar rcs $@ $^
	@echo $(COPYRIGHT)

prov-I/%.o: prov-I/%.c
	$(CC) -Iinclude/prov-I $(CFLAGS) -o $@ -c $<

prov-III/%.o: prov-III/%.c
	$(CC) -Iinclude/prov-III $(CFLAGS) -o $@ -c $<

prov-V/%.o: prov-V/%.c
	$(CC) -Iinclude/prov-V $(CFLAGS) -o $@ -c $<

clean_lib:
	rm -f libHP-PROV-I.a libHP-PROV-III.a libHP-PROV-V.a

clean:
	rm -f Bench-PROV-I Bench-PROV-III Bench-PROV-V prov-I/*.o prov-III/*.o prov-V/*.o

.PHONY: all clean

