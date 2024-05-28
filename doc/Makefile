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

all: refman.pdf

doc:
	doxygen Doxyfile

refman.pdf: Doxyfile
	doxygen Doxyfile
	cd latex;\
	make refman.pdf;\
	mv refman.pdf ../

clean:
	rm -rf html latex

.PHONY: all clean

