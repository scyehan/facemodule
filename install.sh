#! /bin/sh

aclocal
libtoolize -f -c
autoconf
automake --add-missing
./configure
make
sudo make install