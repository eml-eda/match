#!/bin/sh
#currently src lib is empty
#cp $2/x86_src_lib/* $1/src
cp $2/x86_include_lib/* $1/include
cp $2/Makefile $1/Makefile
cd $1
make all
./main