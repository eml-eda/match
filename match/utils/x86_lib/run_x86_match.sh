#!/bin/sh
#currently src lib is empty
#cp x86_src_lib/* $1/src
cp x86_include_lib/* $1/include
cp Makefile $1/Makefile
cd $1
make all
./main | grep ]} &> x86_output.json