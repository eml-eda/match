checkout:
	git submodule update --recursive --init

sources:
	chmod +x ./sourceme.sh

build_tvm:
	cd match-tvm; mkdir build; cp cmake/config.cmake build; cd build; cmake ..;make -j${TVM_NCORES_INSTALL}

build_tvm_single_core:
	cd match-tvm; mkdir build; cp cmake/config.cmake build; cd build; cmake ..;make
