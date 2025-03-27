checkout:
	git submodule update --recursive --init

sources:
	chmod 777 ./sourceme.sh

install_tvm:
	cd match-tvm; mkdir build; cp cmake/config.cmake build; cd build; cmake ..;make -j4