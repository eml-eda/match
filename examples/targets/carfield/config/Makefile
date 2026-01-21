PROJECT_DIR ?= $(abspath ../..)
BUILD_DIR  	?= $(abspath .)
SRC_DIR  	?= $(BUILD_DIR)/src

CAR_ROOT    ?= $(ASTRAL_PATH)
BENDER      = bender -d $(CAR_ROOT)
PYTHON	    ?= python




# MATCH

MATCH_TVM_CRT_DIR 	:= $(abspath $(BUILD_DIR))/runtime
MATCH_CODEGEN_DIR 	:= $(abspath $(BUILD_DIR))/codegen
MATCH_MODEL_DIRS	:= $(wildcard $(MATCH_CODEGEN_DIR)/*)
MATCH_MODELS 		 = $(wildcard $(abspath $(BUILD_DIR))/codegen/*)
MATCH_INCLUDES		 = -I${MATCH_TVM_CRT_DIR}/include \
			 		   -I${MATCH_TVM_CRT_DIR}/src/runtime/crt/include \
			 		   $(foreach model, $(MATCH_MODEL_DIRS), -I$(model)/include) \
			 		   -I ./include/ -I $(BUILD_DIR)

# Source files

LIB_MATCH_SRCS 			= $(wildcard $(SRC_DIR)/match/*.c)
LIB_CARFIELD_LIB_SRCS 	= $(wildcard $(SRC_DIR)/carfield_lib/*.c)
LIB_PULP_NN_SRCS 		= $(wildcard $(SRC_DIR)/pulp_nn/*.c)
LIB_PULP_KERNELS_SRCS   = $(wildcard $(SRC_DIR)/pulp_kernels/*.c)
LIB_REDMULE_SRCS 		= $(wildcard $(SRC_DIR)/redmule/*.c)
LIB_SPATZ_KERNELS_SRCS 	= $(wildcard $(SRC_DIR)/spatz_kernels/*.c)

MATCH_TVM_SRCS = $(SRC_DIR)/tvm_runtime.c \
				 $(MATCH_TVM_CRT_DIR)/src/runtime/crt/common/crt_backend_api.c \
				 $(MATCH_TVM_CRT_DIR)/src/runtime/crt/memory/stack_allocator.c
MATCH_ALL_SRCS = $(foreach model, $(MATCH_MODEL_DIRS), $(wildcard $(model)/src/*.c)) $(wildcard $(SRC_DIR)/nodes/*/*.c) $(wildcard $(SRC_DIR)/*/default_inputs.c) $(wildcard $(SRC_DIR)/*/runtime.c) \
				 $(LIB_CARFIELD_LIB_SRCS) $(LIB_MATCH_SRCS)

MATCH_COMMON_SRCS = $(filter-out %.pulp_cluster.c %/main.c %/carfield.c, $(MATCH_ALL_SRCS))
MATCH_HOST_SRC = $(filter %/carfield.c, $(MATCH_ALL_SRCS)) $(SRC_DIR)/main.c $(MATCH_TVM_SRCS)
MATCH_pulp_cluster_OFFLOAD_SRC = $(filter %.pulp_cluster.c, $(MATCH_ALL_SRCS)) \
								 $(LIB_PULP_NN_SRCS) $(LIB_PULP_KERNELS_SRCS) $(LIB_REDMULE_SRCS)
MATCH_spatz_OFFLOAD_SRC = $(MATCH_COMMON_SRCS) $(LIB_SPATZ_KERNELS_SRCS)



# Compile SPATZ runtime payload

SPATZ_LLVM 	      ?= /usr/pack/riscv-1.0-kgf/spatz-llvm-2023.08.10
SPATZ_GCC		  ?= /usr/pack/riscv-1.0-kgf/spatz-gcc-7.1.1
SPATZ_CMAKE	      ?= cmake-3.18.1

SPATZD_ROOT       ?= $(shell $(BENDER) path spatz)
SPATZD_MAKEDIR    := $(SPATZD_ROOT)/hw/system/spatz_cluster
SPATZ_CLUSTER_CFG ?= spatz_cluster.carfield.l2.hjson
SPATZD_BINARY     ?=
SPATZD_BOOTMODE   ?= 0 # default jtag bootmode

SPATZ_CLUSTER_CFG_PATH    := $(SPATZD_MAKEDIR)/cfg/$(SPATZ_CLUSTER_CFG)
SPATZ_CLUSTER_CFG_DEFINES += -DMEM_DRAM_ORIGIN=$(shell python3 -c "import jstyleson; f = open('$(SPATZ_CLUSTER_CFG_PATH)'); print(jstyleson.load(f)['dram']['address'])")
SPATZ_CLUSTER_CFG_DEFINES += -DMEM_DRAM_SIZE=$(shell python3 -c "import jstyleson; f = open('$(SPATZ_CLUSTER_CFG_PATH)'); print(jstyleson.load(f)['dram']['length'])")
SPATZ_CLUSTER_CFG_DEFINES += -DSNRT_BASE_HARTID=$(shell python3 -c "import jstyleson; f = open('$(SPATZ_CLUSTER_CFG_PATH)'); print(jstyleson.load(f)['cluster']['cluster_base_hartid'])")
SPATZ_CLUSTER_CFG_DEFINES += -DSNRT_CLUSTER_CORE_NUM=$(shell python3 -c "import jstyleson; f = open('$(SPATZ_CLUSTER_CFG_PATH)'); print(len(jstyleson.load(f)['cluster']['cores']))")
SPATZ_CLUSTER_CFG_DEFINES += -DSNRT_TCDM_START_ADDR=$(shell python3 -c "import jstyleson; f = open('$(SPATZ_CLUSTER_CFG_PATH)'); print(jstyleson.load(f)['cluster']['cluster_base_addr'])")
SPATZ_CLUSTER_CFG_DEFINES += -DSNRT_CLUSTER_OFFSET=$(shell python3 -c "import jstyleson; f = open('$(SPATZ_CLUSTER_CFG_PATH)'); print(jstyleson.load(f)['cluster']['cluster_base_offset'])")
SPATZ_CLUSTER_CFG_DEFINES += -DSNRT_TCDM_SIZE=$(shell python3 -c "import jstyleson; f = open('$(SPATZ_CLUSTER_CFG_PATH)'); print(jstyleson.load(f)['cluster']['tcdm']['size'] * 1024)")
SPATZ_CLUSTER_CFG_DEFINES += -DSNRT_NFPU_PER_CORE=$(shell python3 -c "import jstyleson; f = open('$(SPATZ_CLUSTER_CFG_PATH)'); print(jstyleson.load(f)['cluster']['n_fpu'])")

space := $(empty) $(empty)

build-spatz-runtime:
	@echo "Building spatz runtime..."
	mkdir -p $(BUILD_DIR)/build-spatz
	cd $(BUILD_DIR)/build-spatz && ${SPATZ_CMAKE} -DSPATZD_ROOT=${SPATZD_ROOT} -DLLVM_PATH=${SPATZ_LLVM} -DGCC_PATH=${SPATZ_GCC} -DPYTHON=${PYTHON} -DSPATZ_CLUSTER_CFG=${SPATZ_CLUSTER_CFG} -DLINKER_SCRIPT=${CURDIR}/link_spatz.ld -DCOMMON_SRCS="$(subst $(space),;,$(MATCH_spatz_OFFLOAD_SRC))" -DCOMMON_INCS="$(subst $(space),;,$(MATCH_INCLUDES))" ${SPATZ_CLUSTER_CFG_DEFINES} .. && ${MAKE} -j8




# Compile PULPD runtime payload

PULPD_ROOT      ?= $(shell $(BENDER) path pulp_cluster)
# PULPD_RISCV		?= /usr/pack/pulpsdk-1.0-kgf/artifactory/pulp-sdk-release/pkg/pulp_riscv_gcc/1.0.16/bin/riscv32-unknown-elf
PULPD_RISCV = riscv32-unknown-elf

export PULPRT_TARGET	= carfield-cluster
export PULPRUN_TARGET	= carfield-cluster
export CONFIG_NO_FC		= 1
export ARCHI_HMR		= 1
export PULPRT_HOME 		= $(PULPD_ROOT)/pulp-runtime
export PULP_SDK_HOME    = $(PULPRT_HOME)

PULP_APPS = pulp_cluster_runtime
PULP_APP_SRCS = $(MATCH_COMMON_SRCS) $(MATCH_pulp_cluster_OFFLOAD_SRC) $(PULPRT_HOME)/lib/libc/minimal/io.c $(PULPRT_HOME)/lib/libc/minimal/prf.c 

PULP_CFLAGS = -O3 $(MATCH_INCLUDES) -DCLUSTER_COMPILATION -DARCHI_CLUSTER_NB_PE=8 -I$(PULPRT_HOME)/lib/libc/minimal/include -D__pulp_cluster__ -Dhalf=float16 -D_Float16=float16
PULPD_ELF_REMOVE_SECTIONS := --remove-section .l1cluster_g --remove-section .bss_l1

-include $(PULP_SDK_HOME)/install/rules/pulp.mk

PULP_LDFLAGS := $(filter-out -T%, $(PULP_LDFLAGS))
PULP_TARGETS = $(TARGETS)

build-pulpd-runtime: 
	@echo "Building pulp_cluster runtime..."
	$(MAKE) PULP_LDFLAGS="$(PULP_LDFLAGS) -Tlink_pulpd.ld" CONFIG_LIBC_MINIMAL=0 $(PULP_TARGETS)
	/home/fpgauser/pulp_toolchain/bin/riscv32-unknown-elf-objcopy $(PULPD_ELF_REMOVE_SECTIONS) $(BUILD_DIR)/build/pulp_cluster_runtime/pulp_cluster_runtime;

	@echo "Generating objdump..."
	/home/fpgauser/pulp_toolchain/bin/riscv32-unknown-elf-objdump -drwCS $(BUILD_DIR)/build/pulp_cluster_runtime/pulp_cluster_runtime > $(BUILD_DIR)/build/pulp_cluster_runtime/pulp_cluster_runtime.dump;

	@echo "Runtime offload build done."



# Generate runtime payload headers

pulp_cluster_RUNTIME_PAYLOAD_HEADER := $(wildcard $(BUILD_DIR)/codegen/*/include/*_pulp_cluster_runtime_payload.h)
build-pulpd-payload: build-pulpd-runtime
	@echo "Building PULPD runtime payload..."
	$(PYTHON) $(BUILD_DIR)/elf2payload.py --binary $(BUILD_DIR)/build/pulp_cluster_runtime/pulp_cluster_runtime --output $(pulp_cluster_RUNTIME_PAYLOAD_HEADER)
	@echo "PULPD Runtime payload header done."

spatz_RUNTIME_PAYLOAD_HEADER := $(wildcard $(BUILD_DIR)/codegen/*/include/*_spatz_runtime_payload.h)
build-spatz-payload: build-spatz-runtime
	@echo "Building SPATZ runtime payload..."
	$(PYTHON) $(BUILD_DIR)/elf2payload.py --binary $(BUILD_DIR)/build-spatz/main --output $(spatz_RUNTIME_PAYLOAD_HEADER)
	@echo "SPATZ runtime payload header done."



# Compile Host

CAR_SW_DIR  := $(CAR_ROOT)/sw
CHS_ROOT 	?= $(shell $(BENDER) path cheshire)

# CHS_SW_GCC_BINROOT  ?= /usr/pack/riscv-1.0-kgf/riscv64-gcc-14.2.0/bin
CHS_SW_GCC_BINROOT = /home/fpgauser/astral_toolchain/bin

-include $(CHS_ROOT)/cheshire.mk
CHS_BOOTMODE ?= 0 # default passive bootmode
CHS_PRELMODE ?= 1 # default serial link preload
CHS_BINARY   ?=
CHS_IMAGE    ?=

PULPD_ROOT      ?= $(shell $(BENDER) path pulp_cluster)
PULPD_BINARY    ?=
PULPD_TEST_NAME ?=
PULPD_BOOTMODE  ?=

LD_DIR     ?= $(BUILD_DIR)/link
HOST_LDFLAGS ?= $(CHS_SW_LDFLAGS) -L$(CAR_LD_DIR) -mcmodel=medany
#HOST_LD_SCRIPT  ?= $(CAR_SW_DIR)/link/l2.ld
HOST_LD_SCRIPT  ?= link.ld

HOST_INCLUDES    = -I$(CAR_SW_DIR)/include -I$(CHS_SW_DIR)/include $(CHS_SW_DEPS_INCS) -Wno-implicit-function-declaration -Wno-int-conversion -Wno-implicit-int -Wno-incompatible-pointer-types -Wno-unused-function -Wno-unused-variable -Wno-unused-but-set-variable -Wno-unused-parameter -Wno-maybe-uninitialized -Wno-sign-compare -Wno-missing-field-initializers
HOST_LIB_SRCS_S  = $(wildcard $(CAR_SW_DIR)/lib/*.S $(CAR_SW_DIR)/lib/**/*.S)
HOST_LIB_SRCS_C  = $(wildcard $(CAR_SW_DIR)/lib/*.c $(CAR_SW_DIR)/lib/**/*.c)
HOST_LIB_SRCS_O  = $(CAR_SW_DEPS_SRCS:.c=.car.o) $(HOST_LIB_SRCS_S:.S=.car.o) $(HOST_LIB_SRCS_C:.c=.car.o)
HOST_LIB 		 = $(CAR_SW_DIR)/lib/libcarfield.a

$(HOST_LIB): $(HOST_LIB_SRCS_O)
	echo $(CHS_ROOT)
	$(CHS_SW_AR) $(CHS_SW_ARFLAGS) -rcsv $@ $^

$(CAR_SW_DIR)/%.car.o: $(CAR_SW_DIR)/%.c
	$(CHS_SW_CC) $(CAR_SW_INCLUDES) $(CHS_SW_CCFLAGS) -c $< -o $@

HOST_FLAGS := -T$(HOST_LD_SCRIPT) -Wno-pointer-to-int-cast -DIntClustNumCores=8 -Dhalf=_Float16 -g -march=rv64gc_zifencei_zfh -mcmodel=medany




# Build Host

build-host: $(HOST_LIB) build-pulpd-payload build-spatz-payload
	@echo "Building host"
	@echo $(HOST_LIB_SRCS_O)
	$(CHS_SW_CC) $(HOST_INCLUDES) $(MATCH_INCLUDES) $(CHS_SW_LDFLAGS) $(HOST_FLAGS) -o $(BUILD_DIR)/host.elf $(HOST_LIB) $(MATCH_COMMON_SRCS) $(MATCH_HOST_SRC) $(CHS_SW_LIBS) -lm
	@echo "Generating objdump"
	@$(CHS_SW_OBJDUMP) -drwCS $(BUILD_DIR)/host.elf > $(BUILD_DIR)/host.dump
	@echo "Host build done"


build-host-seq: $(HOST_LIB) build-pulpd-payload build-spatz-payload
	@echo "Building host sequential runtime"
	@echo $(HOST_LIB_SRCS_O)
	$(CHS_SW_CC) $(HOST_INCLUDES) $(MATCH_INCLUDES) $(CHS_SW_LDFLAGS) $(HOST_FLAGS) -DMATCHA_DISABLE_ASYNC -o $(BUILD_DIR)/host_seq.elf $(HOST_LIB) $(MATCH_COMMON_SRCS) $(MATCH_HOST_SRC) $(CHS_SW_LIBS) -lm
	@echo "Generating objdump"
	@$(CHS_SW_OBJDUMP) -drwCS $(BUILD_DIR)/host_seq.elf > $(BUILD_DIR)/host_seq.dump
	@echo "Host sequential runtime build done"