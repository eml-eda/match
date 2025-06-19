PROJECT_DIR ?= $(abspath ../..)
BUILD_DIR  	?= $(abspath .)
SRC_DIR  	?= $(BUILD_DIR)/src

CAR_ROOT    ?= $(PROJECT_DIR)/carfield
BENDER      ?= bender -d $(CAR_ROOT)
PYTHON	    ?= python





# MATCH

MATCH_TVM_CRT_DIR 	:= $(abspath $(BUILD_DIR))/runtime
MATCH_CODEGEN_DIR 	:= $(abspath $(BUILD_DIR))/codegen
MATCH_MODEL_DIRS	:= $(wildcard $(MATCH_CODEGEN_DIR)/*)
MATCH_MODELS 		 = $(wildcard $(abspath $(BUILD_DIR))/codegen/*)
MATCH_INCLUDES		 = -I${MATCH_TVM_CRT_DIR}/include \
			 		   -I${MATCH_TVM_CRT_DIR}/src/runtime/crt/include \
			 		   $(foreach model, $(MATCH_MODEL_DIRS), -I$(model)/include) \
			 		   -I ./include/

# Source files
MATCH_ALL_SRCS =  $(foreach model, $(MATCH_MODEL_DIRS), $(wildcard $(model)/src/*.c)) $(shell find $(abspath $(SRC_DIR)) -name '*.c')  $(MATCH_TVM_CRT_DIR)/src/runtime/crt/common/crt_backend_api.c $(MATCH_TVM_CRT_DIR)/src/runtime/crt/memory/stack_allocator.c
MATCH_COMMON_SRCS = $(filter-out %.pulp_cluster.c %/main.c %/carfield.c, $(MATCH_ALL_SRCS))

MATCH_HOST_SRC = $(filter %/main.c %/carfield.c, $(MATCH_ALL_SRCS))
MATCH_pulp_cluster_OFFLOAD_SRC = $(filter %.pulp_cluster.c, $(MATCH_ALL_SRCS))





# Compile runtime payload

PULPD_ROOT      ?= $(shell $(BENDER) path pulp_cluster)
PULPD_RISCV		?= /usr/pack/pulpsdk-1.0-kgf/artifactory/pulp-sdk-release/pkg/pulp_riscv_gcc/1.0.16/bin/riscv32-unknown-elf

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

build-offload: 
	@echo "Building pulp_cluster runtime..."
	$(MAKE) PULP_LDFLAGS="$(PULP_LDFLAGS) -Tlink_pulpd.ld" CONFIG_LIBC_MINIMAL=0 $(PULP_TARGETS)
	$(PULPD_RISCV)-objcopy $(PULPD_ELF_REMOVE_SECTIONS) $(BUILD_DIR)/build/pulp_cluster_runtime/pulp_cluster_runtime;

	@echo "Generating objdump..."
	$(PULPD_RISCV)-objdump -drwCS $(BUILD_DIR)/build/pulp_cluster_runtime/pulp_cluster_runtime > $(BUILD_DIR)/build/pulp_cluster_runtime/pulp_cluster_runtime.dump;

	@echo "Runtime offload build done."





# Compile Host

CAR_SW_DIR  := $(CAR_ROOT)/sw
CHS_ROOT 	?= $(shell $(BENDER) path cheshire)

CHS_SW_GCC_BINROOT  ?= /usr/pack/riscv-1.0-kgf/riscv64-gcc-14.2.0/bin

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
HOST_LDFLAGS ?= $(CHS_SW_LDFLAGS) -L$(CAR_LD_DIR)
#HOST_LD_SCRIPT  ?= $(CAR_SW_DIR)/link/l2.ld
HOST_LD_SCRIPT  ?= link.ld

HOST_INCLUDES    = -I$(CAR_SW_DIR)/include -I$(CHS_SW_DIR)/include $(CHS_SW_DEPS_INCS)
HOST_LIB_SRCS_S  = $(wildcard $(CAR_SW_DIR)/lib/*.S $(CAR_SW_DIR)/lib/**/*.S)
HOST_LIB_SRCS_C  = $(wildcard $(CAR_SW_DIR)/lib/*.c $(CAR_SW_DIR)/lib/**/*.c)
HOST_LIB_SRCS_O  = $(CAR_SW_DEPS_SRCS:.c=.car.o) $(HOST_LIB_SRCS_S:.S=.car.o) $(HOST_LIB_SRCS_C:.c=.car.o)
HOST_LIB 		 = $(CAR_SW_DIR)/lib/libcarfield.a

$(HOST_LIB): $(HOST_LIB_SRCS_O)
	echo $(CHS_ROOT)
	$(CHS_SW_AR) $(CHS_SW_ARFLAGS) -rcsv $@ $^

$(CAR_SW_DIR)/%.car.o: $(CAR_SW_DIR)/%.c
	$(CHS_SW_CC) $(CAR_SW_INCLUDES) $(CHS_SW_CCFLAGS) -c $< -o $@

HOST_FLAGS := -T$(HOST_LD_SCRIPT) -Wno-pointer-to-int-cast -DIntClustNumCores=8 -Dhalf=_Float16 -g -march=rv64gc_zifencei




pulp_cluster_RUNTIME_PAYLOAD_HEADER := $(wildcard $(BUILD_DIR)/codegen/*/include/*_pulp_cluster_runtime_payload.h)
build-payload: build-offload
	@echo "Building runtime payload..."
	$(PYTHON) $(BUILD_DIR)/elf2payload.py --binary $(BUILD_DIR)/build/pulp_cluster_runtime/pulp_cluster_runtime --output $(pulp_cluster_RUNTIME_PAYLOAD_HEADER)
	@echo "Runtime payload header done."



build-host: $(HOST_LIB) build-payload
	@echo "Building host"
	@echo $(HOST_LIB_SRCS_O)
	$(CHS_SW_CC) $(HOST_INCLUDES) $(MATCH_INCLUDES) $(CHS_SW_LDFLAGS) $(HOST_FLAGS) -o $(BUILD_DIR)/host.elf $(HOST_LIB) $(MATCH_COMMON_SRCS) $(MATCH_HOST_SRC) $(CHS_SW_LIBS) -lm
	@echo "Generating objdump"
	@$(CHS_SW_OBJDUMP) -drwCS $(BUILD_DIR)/host.elf > $(BUILD_DIR)/host.dump
	@echo "Host build done"