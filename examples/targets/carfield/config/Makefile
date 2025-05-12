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
MATCH_COMMON_SRCS = $(filter-out %_offload.c %/main.c %/carfield.c, $(MATCH_ALL_SRCS))

MATCH_HOST_SRC = $(filter %/main.c %/carfield.c, $(MATCH_ALL_SRCS))
MATCH_OFFLOAD_SRCS = $(filter %_offload.c, $(MATCH_ALL_SRCS))





# Compile payloads

PULPD_ROOT      ?= $(shell $(BENDER) path pulp_cluster)
PULPD_RISCV		?= /usr/pack/pulpsdk-1.0-kgf/artifactory/pulp-sdk-release/pkg/pulp_riscv_gcc/1.0.16/bin/riscv32-unknown-elf

export PULPRT_TARGET	= carfield-cluster
export PULPRUN_TARGET	= carfield-cluster
export CONFIG_NO_FC		= 1
export ARCHI_HMR		= 1
export PULPRT_HOME 		= $(PULPD_ROOT)/pulp-runtime
export PULP_SDK_HOME    = $(PULPRT_HOME)

PULP_APPS = $(foreach src,$(MATCH_OFFLOAD_SRCS), $(basename $(notdir $(src))))
PULP_APP_SRCS = $(MATCH_COMMON_SRCS) $(PULPRT_HOME)/lib/libc/minimal/io.c $(PULPRT_HOME)/lib/libc/minimal/prf.c 

define PULP_APP_SRCS_FUNC
$(eval PULP_APP_SRCS_$(1) := $(filter %/$(1).c, $(MATCH_ALL_SRCS)))
endef

#$(foreach x, $(PULP_APPS), $(eval PULP_APP_SRCS_$(offload) := $(filter %/$(offload).c, $(MATCH_ALL_SRCS))))
$(foreach x, $(PULP_APPS), $(eval $(call PULP_APP_SRCS_FUNC,$(x))))
PULP_CFLAGS = -O3 $(MATCH_INCLUDES) -DCLUSTER_COMPILATION -DARCHI_CLUSTER_NB_PE=8 -I$(PULPRT_HOME)/lib/libc/minimal/include
PULPD_ELF_REMOVE_SECTIONS := --remove-section .l1cluster_g --remove-section .bss_l1

-include $(PULP_SDK_HOME)/install/rules/pulp.mk

PULP_LDFLAGS := $(filter-out -T%, $(PULP_LDFLAGS))
PULP_TARGETS = $(TARGETS)

build-offload: 
	@echo $(PULP_TARGETS)
	$(MAKE) PULP_LDFLAGS="$(PULP_LDFLAGS) -Tlink_pulpd.ld" CONFIG_LIBC_MINIMAL=0 $(PULP_TARGETS)
	@$(foreach x, $(PULP_APPS), $(PULPD_RISCV)-objcopy $(PULPD_ELF_REMOVE_SECTIONS) $(BUILD_DIR)/build/$(x)/$(x);)

	@echo "Generating objdump"
	@$(foreach x, $(PULP_APPS), $(PULPD_RISCV)-objdump -d -S $(BUILD_DIR)/build/$(x)/$(x) > $(BUILD_DIR)/build/$(x)/$(x).dump;)

	@echo "Offload build done"





# Compile Host

CAR_SW_DIR  := $(CAR_ROOT)/sw
CHS_ROOT 	?= $(shell $(BENDER) path cheshire)

CHS_SW_GCC_BINROOT  ?= /usr/pack/riscv-1.0-kgf/riscv64-gcc-11.2.0/bin

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

HOST_FLAGS := -T$(HOST_LD_SCRIPT) -Wno-pointer-to-int-cast -DIntClustNumCores=8 -g

define ELF2PAYLOAD_FUNC
$(eval $(1)_PAYLOAD_ELF := $(BUILD_DIR)/build/$(1)/$(1))
$(eval $(1)_PAYLOAD_HEADER := $(wildcard $(BUILD_DIR)/include/nodes/*/$(subst _offload,,$(1))_payload.h))
endef
$(foreach x, $(PULP_APPS), $(eval $(call ELF2PAYLOAD_FUNC,$(x))))





build-payloads: build-offload
	@echo "Building payloads"
	
	@$(foreach x, $(PULP_APPS), echo $(x)_PAYLOAD_HEADER=$($(x)_PAYLOAD_HEADER);)
	@$(foreach x, $(PULP_APPS), $(PYTHON) $(BUILD_DIR)/elf2payload.py --binary $($(x)_PAYLOAD_ELF) --output $($(x)_PAYLOAD_HEADER);)
	@echo "Payloads build done"



build-host: $(HOST_LIB) build-payloads
	@echo "Building host"
	@echo $(HOST_LIB_SRCS_O)
	$(CHS_SW_CC) $(HOST_INCLUDES) $(MATCH_INCLUDES) $(CHS_SW_LDFLAGS) $(HOST_FLAGS) -o $(BUILD_DIR)/host.elf $(HOST_LIB) $(MATCH_COMMON_SRCS) $(MATCH_HOST_SRC) $(CHS_SW_LIBS)
	@echo "Generating objdump"
	@$(CHS_SW_OBJDUMP) -d -S $(BUILD_DIR)/host.elf > $(BUILD_DIR)/host.dump
	@echo "Host build done"