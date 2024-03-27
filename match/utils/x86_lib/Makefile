# General settings
BUILD_DIR := .

# Note that this value can automatically be changed by python scripts:
OPT_LEVEL = 0

# Compiler options
PKG_COMPILE_OPTS = -Wall -O$(OPT_LEVEL) -std=gnu99 #-fsanitize=address
# For debug support - comment these out for reduced binary size
PKG_COMPILE_OPTS += -g

# Include paths
STANDALONE_CRT_PATH := $(abspath $(BUILD_DIR))/runtime
CODEGEN_PATH := $(abspath $(BUILD_DIR))/codegen

# Sources
CODEGEN_SRCS = $(wildcard $(abspath $(BUILD_DIR))/codegen/host/src/*.c)
CODEGEN_OBJS = $(subst .c,.o,$(CODEGEN_SRCS))

# Compiler selection
CC = gcc
AR = ar
RANLIB = ranlib

# Compiler options and header includes
PKG_CFLAGS = ${PKG_COMPILE_OPTS}\
	     -I${STANDALONE_CRT_PATH}/include \
	     -I${STANDALONE_CRT_PATH}/src/runtime/crt/include \
	     -I${CODEGEN_PATH}/host/include \
	     -I${BUILD_DIR}/include 

# These targets don't make self-titled files
.PHONY: all clean

all: $(BUILD_DIR)/stack_allocator.o $(BUILD_DIR)/crt_backend_api.o  $(BUILD_DIR)/libcodegen.a $(BUILD_DIR)/main

$(BUILD_DIR)/stack_allocator.o: $(STANDALONE_CRT_PATH)/src/runtime/crt/memory/stack_allocator.c
	mkdir -p $(@D)
	$(CC) -c $(PKG_CFLAGS) -o $@ $^

$(BUILD_DIR)/crt_backend_api.o: $(STANDALONE_CRT_PATH)/src/runtime/crt/common/crt_backend_api.c
	mkdir -p $(@D)
	$(CC) -c $(PKG_CFLAGS) -o $@ $^

$(BUILD_DIR)/libcodegen.a: $(CODEGEN_SRCS)
	cd $(abspath $(BUILD_DIR)/codegen/host/src) && $(CC) -c $(PKG_CFLAGS) $(CODEGEN_SRCS)
	$(AR) -cr $(abspath $(BUILD_DIR)/libcodegen.a) $(CODEGEN_OBJS)
	$(RANLIB) $(abspath $(BUILD_DIR)/libcodegen.a)

$(BUILD_DIR)/main: $(BUILD_DIR)/src/main.c $(BUILD_DIR)/stack_allocator.o $(BUILD_DIR)/crt_backend_api.o $(BUILD_DIR)/libcodegen.a
	mkdir -p $(@D)
	$(CC) $(PKG_CFLAGS) -o  $@ $^ -lm

clean:
	rm -f $(CODEGEN_OBJS) $(BUILD_DIR)/stack_allocator.o $(BUILD_DIR)/crt_backend_api.o $(BUILD_DIR)/libcodegen.a $(BUILD_DIR)/main