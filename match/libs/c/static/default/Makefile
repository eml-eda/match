# Compiler and flags
CC = gcc

# Paths
LIBS = -L./lib -lm
BUILD_DIR = ./
SRC_DIR = ./src
# Output file name
TARGET = match_app
# Include paths
STANDALONE_CRT_PATH := $(abspath $(BUILD_DIR))/runtime
CODEGEN_PATH := $(abspath $(BUILD_DIR))/codegen
MODEL_DIRS := $(wildcard $(CODEGEN_PATH)/*)
MODELS = $(wildcard $(abspath $(BUILD_DIR))/codegen/*)
INCLUDES = -I${STANDALONE_CRT_PATH}/include \
			 -I${STANDALONE_CRT_PATH}/src/runtime/crt/include \
			 $(foreach model, $(MODEL_DIRS), -I$(model)/include) \
			 -I ./include/

# Source files
SRC =  $(foreach model, $(MODEL_DIRS), $(wildcard $(model)/src/*.c)) $(shell find $(abspath $(SRC_DIR)) -name '*.c')  $(STANDALONE_CRT_PATH)/src/runtime/crt/common/crt_backend_api.c $(STANDALONE_CRT_PATH)/src/runtime/crt/memory/stack_allocator.c


CFLAGS = -Wall -Wextra -O3 -g $(INCLUDES)
# -pg -g
CFLAGS += -ffast-math -funroll-loops -fomit-frame-pointer -fno-strict-aliasing -fno-common -fno-builtin -fno-stack-protector -fno-exceptions -fno-rtti -fno-threadsafe-statics -fno-strict-overflow
# Object files (derived from source files)
OBJ = $(SRC:.c=.o)

# Rules
all: $(TARGET)

$(TARGET): $(OBJ)
	$(CC) $(CFLAGS) -o $@ $^ $(LIBS)

%.o: %.c
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

clean:
	rm -f $(OBJ) $(TARGET)

.PHONY: all clean
