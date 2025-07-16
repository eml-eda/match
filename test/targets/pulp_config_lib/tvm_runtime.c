/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */
#include <tvm_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

#define __PREFER_MATCH_MEM__ 1

void __attribute__((noreturn)) TVMPlatformAbort(tvm_crt_error_t error_code) {
  abort();
  exit(-1);
}

int TVMPlatformMemoryAllocate(size_t num_bytes, DLDevice dev, void** out_ptr) {
  #ifndef PULP_ADRESSABLE_RAM
  #ifdef __PREFER_MATCH_MEM__
  *out_ptr = match_try_alloc_from_match_mem(num_bytes);
  if(*out_ptr == NULL)  *out_ptr = malloc_wrapper(num_bytes);
  #else
  *out_ptr = malloc_wrapper(num_bytes);
  if(*out_ptr == NULL)  *out_ptr = match_try_alloc_from_match_mem(num_bytes);
  #endif
  #else
  printf("[TVMPlatformMemoryAllocate] Allocating %zu bytes\n", num_bytes);
  int* ram_pt = pulp_alloc_ram(num_bytes + 4);
  *ram_pt = (int)num_bytes; // Store the size in the first 4 bytes
  *out_ptr = (void*)(((uint8_t*)ram_pt) + 4); // Return the pointer after the size header
  #endif
  // Return nonzero exit code to caller on failure to allocate
  if (*out_ptr == NULL){
    return 1;
  }
  return 0;
}

int TVMPlatformMemoryFree(void* ptr, DLDevice dev) {
  #ifndef PULP_ADRESSABLE_RAM
  if(match_try_free_from_match_mem(ptr) != 0) {
    free_wrapper(ptr);
  }
  #else
  printf("[TVMPlatformMemoryFree] Freeing memory at %p\n", ptr);
  void* actual_ptr = (void*)((uint8_t*)ptr - 4); // Get the pointer before the size header
  int size = *((int*)actual_ptr); // Read the size from the first 4
  printf("[TVMPlatformMemoryFree] Freeing %d bytes at %p\n", size + 4, actual_ptr);
  ram_free(actual_ptr, size + 4); // Free the memory including the size header
  #endif
  return 0;
}

void TVMLogf(const char* msg, ...) {
  // FIX for GAP9
  printf(msg);
  //va_list args;
  //va_start(args, msg);
  //vfprintf(stdout, msg, args);
  //va_end(args);
}

TVM_DLL int TVMFuncRegisterGlobal(const char* name, TVMFunctionHandle f, int override) { return 0; }

#ifdef __cplusplus
}
#endif