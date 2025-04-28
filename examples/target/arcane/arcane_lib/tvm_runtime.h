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
 
#ifndef __MATCH_ARCANE_TVM_RUNTIME_H__
#define __MATCH_ARCANE_TVM_RUNTIME_H__
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <tvm/runtime/c_runtime_api.h>
#include <tvm/runtime/crt/stack_allocator.h>
 
#ifdef __cplusplus
extern "C" {
#endif
 
int TVMPlatformMemoryAllocate(size_t num_bytes, DLDevice dev, void** out_ptr);
 
int TVMPlatformMemoryFree(void* ptr, DLDevice dev);
 
void TVMLogf(const char* msg, ...);
 
#ifdef __cplusplus
}
#endif
#endif