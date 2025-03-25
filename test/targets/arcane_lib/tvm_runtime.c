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
 
 int TVMPlatformMemoryAllocate(size_t num_bytes, DLDevice dev, void** out_ptr) {
   *out_ptr = malloc(num_bytes);
   // Return nonzero exit code to caller on failure to allocate
   if (*out_ptr == NULL){
       return 1;
   }
   return 0;
 }
 
 int TVMPlatformMemoryFree(void* ptr, DLDevice dev) {
   free(ptr);
   return 0;
 }
 
 void TVMLogf(const char* msg, ...) {
   printf(msg);
   //va_list args;
   //va_start(args, msg);
   //vfprintf(stdout, msg, args);
   //va_end(args);
 }
 
 #ifdef __cplusplus
 }
 #endif