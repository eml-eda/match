#ifndef __PULP_MEM_RAM_H__
#define __PULP_MEM_RAM_H__

#include <pulp_mem/mem.h>

void* pulp_init_ram(int size);

void pulp_load_file(const char* filename, void* ext_pt, int size);

void pulp_memcpy_from_ram(void* l2_pt, void* ext_pt, int size);

void pulp_memcpy_to_ram(void* l2_pt, void* ext_pt, int size);

void pulp_shutdown_ram(void* ext_pt, int size);

#endif