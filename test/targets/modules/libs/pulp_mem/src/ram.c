#include <pulp_mem/ram.h>

void* pulp_init_ram(int size){
    mem_init();
    return ram_malloc(size);
}

void pulp_load_file(const char* filename, void* ext_pt, int size){
    load_file_to_ram(ext_pt, filename);
}

void pulp_memcpy_from_ram(void* l2_pt, void* ext_pt, int size){
    ram_read(l2_pt, ext_pt, size);
}

void pulp_memcpy_to_ram(void* l2_pt, void* ext_pt, int size){
    ram_read(ext_pt, l2_pt, size);
}

void pulp_shutdown_ram(void* ext_pt, int size){
    ram_free(ext_pt, size);
}
