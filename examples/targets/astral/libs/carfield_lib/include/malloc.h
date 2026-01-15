#ifndef CAR_LIB_MALLOC_H
#define CAR_LIB_MALLOC_H

#include <stdint.h>
#include <stddef.h>

// Import heap symbols defined in the linker script
extern uint8_t __l2_heap_start;
extern uint8_t __l2_heap_end;

// Memory block header structure
typedef struct block_header {
    uint32_t size;      // size_t       
    uint32_t is_free;   // bool 
    uint32_t next;      // block_header_t*
} block_header_t;

// Memory pool
extern uint8_t* memory_pool_l2;
extern block_header_t* free_list;

#define MIN_ALLOC_SIZE sizeof(block_header_t)

void mem_init_l2(void);
void* malloc_l2(size_t size);
void free_l2(void* ptr);

void* malloc(size_t size);
void free(void* ptr);

#endif // CAR_LIB_MALLOC_H
