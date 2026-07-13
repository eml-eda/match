/* 
 * Basic malloc implementation for L2 SPM, 
 * pointers are uint32_t so that, in theory,
 * the same memory pool could be shared between 
 * 64bit host and 32bit cluster cores.
 */

#include "carfield_lib/malloc.h"
#include "carfield_lib/printf.h"

#include <stddef.h>
#include <stdint.h>

#define DEBUG_MALLOC 0

uint8_t* memory_pool_l2 = &__l2_heap_start;
block_header_t* free_list = NULL;


static size_t l2_heap_size(void) {
    return (size_t)(&__l2_heap_end - &__l2_heap_start);
}

// Offset helpers
static inline uint32_t ptr_to_offset(void* ptr) {
    if (!ptr) return 0;
    return (uint32_t)((uint8_t*)ptr - memory_pool_l2);
}

static inline block_header_t* offset_to_ptr(uint32_t offset) {
    if (!offset) return NULL;
    return (block_header_t*)(memory_pool_l2 + offset);
}

static void print_free_list() {
    mini_printf("Free list:\r\n");
    block_header_t* curr = free_list;
    while (curr != NULL) {
        mini_printf("  Block at %p: size %d, is_free %d, next %p\r\n", 
                    curr, (int)curr->size, curr->is_free, offset_to_ptr(curr->next));
        curr = offset_to_ptr(curr->next);
    }
}

/**
 * Initialize the memory allocator
 */
void mem_init_l2(void) {
    // Create initial free block spanning the entire memory pool
    free_list = (block_header_t*)memory_pool_l2;
    free_list->size = (uint32_t)l2_heap_size() - sizeof(block_header_t);
    free_list->is_free = 1;
    free_list->next = 0;
#if DEBUG_MALLOC
    mini_printf("[MALLOC] L2 memory pool initialized with size %d bytes.\r\n", (int)free_list->size);
#endif
}



void *malloc_l2(size_t size) {
    block_header_t *curr, *prev;
    void *res = NULL;

    size = (((size + sizeof(block_header_t) - 1) / sizeof(block_header_t)) + 2) * sizeof(block_header_t); // Align size to 8 bytes
    if (size < MIN_ALLOC_SIZE) size = MIN_ALLOC_SIZE;

    if (free_list == NULL) mem_init_l2();

    curr = free_list;
    prev = NULL;

    while (curr != NULL) {
        if (curr->is_free && curr->size >= size) {
            if (curr->size >= size + sizeof(block_header_t) + MIN_ALLOC_SIZE) {
                block_header_t *new_block = (block_header_t*)((uint8_t*)curr + sizeof(block_header_t) + size);
                new_block->size = curr->size - size - sizeof(block_header_t);
                new_block->is_free = 1;
                new_block->next = curr->next;

                curr->size = (uint32_t)size;
                curr->next = ptr_to_offset(new_block);
            }
            curr->is_free = 0;

            res = (void*)((uint8_t*)curr + sizeof(block_header_t));
#if DEBUG_MALLOC
            mini_printf("[MALLOC] Allocated %d bytes at %p\r\n", (int)size, res);
            print_free_list();
#endif
            return res;
        }

        prev = curr;
        curr = offset_to_ptr(curr->next);
    }
#if DEBUG_MALLOC
    mini_printf("[MALLOC] Could not find suitable block in free list.\r\n");
#endif
    return NULL;
}


void free_l2(void* ptr) {
    if (ptr == NULL) return;

    block_header_t *curr, *prev, *next;

    curr = (block_header_t*)((uint8_t*)ptr - sizeof(block_header_t));
    curr->is_free = 1;

    next = offset_to_ptr(curr->next);
    if (next != NULL && next->is_free) {
        curr->size += sizeof(block_header_t) + next->size;
        curr->next = next->next;
    }

    prev = free_list;
    if (prev == curr) goto end;

    while (prev != NULL && offset_to_ptr(prev->next) != curr) {
        prev = offset_to_ptr(prev->next);
    }

    if (prev != NULL && prev->is_free) {
        prev->size += sizeof(block_header_t) + curr->size;
        prev->next = curr->next;
    }

end:
#if DEBUG_MALLOC
    mini_printf("[MALLOC] Freed memory at %p, size %d bytes\r\n", ptr, (int)curr->size);
    print_free_list();
#endif
    return;
}


void* malloc(size_t size) {
    return malloc_l2(size);
}

void free(void* ptr) {
    free_l2(ptr);
}