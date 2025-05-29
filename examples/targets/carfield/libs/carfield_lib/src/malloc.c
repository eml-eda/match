/**
 * Malloc implementation for bare metal systems using linker-defined heap
 */

#include "carfield_lib/malloc.h"
#include "carfield_lib/carfield.h"

#include <stddef.h>
#include <stdint.h>


uint8_t* memory_pool_l2 = &__l2_heap_start;
block_header_t* free_list = NULL;

/**
 * Initialize the memory allocator
 */
void mem_init_l2(void) {
    // Create initial free block spanning the entire memory pool
    free_list = (block_header_t*)memory_pool_l2;
    free_list->size = (uint32_t)(&__l2_heap_end - &__l2_heap_start);
    free_list->is_free = 1;
    free_list->next = 0;
}

/**
 * Allocate memory of specified size
 * 
 * @param size Size of memory to allocate in bytes
 * @return Pointer to allocated memory or NULL if allocation fails
 */
void* malloc_l2(size_t size) {
    carprint("malloc\r\n");
    block_header_t *curr, *prev, *new_block;
    void* result = NULL;
    
    // Adjust size to include the header and ensure alignment (8-byte in this case)
    size_t aligned_size = (size + sizeof(block_header_t) + 7) & ~7;
    
    // Ensure minimum allocation size
    if (aligned_size < MIN_ALLOC_SIZE + sizeof(block_header_t))
        aligned_size = MIN_ALLOC_SIZE + sizeof(block_header_t);
    
    // Initialize memory pool if not already done
    if (free_list == NULL)
        mem_init_l2();
    
    // First-fit search for a free block
    prev = NULL;
    curr = free_list;
    
    while (curr != NULL) {
        if (curr->is_free && curr->size >= aligned_size) {
            // Found a suitable block
            
            // Split the block if it's significantly larger than requested
            if (curr->size >= aligned_size + sizeof(block_header_t) + MIN_ALLOC_SIZE) {
                new_block = (block_header_t*)((uint8_t*)curr + aligned_size);
                new_block->size = curr->size - aligned_size;
                new_block->is_free = 1;
                new_block->next = curr->next;
                
                curr->size = aligned_size;
                curr->next = (uint32_t)new_block;
            }
            
            // Mark block as allocated
            curr->is_free = 0;
            
            // Return pointer to usable memory (after header)
            result = (void*)((uint8_t*)curr + sizeof(block_header_t));
            break;
        }
        
        prev = curr;
        curr = (block_header_t*)curr->next;
    }
    
    return result;
}

/**
 * Free previously allocated memory
 * 
 * @param ptr Pointer to memory to free
 */
void free_l2(void* ptr) {
    block_header_t *block, *next, *prev;
    
    if (ptr == NULL)
        return;
    
    // Get the block header from the pointer
    block = (block_header_t*)((uint8_t*)ptr - sizeof(block_header_t));
    
    // Sanity check - ensure the pointer is within our heap
    if ((uint8_t*)block < memory_pool_l2 || 
        (uint8_t*)block >= memory_pool_l2 + (&__l2_heap_end - &__l2_heap_start))
        return;  // Ignore attempts to free memory outside our heap
    
    // Mark block as free
    block->is_free = 1;
    
    // Coalesce with adjacent free blocks
    
    // Find the previous block
    prev = NULL;
    next = free_list;
    while (next != NULL && next < block) {
        prev = next;
        next = (block_header_t*)next->next;
    }
    
    // Merge with next block if adjacent and free
    if ((uint8_t*)block + block->size == (uint8_t*)next && next->is_free) {
        block->size += next->size;
        block->next = next->next;
    } else {
        block->next = (uint32_t)next;
    }
    
    // Merge with previous block if adjacent and free
    if (prev != NULL && (uint8_t*)prev + prev->size == (uint8_t*)block && prev->is_free) {
        prev->size += block->size;
        prev->next = block->next;
    } else if (prev != NULL) {
        prev->next = (uint32_t)block;
    } else {
        free_list = block;
    }
}

// [calloc and realloc implementations remain the same as before]


void* malloc(size_t size) {
    return malloc_l2(size);
}

void free(void* ptr) {
    free_l2(ptr);
}