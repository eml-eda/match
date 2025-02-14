#include <pulp_target/malloc_wrapper.h>
#include <stdint.h>

#ifdef PULP
#include <pmsis.h>
#endif

// This number should never be negative
volatile uint32_t peak_l2_alloc = 0;
volatile uint32_t current_l2_alloc = 0;

void log_peak_alloc(size_t size);
void log_peak_free(size_t size);

void* malloc_wrapped(size_t size){
#ifdef PULP
    //rt_alloc_t *shared_l2_allocator = &__rt_alloc_l2[2];
    //void* ptr = rt_user_alloc(shared_l2_allocator, size);
    void *ptr = pi_l2_malloc(size);
#else //fallback x86 compilation
    void* ptr = malloc(size);
#endif
    return ptr;
}

void free_wrapped(void* ptr, size_t size){
#ifdef PULP
    pi_l2_free(ptr, size);
#else //fallback x86 compilation
    // This implementation actually doesn't need a wrapper, so size is ignored
    free(ptr);
#endif
}

void* malloc_wrapper(size_t size){
  // Allocate extra 4 bytes header in the first that stores the size
  size_t actual_size = size + 4;
  void* pointer = malloc_wrapped(actual_size);
  if (pointer == NULL){
      return NULL;
  }
  // Write to this value as if it where a uint32_t
  ((uint32_t*)pointer)[0] = (uint32_t)size;
  // return the allocated section without the header
  log_peak_alloc(actual_size);
  void* wrapped_pointer = pointer + 4;
  return wrapped_pointer;
}

void free_wrapper(void* wrapped_pointer){
  // The actual allocation started from ptr - 4
  void* actual_ptr = wrapped_pointer - 4;
  // unpack the header of the pointer
  uint32_t size = ((uint32_t*)actual_ptr)[0];
  // Actual size is size + 4
  log_peak_free(size + 4);
  free_wrapped(actual_ptr, size);
}

void log_peak_free(size_t size){
   current_l2_alloc = current_l2_alloc - (uint32_t)size;
}

void log_peak_alloc(size_t size){
   current_l2_alloc = current_l2_alloc + (uint32_t)size;
   if (current_l2_alloc > peak_l2_alloc){
       peak_l2_alloc = current_l2_alloc;
   }
}
