#ifndef PULP_RT_MALLOC_WRAPPER_H
#define PULP_RT_MALLOC_WRAPPER_H

#ifdef PULP
#include <pmsis.h>
#else
#include <stdlib.h>
#include <stddef.h>
#endif



// Wrappers for pulp rt_user_alloc
void* malloc_wrapped(size_t size);
void free_wrapped(void* ptr, size_t size);
// Wrappers for pulp wrappers.
/* 
 These wrappers store an extra four byte field before
 before every allocation to store the size, so free
 can be called without the size.

 Rather, free_wrapper will use the header to determine the
 size with a certain allocation and then call
 the free_pulp wrapper with the right size automatically.
 */

void* malloc_wrapper(size_t size);
void free_wrapper(void* wrapped_pointer);

#endif // PULP_RT_MALLOC_WRAPPER_H
