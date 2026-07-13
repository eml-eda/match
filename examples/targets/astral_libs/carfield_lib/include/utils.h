#ifndef CAR_LIB_UTILS_H
#define CAR_LIB_UTILS_H

#include <stdint.h>
#include <stddef.h>

extern const uint32_t crc32_table[256];

uint32_t crc32(const void* buf, size_t size);

#endif