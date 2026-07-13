#ifndef CAR_LIB_PRINTF_H
#define CAR_LIB_PRINTF_H

#include <stdarg.h>
#include <stddef.h>

void mini_itoa(int value, char *str, int base);

void mini_ultoa(unsigned long value, char *str, int base);

size_t mini_vsnprintf(char *out, size_t n, const char *fmt, va_list args);

size_t mini_snprintf(char *out, size_t n, const char *fmt, ...);

void mini_vprintf(const char *fmt, va_list args);

void mini_printf(const char *fmt, ...);

#endif // CAR_LIB_PRINTF_H