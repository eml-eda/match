#include "crappy_runtime/printf.h"

#include <stdarg.h>
#include <stddef.h>
#include <stdio.h>

#include "carfield_lib/uart.h"

// Convert integers to strings with support for different sizes
void mini_itoa(int value, char *str, int base) {
    char *ptr = str, *ptr1 = str, tmp_char;
    int tmp_value;

    if (value < 0 && base == 10) {
        *ptr++ = '-';
        value = -value;
    }

    do {
        tmp_value = value;
        value /= base;
        *ptr++ = "0123456789ABCDEF"[tmp_value - value * base];
    } while (value);

    *ptr-- = '\0';

    if (*str == '-') ptr1++;
    while (ptr1 < ptr) {
        tmp_char = *ptr;
        *ptr-- = *ptr1;
        *ptr1++ = tmp_char;
    }
}

// Convert unsigned long integers to strings (for pointers)
void mini_ultoa(unsigned long value, char *str, int base) {
    char *ptr = str, *ptr1 = str, tmp_char;
    unsigned long tmp_value;

    do {
        tmp_value = value;
        value /= base;
        *ptr++ = "0123456789ABCDEF"[tmp_value - value * base];
    } while (value);

    *ptr-- = '\0';

    while (ptr1 < ptr) {
        tmp_char = *ptr;
        *ptr-- = *ptr1;
        *ptr1++ = tmp_char;
    }
}


size_t mini_vsnprintf(char *out, size_t n, const char *fmt, va_list args) {
    char *out_ptr = out;
    size_t remaining = n;

    if (n == 0) return 0;  // Handle zero-sized buffer

    char buffer[20]; // Increased to handle 64-bit pointers (16 hex digits + null terminator)

    while (*fmt && remaining > 1) {  // Keep space for null terminator
        if (*fmt == '%') {
            fmt++;
            if (*fmt == 'd' || *fmt == 'i') {
                int val = va_arg(args, int);
                mini_itoa(val, buffer, 10);
                for (char *p = buffer; *p && remaining > 1; p++) {
                    *out_ptr++ = *p;
                    remaining--;
                }
            } else if (*fmt == 's') {
                char *str = va_arg(args, char*);
                if (str) {  // Check for NULL pointer
                    while (*str && remaining > 1) {
                        *out_ptr++ = *str++;
                        remaining--;
                    }
                } else {
                    // Handle NULL string
                    const char *null_str = "NULL";
                    for (const char *p = null_str; *p && remaining > 1; p++) {
                        *out_ptr++ = *p;
                        remaining--;
                    }
                }
            } else if (*fmt == 'x') {
                unsigned int val = va_arg(args, unsigned int);
                mini_itoa(val, buffer, 16);
                // Only add 0x prefix if there's room
                if (remaining > 2) {
                    *out_ptr++ = '0'; remaining--;
                    *out_ptr++ = 'x'; remaining--;
                    for (char *p = buffer; *p && remaining > 1; p++) {
                        *out_ptr++ = *p;
                        remaining--;
                    }
                }
            } else if (*fmt == 'p' || *fmt == 'P') {
                // Handle pointer type with proper casting
                void *ptr = va_arg(args, void*);
                if (ptr == NULL) {
                    // Handle NULL pointer
                    const char *null_ptr = "NULL";
                    for (const char *p = null_ptr; *p && remaining > 1; p++) {
                        *out_ptr++ = *p;
                        remaining--;
                    }
                } else {
                    // Convert pointer to hex representation with proper size
                    unsigned long ptr_val = (unsigned long)ptr;
                    mini_ultoa(ptr_val, buffer, 16);
                    
                    // Add leading zeros to ensure consistent width
                    int len = 0;
                    for (char *p = buffer; *p; p++) len++;
                    
                    // Add 0x prefix and pad with zeros if there's room
                    if (remaining > 2) {
                        *out_ptr++ = '0'; remaining--;
                        *out_ptr++ = 'x'; remaining--;
                        
                        // Add padding zeros for consistent pointer width
                        // For 32-bit: 8 hex digits, For 64-bit: 16 hex digits
                        int target_width;
                        
                        // Determine if we're using a 32-bit or 64-bit pointer
                        int is_64bit = 0;
                        #if defined(__LP64__) || defined(_LP64) || defined(__x86_64__) || defined(_M_X64)
                        is_64bit = 1;
                        #endif
                        
                        // When using %p, print as 32-bit (8 hex digits)
                        // When using %P, print as 64-bit (16 hex digits)
                        // Or when using %p but the pointer requires 64-bit representation
                        if (*fmt == 'P' || (is_64bit && len > 8)) {
                            target_width = 16;  // 64-bit format
                        } else {
                            target_width = 8;   // 32-bit format
                        }
                        
                        int padding = target_width - len;
                        while (padding > 0 && remaining > 1) {
                            *out_ptr++ = '0';
                            remaining--;
                            padding--;
                        }
                        
                        // Add the actual hex digits
                        for (char *p = buffer; *p && remaining > 1; p++) {
                            *out_ptr++ = *p;
                            remaining--;
                        }
                    }
                }
            } else if (*fmt == '%') {
                // Handle %% escape
                if (remaining > 1) {
                    *out_ptr++ = '%';
                    remaining--;
                }
            } else {
                // Unknown specifier, print as is
                if (remaining > 1) {
                    *out_ptr++ = '%'; remaining--;
                }
                if (*fmt && remaining > 1) {
                    *out_ptr++ = *fmt; remaining--;
                }
            }
        } else {
            *out_ptr++ = *fmt;
            remaining--;
        }
        fmt++;
    }
    *out_ptr = '\0';  // Null-terminate
    
    return n - remaining;  // Return number of characters written (not including null terminator)
}


size_t mini_snprintf(char *out, size_t n, const char *fmt, ...) {
    va_list args;
    va_start(args, fmt);
    
    // Call the va_list version of the function
    size_t ret = mini_vsnprintf(out, n, fmt, args);
    
    va_end(args);
    return ret;
}


int printf(const char *fmt, ...) {
    char buf[128]; 
    va_list args;
    va_start(args, fmt);
    
    // Use vsnprintf instead of snprintf with va_list
    size_t len = mini_vsnprintf(buf, sizeof(buf), fmt, args);
    
    va_end(args);

    car_uart_print_str(buf);

    return len; 
}


void mini_printf(const char *fmt, ...) {
    char buf[128];
    va_list args;
    va_start(args, fmt);
    
    // Use vsnprintf instead of snprintf with va_list
    size_t len = mini_vsnprintf(buf, sizeof(buf), fmt, args);
    
    va_end(args);

    car_uart_print_str(buf);
}


//#endif