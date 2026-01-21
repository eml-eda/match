#include "carfield_lib/printf.h"

#include <stdarg.h>
#include <stddef.h>
#include <stdio.h>
#include <math.h> 

#include "carfield_lib/uart.h"


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


static void mini_ftoa(double f, char *buf, int precision) {
    if (isnan(f)) {
        buf[0] = 'n'; buf[1] = 'a'; buf[2] = 'n'; buf[3] = '\0';
        return;
    }
    if (isinf(f)) {
        if (f < 0) {
            buf[0] = '-'; buf[1] = 'i'; buf[2] = 'n'; buf[3] = 'f'; buf[4] = '\0';
        } else {
            buf[0] = 'i'; buf[1] = 'n'; buf[2] = 'f'; buf[3] = '\0';
        }
        return;
    }
    if (f < 0) {
        *buf++ = '-';
        f = -f;
    }
    unsigned long ipart = (unsigned long)f;
    double fpart = f - (double)ipart;

    // Integer part
    char tmp[20];
    mini_ultoa(ipart, tmp, 10);
    char *p = tmp;
    while (*p) *buf++ = *p++;

    // Decimal point and fractional part
    if (precision > 0) {
        *buf++ = '.';
        // Multiply out for specified precision, round correctly
        double rounding = 0.5;
        for (int i = 0; i < precision; ++i)
            rounding /= 10.0;
        fpart += rounding;

        for (int i = 0; i < precision; ++i) {
            fpart *= 10.0;
            int digit = (int)fpart;
            *buf++ = '0' + digit;
            fpart -= digit;
        }
    }
    *buf = '\0';
}


size_t mini_vsnprintf(char *out, size_t n, const char *fmt, va_list args) {
    char *out_ptr = out;
    size_t remaining = n;

    if (n == 0) return 0;

    char buffer[32];
    
    while (*fmt && remaining > 1) {
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
                if (str) {
                    while (*str && remaining > 1) {
                        *out_ptr++ = *str++;
                        remaining--;
                    }
                } else {
                    const char *null_str = "NULL";
                    for (const char *p = null_str; *p && remaining > 1; p++) {
                        *out_ptr++ = *p;
                        remaining--;
                    }
                }
            } else if (*fmt == 'x') {
                unsigned int val = va_arg(args, unsigned int);
                mini_itoa(val, buffer, 16);
                if (remaining > 2) {
                    *out_ptr++ = '0'; remaining--;
                    *out_ptr++ = 'x'; remaining--;
                    for (char *p = buffer; *p && remaining > 1; p++) {
                        *out_ptr++ = *p;
                        remaining--;
                    }
                }
            } else if (*fmt == 'p' || *fmt == 'P') {
                void *ptr = va_arg(args, void*);
                if (ptr == NULL) {
                    const char *null_ptr = "NULL";
                    for (const char *p = null_ptr; *p && remaining > 1; p++) {
                        *out_ptr++ = *p;
                        remaining--;
                    }
                } else {
                    unsigned long ptr_val = (unsigned long)ptr;
                    mini_ultoa(ptr_val, buffer, 16);

                    int len = 0;
                    for (char *p = buffer; *p; p++) len++;

                    if (remaining > 2) {
                        *out_ptr++ = '0'; remaining--;
                        *out_ptr++ = 'x'; remaining--;
                        int target_width;
                        int is_64bit = 0;
                        #if defined(__LP64__) || defined(_LP64) || defined(__x86_64__) || defined(_M_X64)
                        is_64bit = 1;
                        #endif
                        if (*fmt == 'P' || (is_64bit && len > 8)) {
                            target_width = 16;
                        } else {
                            target_width = 8;
                        }
                        int padding = target_width - len;
                        while (padding > 0 && remaining > 1) {
                            *out_ptr++ = '0';
                            remaining--;
                            padding--;
                        }
                        for (char *p = buffer; *p && remaining > 1; p++) {
                            *out_ptr++ = *p;
                            remaining--;
                        }
                    }
                }
            } else if (*fmt == 'f') {
                // Default: 6 decimal places
                double val = va_arg(args, double);
                mini_ftoa(val, buffer, 6);
                for (char *p = buffer; *p && remaining > 1; p++) {
                    *out_ptr++ = *p;
                    remaining--;
                }
            } else if (*fmt == '%') {
                if (remaining > 1) {
                    *out_ptr++ = '%';
                    remaining--;
                }
            } else {
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
    *out_ptr = '\0';
    return n - remaining;
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


void mini_vprintf(const char *fmt, va_list args) {
    char buf[128];
    size_t len = mini_vsnprintf(buf, sizeof(buf), fmt, args);
    car_uart_print_str(buf);
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