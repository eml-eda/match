#include "carfield_lib/uart.h"

#include <stdint.h>

#define REG8(base, offs) ((volatile uint8_t *)(base + offs))

void car_uart_init(void *base, uint32_t freq, uint32_t baud) {
    uint32_t div = freq / (baud << 4);
    *REG8(base, CAR_UART_INTR_ENABLE_REG_OFFSET) = 0;       // Disable interrupts
    *REG8(base, CAR_UART_LINE_CONTROL_REG_OFFSET) = 0x80;   // Enable DLAB
    *REG8(base, CAR_UART_DLAB_LSB_REG_OFFSET) = div;        // divisor (lo)
    *REG8(base, CAR_UART_DLAB_MSB_REG_OFFSET) = div >> 8;   // divisor (hi)
    *REG8(base, CAR_UART_LINE_CONTROL_REG_OFFSET) = 0x03;   // 8N1
    *REG8(base, CAR_UART_FIFO_CONTROL_REG_OFFSET) = 0xC7;   // Enable FIFO, clear, 14B threshold
    *REG8(base, CAR_UART_MODEM_CONTROL_REG_OFFSET) = 0x20;  // Autoflow
}

int car_uart_read_ready(void *base) {
    return *REG8(base, CAR_UART_LINE_STATUS_REG_OFFSET) & (1 << CAR_UART_LINE_STATUS_DATA_READY_BIT);
}

static inline int __uart_write_ready(void *base) {
    return *REG8(base, CAR_UART_LINE_STATUS_REG_OFFSET) & (1 << CAR_UART_LINE_STATUS_THR_EMPTY_BIT);
}

static inline int __uart_write_idle(void *base) {
    return __uart_write_ready(base) && 
           (*REG8(base, CAR_UART_LINE_STATUS_REG_OFFSET) & (1 << CAR_UART_LINE_STATUS_TMIT_EMPTY_BIT));
}

void car_uart_write(void *base, uint8_t byte) {
    while (!__uart_write_ready(base));
    *REG8(base, CAR_UART_THR_REG_OFFSET) = byte;
}

void car_uart_write_str(void *base, void *src, uint32_t len) {
    uint8_t *s = (uint8_t*)src;
    while (len--) car_uart_write(base, *s++);
}

void car_uart_write_flush(void *base) {
    asm volatile("fence" ::: "memory");
    while (!__uart_write_idle(base));
}

uint8_t car_uart_read(void *base) {
    while (!car_uart_read_ready(base));
    return *REG8(base, CAR_UART_RBR_REG_OFFSET);
}

void car_uart_read_str(void *base, void *dst, uint32_t len) {
    uint8_t *d = (uint8_t*)dst;
    while (len--) *d++ = car_uart_read(base);
}

void car_uart_print_str(char* str) {
    car_uart_write_str(&__base_uart, str, strlen(str));
    car_uart_write_flush(&__base_uart);
}