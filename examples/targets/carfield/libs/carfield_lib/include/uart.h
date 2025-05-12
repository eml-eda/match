// Copyright 2022 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Nils Wistoff <nwistoff@iis.ee.ethz.ch>
// Paul Scheffler <paulsc@iis.ee.ethz.ch>

#ifndef CAR_LIB_UART_H
#define CAR_LIB_UART_H

#include <stdint.h>

extern void *__base_uart;

// Register offsets
#define CAR_UART_RBR_REG_OFFSET 0
#define CAR_UART_THR_REG_OFFSET 0
#define CAR_UART_INTR_ENABLE_REG_OFFSET 4
#define CAR_UART_INTR_IDENT_REG_OFFSET 8
#define CAR_UART_FIFO_CONTROL_REG_OFFSET 8
#define CAR_UART_LINE_CONTROL_REG_OFFSET 12
#define CAR_UART_MODEM_CONTROL_REG_OFFSET 16
#define CAR_UART_LINE_STATUS_REG_OFFSET 20
#define CAR_UART_MODEM_STATUS_REG_OFFSET 24
#define CAR_UART_DLAB_LSB_REG_OFFSET 0
#define CAR_UART_DLAB_MSB_REG_OFFSET 4

// Register fields
#define CAR_UART_LINE_STATUS_DATA_READY_BIT 0
#define CAR_UART_LINE_STATUS_THR_EMPTY_BIT 5
#define CAR_UART_LINE_STATUS_TMIT_EMPTY_BIT 6

void car_uart_init(void *uart_base, uint32_t freq, uint32_t baud);

int car_uart_read_ready(void *uart_base);

void car_uart_write(void *uart_base, uint8_t byte);

void car_uart_write_str(void *uart_base, void *src, uint32_t len);

void car_uart_write_flush(void *uart_base);

uint8_t car_uart_read(void *uart_base);

void car_uart_read_str(void *uart_base, void *dst, uint32_t len);

void car_uart_print_str(char* str);

#endif // CAR_LIB_UART_H