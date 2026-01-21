/* Copyright 2022 ETH Zurich and University of Bologna. */
/* Licensed under the Apache License, Version 2.0, see LICENSE for details. */
/* SPDX-License-Identifier: Apache-2.0 */

/* Nicole Narr <narrn@student.ethz.ch> */
/* Christopher Reinwardt <creinwar@student.ethz.ch> */
/* Paul Scheffler <paulsc@iis.ee.ethz.ch> */

/* This header defines symbols and rules universal to bare-metal execution */

ENTRY(_start)

MEMORY {
  bootrom (rx)  : ORIGIN = 0x02000000, LENGTH = 16K
  /* We assume at least 64 KiB SPM, same minus stack for ROMs. */
  /* If more SPM is available, CRT0 repoints the stack. */
  extrom (rx)   : ORIGIN = 0x00000000, LENGTH = 48K
  spm (rwx)     : ORIGIN = 0x10000000, LENGTH = 64K
  /* We  assume at least 8 MiB of DRAM (minimum for Linux). */
  dram (rwx)    : ORIGIN = 0x80000000, LENGTH = 8M
  l2 (rwx)       : ORIGIN = 0x78000000, LENGTH = 32K
  l2_common (rw) : ORIGIN = 0x7800FF00, LENGTH = 1K
  l1_pulpd     : ORIGIN = 0x50000000, LENGTH = 128K
  l1_spatz (rw)  : ORIGIN = 0x51000000, LENGTH = 128K
}

SECTIONS {
  /* Keep binaries lean */
  /DISCARD/ : { *(.riscv.attributes) *(.comment) }

  /* Global and stack pointer */
  /* By default, keep the calling context (boot ROM) stack pointer */
  __global_pointer$ = ADDR(.misc) + SIZEOF(.misc) / 2;
  __stack_pointer$  = 0;

  /* Further addresses */
  __base_dma      = 0x01000000;
  __base_bootrom  = 0x02000000;
  __base_clint    = 0x02040000;
  __base_axirt    = 0x020C0000;
  __base_axirtgrd = 0x020C1ffc;
  __base_regs     = 0x03000000;
  __base_llc      = 0x03001000;
  __base_uart     = 0x03002000;
  __base_i2c      = 0x03003000;
  __base_spih     = 0x03004000;
  __base_gpio     = 0x03005000;
  __base_slink    = 0x03006000;
  __base_vga      = 0x03007000;
  __base_usb      = 0x03008000;
  __base_bus_err  = 0x03009000;
  __base_plic     = 0x04000000;
  __base_clic     = 0x08000000;
  __base_spm      = ORIGIN(spm);
  __base_dram     = ORIGIN(dram);
  __base_l2       = ORIGIN(l2);

  __stack_pointer$  = ORIGIN(dram) + LENGTH(dram) - 8;

  .text : {
    *(.text._start)
    *(.text)
    *(.text.*)
  } > dram

  .misc : ALIGN(16) {
    *(.rodata)
    *(.rodata.*)
    *(.data)
    *(.data.*)
    *(.srodata)
    *(.srodata.*)
    *(.sdata)
    *(.sdata.*)
  } > dram

  . = ALIGN(32);
  .bss : ALIGN(16) {
    __bss_start = .;
    *(.bss)
    *(.bss.*)
    *(.sbss)
    *(.sbss.*)
    __bss_end = .;
  } > dram
  . = ALIGN(32);
  __bss_end = .;

  .bulk : ALIGN(16) {
    *(.bulk)
    *(.bulk.*)
  } > dram

  .l2_heap (NOLOAD) : ALIGN(32) {
    __l2_heap_start = .;
    /* . = ORIGIN(l2) + LENGTH(l2) - LENGTH(l2_common); */
    __l2_heap_end = .;
  } > l2

  .l2_common : ALIGN(32) {
    __l2_common_start = .;
    *(.l2_common)
    *(.l2_common.*)
    . = ALIGN(32);
    __l2_common_end = .;
  } > l2_common

  .offload : ALIGN(32) {
    *(.offload)
    *(.offload.*)
  } > dram
}