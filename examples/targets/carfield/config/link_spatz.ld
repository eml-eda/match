/* Copyright 2020 ETH Zurich and University of Bologna. */
/* Solderpad Hardware License, Version 0.51, see LICENSE for details. */
/* SPDX-License-Identifier: SHL-0.51 */

OUTPUT_ARCH( "riscv" )
ENTRY(_start)

MEMORY
{
    l2           : ORIGIN = 0x78200000 + 768K + 64 + 128K, LENGTH =  1M - 768K - 64 - 128K
    l2_common    : ORIGIN = 0x78200000 + 768K - 256, LENGTH = 256
    l1_spatz     : ORIGIN = 0x51000000, LENGTH = 128K
}

SECTIONS
{  
  /* The program code and other data goes into l2 */
  .text :
  {
    . = ALIGN(4);
    *(.init)
    *(.text.startup)
    *(.text)
    *(.text*)
    . = ALIGN(4);
    _etext = .;
  } >l2

  .rodata :
  {
    . = ALIGN(4);
    *(.rodata)
    *(.rodata*)
    . = ALIGN(4);
  } >l2

  /* HTIF */
  .htif         : { } >l2

  /* Thread Local Storage sections  */
  .tdata    :
  {
    __tdata_start = .;
    *(.tdata .tdata.* .gnu.linkonce.td.*)
    __tdata_end = .;
  } >l2

  .tbss :
  {
    __tbss_start = .;
    *(.tbss .tbss.* .gnu.linkonce.tb.*)
    *(.tcommon)
    __tbss_end = .;
  } >l2

  /* add a section after .tbss to put the __tbss_end symbol into for
     the LLD linker */
  .tbssend : { __tbss_end2 = .; } >l2

  /* used by the startup to initialize data */
  _sidata = LOADADDR(.data);

  /* small data section that can be addressed through the global pointer */
  .sdata          :
  {
    __SDATA_BEGIN__ = .;
    __global_pointer$ = . + 0x7f0;
    *(.srodata.cst16) *(.srodata.cst8) *(.srodata.cst4) *(.srodata.cst2) *(.srodata .srodata.*)
    *(.sdata .sdata.* .gnu.linkonce.s.*)
  } >l2

  /* Initialized data sections goes into l2 */
  .data :
  {
    __DATA_BEGIN__ = .;
    *(.data .data.* .gnu.linkonce.d.*)
    SORT(CONSTRUCTORS)
  } >l2
  _edata = .; PROVIDE (edata = .);

  /* small bss section */
  . = .;
  __bss_start = .;
  .sbss           :
  {
    *(.dynsbss)
    *(.sbss .sbss.* .gnu.linkonce.sb.*)
    *(.scommon)
  } >l2

  /* Uninitialized data section */
  .bss            :
  {
   *(.dynbss)
   *(.bss .bss.* .gnu.linkonce.b.*)
   *(COMMON)
   /* Align here to ensure that the .bss section occupies space up to
      _end.  Align after .bss to ensure correct alignment even if the
      .bss section disappears because there are no input sections. */
   . = ALIGN(. != 0 ? 32 / 8 : 1);
  } >l2
  . = ALIGN(32 / 8);
  . = SEGMENT_START("ldata-segment", .);
  . = ALIGN(32 / 8);
  __BSS_END__ = .;
  _end = .; PROVIDE (end = .);

  /* Uninitialized data section in l2 */
  .l2 :
  {
    *(.l2)
    _edram = .;
  } >l2
  
  .stack : {
  . = ALIGN(4);
    . = ALIGN(16);
    stack_start = .;
    . = . + 0x800;
    stack = .;
  } > l2


  __base_uart     = 0x03002000;
}
