ENTRY(_start)

/* We assume at least 64 KiB SPM, same minus stack for ROMs. */
/* If more SPM is available, CRT0 repoints the stack. */
/* We  assume at least 8 MiB of DRAM (minimum for Linux). */

MEMORY {
  bootrom (rx)   : ORIGIN = 0x02000000, LENGTH = 16K
  extrom (rx)    : ORIGIN = 0x00000000, LENGTH = 48K
  spm (rwx)      : ORIGIN = 0x10000000, LENGTH = 64K
  dram (rwx)     : ORIGIN = 0x80010000, LENGTH = 64M
  l2 (rwx)       : ORIGIN = 0x78000000, LENGTH = 768K - 256 - 64
  l2_common (rw) : ORIGIN = 0x78000000 + 768K - 256, LENGTH = 256
  l1_pulpd (rw)  : ORIGIN = 0x50000000, LENGTH = 256K 
  l1_spatz (rw)  : ORIGIN = 0x51000000, LENGTH = 128K
}

SECTIONS {
  /* Keep binaries lean */
  /* /DISCARD/ : { *(.riscv.attributes) *(.comment) } */

  /* Global and stack pointer */
  /* By default, keep the calling context (boot ROM) stack pointer */
  __global_pointer$ = ADDR(.misc) + SIZEOF(.misc) / 2;

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
  __base_plic     = 0x04000000;
  __base_spm      = ORIGIN(spm);
  __base_dram     = ORIGIN(dram);
  __base_l2       = ORIGIN(l2);

  __stack_pointer$  = ORIGIN(l2) + LENGTH(l2) - 8;

  .text : {
    *(.text._start)
    *(.text)
    *(.text.*)
  } > l2

  .misc : ALIGN(16) {
    *(.rodata)
    *(.rodata.*)
    *(.data)
    *(.data.*)
    *(.srodata)
    *(.srodata.*)
    *(.sdata)
    *(.sdata.*)
  } > l2

  . = ALIGN(32);
  __bss_start = .;
  .bss : {
    *(.bss)
    *(.bss.*)
    *(.sbss)
    *(.sbss.*)
  } > l2
  . = ALIGN(32);
  __bss_end = .;

  .bulk : ALIGN(16) {
    *(.bulk)
    *(.bulk.*)
  } > l2

  .l2_heap (NOLOAD) : ALIGN(32) {
    __l2_heap_start = .;
    . = ORIGIN(l2) + LENGTH(l2) - LENGTH(l2_common);
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
