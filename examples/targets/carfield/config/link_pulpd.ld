
OUTPUT_ARCH(riscv)
ENTRY( _start )

/* L2 uses port 2 instead of port 1 used by host */
MEMORY
{
  l2           : ORIGIN = 0x78200000 + 768K + 64, LENGTH = 1M - 768K - 64
  l2_common    : ORIGIN = 0x78200000 + 768K - 256, LENGTH = 256
  l1_pulpd     : ORIGIN = 0x50000000, LENGTH = 256K
}

/*
 * This linker script try to put FC data in l2 private bank0 and FC code
 * in l2 private bank1 to avoid contention between FC code and data
 * as FC has no instruction cache and is so often accessing l2 to
 * get instructions. Everything can be shifted in case one bank is full.
 *
 * Cluster code and initialized data are put in shared banks to not polute
 * private banks which are quite small, and also avoid contentions between
 * cluster cache refill and FC.
 */


SECTIONS
{
  __base_uart     = 0x03002000;
  /*
   * l2 PRIVATE BANK0
   *
   * Contains FC data
   */

  .init :
  {
    . = ALIGN(4);
    KEEP( *(.init) )
  } > l2


  .fini :
  {
    . = ALIGN(4);
    KEEP( *(.fini) )
  } > l2


  .preinit_array : {
    . = ALIGN(4);
    PROVIDE_HIDDEN (__preinit_array_start = .);
    KEEP (*(.preinit_array))
    PROVIDE_HIDDEN (__preinit_array_end = .);
  } > l2


  .init_array : {
    . = ALIGN(4);
    PROVIDE_HIDDEN (__init_array_start = .);
    __CTOR_LIST__ = .;
    LONG((__CTOR_END__ - __CTOR_LIST__) / 4 - 2)
    KEEP(*(.ctors.start))
    KEEP(*(.ctors))
    KEEP (*(SORT(.init_array.*)))
    KEEP (*(.init_array ))
    LONG(0)
    __CTOR_END__ = .;
    PROVIDE_HIDDEN (__init_array_end = .);
  } > l2


  .fini_array : {
    . = ALIGN(4);
    PROVIDE_HIDDEN (__fini_array_start = .);
    __DTOR_LIST__ = .;
    LONG((__DTOR_END__ - __DTOR_LIST__) / 4 - 2)
    KEEP(*(.dtors.start))
    KEEP(*(.dtors))
    LONG(0)
    __DTOR_END__ = .;
    KEEP (*(SORT(.fini_array.*)))
    KEEP (*(.fini_array ))
    PROVIDE_HIDDEN (__fini_array_end = .);
  } > l2


  .boot : {
    . = ALIGN(4);
    *(.boot)
    *(.boot.data)
  } > l2


  .rodata : {
    . = ALIGN(4);
    *(.rodata);
    *(.rodata.*)
    *(.srodata);
    *(.srodata.*)
    *(.eh_frame*)
  } > l2


  .got : {
    . = ALIGN(4);
    *(.got.plt) * (.igot.plt) *(.got) *(.igot)
  } > l2


  .shbss : {
    . = ALIGN(4);
    *(.shbss)
  } > l2


  .talias : {
  } > l2


  .gnu.offload_funcs : {
    . = ALIGN(4);
    KEEP(*(.gnu.offload_funcs))
  } > l2


  .gnu.offload_vars : {
    . = ALIGN(4);
    KEEP(*(.gnu.offload_vars))
  } > l2


  .stack : {
    . = ALIGN(4);
    . = ALIGN(16);
    stack_start = .;
    . = . + 0x800;
    stack = .;
  } > l2


  .data : {
    . = ALIGN(4);
    sdata  =  .;
    _sdata  =  .;
    *(.data_fc)
    *(.data_fc.*)
    *(.data);
    *(.data.*)
    *(.sdata);
    *(.sdata.*)
    *(.heapl2ram)
    *(.fcTcdm)
    *(.fcTcdm.*)
    *(.fcTcdm_g)
    *(.fcTcdm_g.*)
    . = ALIGN(4);
    edata  =  .;
    _edata  =  .;
  } > l2


  .bss : {
    . = ALIGN(8);
    _bss_start = .;
    *(.bss)
    *(.bss.*)
    *(.sbss)
    *(.sbss.*)
    *(COMMON)
    . = ALIGN(4);
    _bss_end = .;
  } > l2


  __l2_priv0_end = ALIGN(4);




  /*
   * l2 PRIVATE BANK1
   *
   * Contains FC code
   */

  .vectors MAX(ORIGIN(l2)+0x8000,ALIGN(256)) :
  {
    /*. = ALIGN(256);*/
    __irq_vector_base = .;
    KEEP(*(.vectors))
  } > l2

  .text :
  {
    . = ALIGN(4);
    _stext = .;
    *(.text)
    *(.text.*)
    _etext  =  .;
    *(.lit)
    *(.shdata)
    _endtext = .;
    . = ALIGN(4);
  } > l2

  __l2_priv1_end = ALIGN(4);


  /*
   * l2 SHARED BANKS
   *
   * Contains other data such as peripheral data and cluster code and data
   */

  .l2_data MAX(ORIGIN(l2)+0x10000,ALIGN(4)) :
  {
    . = ALIGN(4);
    __cluster_text_start = .;
    *(.cluster.text)
    *(.cluster.text.*)
    . = ALIGN(4);
    __cluster_text_end = .;
    *(.l2_data)
    *(.l2_data.*)
    *(.data_fc_shared)
    *(.data_fc_shared.*)
    . = ALIGN(4);
  } > l2

  __l2_data_end = .;


  /* Shared with host */
  .l2_common (NOLOAD) : ALIGN(32) {
    __l2_common_start = .;
    *(.l2_common)
    *(.l2_common.*)
    . = ALIGN(32);
    __l2_common_end = .;
  } > l2_common

  .l1cluster_g : {
    . = ALIGN(4);
    *(.heapsram)
    *(.heapsram.*)
    *(.l1cluster_g)
    *(.l1cluster_g.*)
    *(.data_l1)
    *(.data_l1.*)
    . = ALIGN(4);
    _libgomp_start = .;
    *(.libgomp)
    *(.libgomp.*)
    . = ALIGN(4);
  } > l1_pulpd

  .bss_l1 : {
    . = ALIGN(4);
    *(.bss_l1)
    *(.bss_l1.*)
    . = ALIGN(4);
  } > l1_pulpd

  __l1_end = ALIGN(4);

  __l2_shared_end = __l2_data_end + SIZEOF(.l1cluster_g) + SIZEOF(.bss_l1);




  __cluster_text_size = __cluster_text_end - __cluster_text_start;

  __l1_heap_start = ALIGN(4);
  __l1_heap_size = LENGTH(l1_pulpd) - __l1_heap_start + ORIGIN(l1_pulpd);
}
