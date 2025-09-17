#include "carfield_lib/carfield.h"

#include <stdint.h>
#include <string.h>

// Carfield
#include "car_util.h"
#include "car_memory_map.h"
#include "regs/system_timer.h"
// Cheshire
#include "dif/clint.h"
#include "dif/dma.h"
#include "params.h"
#include "regs/cheshire.h"
#include "util.h"
// OpenTitan Peripherals (PLIC)
#include "sw/device/lib/dif/dif_rv_plic.h" 

// Carfield Lib
#include "carfield_lib/uart.h"
#include "carfield_lib/mbox.h"

// TODO - make it general
#include "model_graph.h"


#define VERIFY_DMA 0


void carfield_init() {
    // Initialize the Carfield SoC
    car_enable_domain(CAR_PULP_RST);
    car_enable_domain(CAR_SPATZ_RST);
    // Initialize the UART
    carfield_init_uart();
    // If mailboxes are used initialize the PLIC
    carfield_init_plic();
    // Start system timer
    carfield_timer_start();

    mini_printf("Hi, there. I'm Carfield 🐱\r\n\n");
}

void carfield_shutdown() {
    carfield_timer_stop();
    car_disable_domain(CAR_PULP_RST);
    mini_printf("\r\nBye.\r\n");
}

void carfield_wait_eoc() {
    asm volatile("wfi":::"memory");
    asm volatile("fence rw,rw":::"memory"); // important
}

/* PULP CLUSTER */

void pulp_cluster_reset() {
    volatile uint32_t *booten_addr = (uint32_t*)(CAR_INT_CLUSTER_BOOTEN_ADDR(car_soc_ctrl));
    writew(0, booten_addr);
    volatile uint32_t *fetchen_addr = (uint32_t*)(CAR_INT_CLUSTER_FETCHEN_ADDR(car_soc_ctrl));
    writew(0, fetchen_addr);
    volatile uint32_t *pulp_eoc_addr = (uint32_t*)(CAR_INT_CLUSTER_EOC_ADDR(car_soc_ctrl));
    writew(0, pulp_eoc_addr);
    pulp_cluster_set_bootaddress(0);
    car_reset_domain(CAR_PULP_RST);
}


void pulp_cluster_offload_async(void* boot_addr)
{
    mini_printf("Starting PULP cluster...\r\n");
    pulp_cluster_reset();
    pulp_cluster_set_bootaddress(boot_addr);
    pulp_cluster_start();
    //mini_printf("> Started PULP cluster. Waiting...\r\n");
}


void pulp_cluster_offload_blk(void* boot_addr)
{
    pulp_cluster_offload_async(boot_addr);
    pulp_cluster_wait_eoc();
    mini_printf("> Cluster finished.\r\n");
}


void pulp_cluster_send_task_poll(volatile uint32_t* args, uint32_t task_id) {
    asm volatile("fence rw,rw":::"memory");
    args[0] = task_id;
    asm volatile("fence rw,rw":::"memory");
}


int pulp_cluster_wait_end_of_task_poll(volatile uint32_t* args, uint32_t task_id) {
    while (args[0] != 0xFFFFFFF0) {
        asm volatile("fence r,rw" ::: "memory");
    }
    return 0;
}


void pulp_cluster_send_task_mbox(volatile uint32_t* args, uint32_t task_id) {
    asm volatile("fence rw,rw" ::: "memory");
    mailbox_send(HOST_TO_CLUSTER_MBOX, args+1, task_id);
    asm volatile("fence rw,rw" ::: "memory");
}

volatile uint32_t last_completed_node_id = 0xFFFFFFFF;
volatile uint32_t last_task_error_code = 0;

int pulp_cluster_wait_end_of_task_mbox(volatile uint32_t* args, uint32_t task_id) {
    while (last_completed_node_id != task_id) {
        asm volatile("fence rw,rw" ::: "memory");
        asm volatile("wfi":::"memory");
        asm volatile("fence rw,rw" ::: "memory");
        //mini_printf("In effetti qua ci siamo...\r\n");
    }
    return last_task_error_code;
}

/* SPATZ */


void spatz_reset() {
    writew(0, CAR_MBOX_BASE_ADDR +  MBOX_INT_SND_SET_OFFSET + 0*0x100);
    writew(0, CAR_MBOX_BASE_ADDR +  MBOX_INT_SND_SET_OFFSET + 1*0x100);

    writew(0, CAR_MBOX_BASE_ADDR +  MBOX_INT_SND_EN_OFFSET + 0*0x100);
    writew(0, CAR_MBOX_BASE_ADDR +  MBOX_INT_SND_EN_OFFSET + 1*0x100);
    
    writew(0, CAR_FP_CLUSTER_PERIPHERAL_CLUSTER_BOOT_CONTROL_REG_ADDR(car_spatz_cluster));

    car_reset_domain(CAR_SPATZ_RST);
}


void spatz_offload_async(void* boot_addr)
{
    mini_printf("Starting SPATZ cluster...\r\n");
    spatz_reset();

    writew(boot_addr, CAR_FP_CLUSTER_PERIPHERAL_CLUSTER_BOOT_CONTROL_REG_ADDR(car_spatz_cluster));
    
    writew(1, CAR_MBOX_BASE_ADDR + MBOX_INT_SND_SET_OFFSET + 0*0x100);
    writew(1, CAR_MBOX_BASE_ADDR + MBOX_INT_SND_SET_OFFSET + 1*0x100);

    writew(1, CAR_MBOX_BASE_ADDR + MBOX_INT_SND_EN_OFFSET + 0*0x100);
    writew(1, CAR_MBOX_BASE_ADDR + MBOX_INT_SND_EN_OFFSET + 1*0x100);
    mini_printf("SPATZ started...\r\n");
    for (volatile int i = 0; i < 10000; i++) {
        asm volatile("fence rw,rw":::"memory");
    }
}


void spatz_offload_blk(void* boot_addr)
{
    spatz_offload_async(boot_addr);
    
    volatile uint32_t spatzd_corestatus;
    volatile uintptr_t *spatzd_corestatus_addr = (uintptr_t*)CAR_FP_CLUSTER_PERIPHERAL_CORESTATUS_REG_ADDR(car_spatz_cluster);

    while (!(uint32_t)readw(spatzd_corestatus_addr)) {
    	asm volatile("fence rw,rw":::"memory");
    }
    spatzd_corestatus = (uint32_t)readw(spatzd_corestatus_addr);

    mini_printf("> Spatz finished.\r\n");
}


void spatz_send_task_poll(volatile uint32_t* args, uint32_t task_id) {
    asm volatile("fence rw,rw":::"memory");
    args[0] = task_id;
    asm volatile("fence rw,rw":::"memory");
}

int spatz_wait_end_of_task_poll(volatile uint32_t* args, uint32_t task_id) {
    while (args[0] != 0xFFFFFFF0) {
        asm volatile("fence r,rw" ::: "memory");
    }
    return 0;
}


void spatz_send_task_mbox(volatile uint32_t* args, uint32_t task_id) {
    asm volatile("fence rw,rw" ::: "memory");
    mailbox_send(HOST_TO_SPATZ_C0_MBOX, args+1, task_id);
    asm volatile("fence rw,rw" ::: "memory");
}

int spatz_wait_end_of_task_mbox(volatile uint32_t* args, uint32_t task_id) {
    while (last_completed_node_id != task_id) {
        asm volatile("fence rw,rw" ::: "memory");
        asm volatile("wfi":::"memory");
        asm volatile("fence rw,rw" ::: "memory");
    }
    return last_task_error_code;
}


// Host interrupt related things


static dif_rv_plic_t plic0;


void carfield_init_plic() {
    // Reset PLIC
    dif_rv_plic_reset(&plic0);
    
    // Set global interrupt enable in CVA6 csr
    unsigned long mstatus;
    asm volatile ("csrr %0, mstatus" : "=r"(mstatus));
    mstatus |= GLOBAL_IRQ_ENABLE;
    asm volatile ("csrw mstatus, %0" :: "r"(mstatus));

    // Set external interrupt enable in CVA6 csr
    unsigned long mie;
    asm volatile ("csrr %0, mie" : "=r"(mie));
    mie |= EXTERNAL_IRQ_ENABLE;
    asm volatile ("csrw mie, %0" :: "r"(mie));

    // Setup PLIC
    mmio_region_t plic_base_addr = mmio_region_from_addr(PLIC_BASE_ADDRESS);
    dif_result_t t = dif_rv_plic_init(plic_base_addr, &plic0);

    // Enable CLUSTER_TO_HOST_MBOX interrupt
    const int irq = HOST_MBOX_IRQ;
    const int priority = 0x1;
    t = dif_rv_plic_irq_set_priority(&plic0, irq, priority);
    t = dif_rv_plic_irq_set_enabled(&plic0, irq, 0, kDifToggleEnabled);
    if (t != kDifOk) {
        mini_printf("Error setting PLIC IRQ %d\r\n", irq);
    }
}


void handle_interrupt_pulp_cluster_mbox() {
    mailbox_read(CLUSTER_TO_HOST_MBOX, &last_completed_node_id, &last_task_error_code);
    mailbox_clear(CLUSTER_TO_HOST_MBOX);
    //mini_printf("Hey IIiIIIIIIIInterrupt!\r\n");
}


void trap_vector(void) {
    dif_rv_plic_irq_id_t claim_irq;
    dif_rv_plic_irq_claim(&plic0, 0, &claim_irq);
    if (claim_irq == HOST_MBOX_IRQ) {
        handle_interrupt_pulp_cluster_mbox();
        match_model_runtime_eoc_callback(last_completed_node_id); // TODO make it general
        dif_rv_plic_irq_complete(&plic0, 0, claim_irq);
    }
}


// Other things

void handle_host_dma_transfer(void* src, void* dst, size_t size) 
{
    mini_printf("Starting DMA transfer...\r\n");
    sys_dma_2d_blk_memcpy(dst, src, size, 0, 0, 1);

    mini_printf("Transfer complete.\r\n");

    #if VERIFY_DMA
    // Verify
    volatile uint8_t* src_ptr = (uint8_t*)src;
    volatile uint8_t* dst_ptr = (uint8_t*)dst;
    bool transfer_success = true;
    for (int i = 0; i < size; i++) {
        volatile uint8_t sval = src_ptr[i];
        volatile uint8_t dval = dst_ptr[i];
        if (sval != dval) {
            mini_printf("DMA transfer failed at byte %d\r\n", i);
            mini_printf("src_ptr %p -> %d\r\n", src_ptr + i, sval);
            mini_printf("dst_ptr %p -> %d\r\n", dst_ptr + i, dval);
            transfer_success = false;
            break;
        }
    }
    if (transfer_success)  {
        mini_printf("Transfer Verified Successfully.\r\n");
    }
    #endif
}


void carfield_init_uart() {
    uint32_t rtc_freq = *reg32(&__base_regs, CHESHIRE_RTC_FREQ_REG_OFFSET);
    uint64_t reset_freq = clint_get_core_freq(rtc_freq, 2500);
    car_uart_init(&__base_uart, reset_freq, 115200);
}


void carfield_timer_start() {
    writed(1, CAR_SYSTEM_TIMER_BASE_ADDR + TIMER_RESET_LO_OFFSET);
    *(volatile uint32_t*)(CAR_SYSTEM_TIMER_BASE_ADDR + TIMER_CFG_LO_OFFSET) |= (1 << TIMER_CFG_LO_CCFG_BIT);
    writed(1, CAR_SYSTEM_TIMER_BASE_ADDR + TIMER_START_LO_OFFSET);
}
    
void carfield_timer_stop() {
    writed(0, CAR_SYSTEM_TIMER_BASE_ADDR + TIMER_CFG_LO_OFFSET);
}

uint64_t carfield_timer_read() {
    asm volatile("" ::: "memory");
    volatile uint64_t counter = readd(CAR_SYSTEM_TIMER_BASE_ADDR + TIMER_CNT_LO_OFFSET);
    asm volatile("" ::: "memory");
    return counter;
}

float carfield_timer_to_ms_factor() {
    uint32_t rtc_freq = *reg32(&__base_regs, CHESHIRE_RTC_FREQ_REG_OFFSET);
    return 1000.0f / (float)(rtc_freq);
}

// External memory management

void* carfield_init_ram(size_t size) {
    // TODO
    return NULL;
}

void carfield_load_file_to_ram(const char* file_name, void* dst, size_t size) {
    // TODO
}

void carfield_memcpy_from_ram(void* loc, const void* ext, size_t size) {
    handle_host_dma_transfer(ext, loc, size);
}

void carfield_memcpy_to_ram(const void* loc, void* ext, size_t size) {
    handle_host_dma_transfer(loc, ext, size);
}

void carfield_free_ram(void* ext, size_t size) {
    // TODO
}