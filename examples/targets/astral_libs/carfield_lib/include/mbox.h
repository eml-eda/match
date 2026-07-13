#ifndef CAR_LIB_MBOX_H
#define CAR_LIB_MBOX_H

#include <stdint.h>

#define CAR_MBOX_BASE_ADDR  0x40000000
#define MBOX_CAR_INT_SND_STAT(id)   (CAR_MBOX_BASE_ADDR + 0x00 + (id * 0x100))
#define MBOX_CAR_INT_SND_SET(id)    (CAR_MBOX_BASE_ADDR + 0x04 + (id * 0x100))
#define MBOX_CAR_INT_SND_CLR(id)    (CAR_MBOX_BASE_ADDR + 0x08 + (id * 0x100))
#define MBOX_CAR_INT_SND_EN(id)     (CAR_MBOX_BASE_ADDR + 0x0C + (id * 0x100))
#define MBOX_CAR_INT_RCV_STAT(id)   (CAR_MBOX_BASE_ADDR + 0x40 + (id * 0x100))
#define MBOX_CAR_INT_RCV_SET(id)    (CAR_MBOX_BASE_ADDR + 0x44 + (id * 0x100))
#define MBOX_CAR_INT_RCV_CLR(id)    (CAR_MBOX_BASE_ADDR + 0x48 + (id * 0x100))
#define MBOX_CAR_INT_RCV_EN(id)     (CAR_MBOX_BASE_ADDR + 0x4C + (id * 0x100))
#define MBOX_CAR_LETTER0(id)        (CAR_MBOX_BASE_ADDR + 0x80 + (id * 0x100))
#define MBOX_CAR_LETTER1(id)        (CAR_MBOX_BASE_ADDR + 0x84 + (id * 0x100))


inline void mb_write(uint32_t val, uintptr_t addr) {
    asm volatile("sw %0, 0(%1)" : : "r"(val), "r"((volatile uint32_t*)addr) : "memory");
}

inline uint32_t mb_read(const uintptr_t addr)
{
    uint32_t val;
    asm volatile("lw %0, 0(%1)" : "=r"(val) : "r"((const volatile uint32_t*)addr) : "memory");
    return val;
}

inline void mailbox_send(uint32_t id, uint32_t letter0, uint32_t letter1) {
    mb_write(letter0, MBOX_CAR_LETTER0(id));
    mb_write(letter1, MBOX_CAR_LETTER1(id));
    mb_write(1, MBOX_CAR_INT_SND_SET(id));
    mb_write(1, MBOX_CAR_INT_SND_EN(id));
}

inline void mailbox_read(uint32_t id, volatile uint32_t* letter0, volatile uint32_t* letter1) {
    *letter0 = mb_read(MBOX_CAR_LETTER0(id));
    *letter1 = mb_read(MBOX_CAR_LETTER1(id));
}

inline void mailbox_clear(uint32_t id) {
    mb_write(0, MBOX_CAR_LETTER0(id));
    mb_write(0, MBOX_CAR_LETTER1(id));
    mb_write(1, MBOX_CAR_INT_SND_CLR(id));
    mb_write(0, MBOX_CAR_INT_SND_EN(id));
}


#define HOST_TO_CLUSTER_MBOX    6
#define CLUSTER_TO_HOST_MBOX    15

#define CLUSTER_MBOX_EVT        22
#define HOST_MBOX_IRQ           58


#endif