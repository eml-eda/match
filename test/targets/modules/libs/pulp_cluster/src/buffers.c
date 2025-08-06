#include <pulp_cluster/buffers.h>

static void* im2col_pt_ = 0x0;
static void* pwt_pt_ = 0x0;
static void* bt_buffer_pt_ = 0x0;
#ifndef GAP_SDK
static void* pulp_open_l1_pt_ = 0x0;
#endif

void set_im2col_pt(void* im2col_pt){
    im2col_pt_ = im2col_pt;
}

void set_pwt_pt(void* pwt_pt){
    pwt_pt_ = pwt_pt;
}

void set_bt_buffer_pt(void* bt_buffer_pt){
    bt_buffer_pt_ = bt_buffer_pt;
}

void* get_im2col_pt(){
    return im2col_pt_;
}

void* get_pwt_pt(){
    return pwt_pt_;
}

void* get_bt_buffer_pt(){
    return bt_buffer_pt_;
}

#ifndef GAP_SDK
void set_pulp_open_l1_pt(void* pulp_open_l1_pt){
    pulp_open_l1_pt_ = pulp_open_l1_pt;
}

void* get_pulp_open_l1_pt(){
    return pulp_open_l1_pt_;
}
#endif

void free_pulp_device_buffers(){
    im2col_pt_ = 0x0;
    pwt_pt_ = 0x0;
    bt_buffer_pt_ = 0x0;
#ifndef GAP_SDK
    pulp_open_l1_pt_ = 0x0;
#endif
}