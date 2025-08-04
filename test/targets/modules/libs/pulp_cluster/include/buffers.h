#ifndef __PULP_CLUSTER_BUFFERS_H__
#define __PULP_CLUSTER_BUFFERS_H__

void set_im2col_pt(void* im2col_pt);
void set_pwt_pt(void* pwt_pt);
void set_bt_buffer_pt(void* bt_buffer_pt);
void* get_im2col_pt();
void* get_pwt_pt();
void* get_bt_buffer_pt();
#ifndef GAP_SDK
void set_pulp_open_l1_pt(void* pulp_open_l1_pt);
void* get_pulp_open_l1_pt();
#endif

void free_pulp_device_buffers();

#endif // __PULP_CLUSTER_BUFFERS_H__