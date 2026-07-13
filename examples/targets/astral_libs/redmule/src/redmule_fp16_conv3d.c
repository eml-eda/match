#ifdef __pulp_cluster__

#include <stdint.h>

#include "pulp.h"

#include "redmule/redmule_arch.h"
#include "redmule/redmule_hal.h"
#include "redmule/redmule_kernels.h"
#include "redmule/redmule_defines.h"

#define __PROFILE_REDMULE__ 0
#define __CONV_FILL_BIAS_WITH_IM2COL__ 1

void redmule_fp16_conv3d_rd(
    const fp16  *input,       // Input 3D tensor (Shape: C_in x D x H x W)
    const fp16  *weights,     // Pre-flattened Weight Matrix (Shape: N x K)
    fp16  *output_matrix,     // Destination Matrix for GEMM result (Shape: M x K)
    fp16  *im2col_buf,        // Allocated L1 TCDM buffer for vol2col (Shape: M x N)
    const fp16  *bias,              // Bias vector (Shape: K)
    int apply_relu,
    int c_in, int d_in, int h_in, int w_in,
    int c_out, int d_out, int h_out, int w_out,
    int kd, int kh, int kw,
    int stride_d, int stride_h, int stride_w,
    int pad_d_low, int pad_d_high,   // Depth padding (front, back)
    int pad_h_low, int pad_h_high,   // Height padding (top, bottom)
    int pad_w_low, int pad_w_high    // Width padding (left, right)
) {
    uint16_t m_size = d_out * h_out * w_out;     // M: Rows of X (Total Output Pixels)
    uint16_t n_size = c_in * kd * kh * kw;       // N: Columns of X / Rows of W (Patch Size)
    uint16_t k_size = c_out;                     // K: Columns of W / Columns of Output

    int chunk_size = (m_size + nthreads - 1) / nthreads;
    int start_m = tid * chunk_size;
    int end_m = (start_m + chunk_size > m_size) ? m_size : (start_m + chunk_size);

    for (int m = start_m; m < end_m; m++) {
        for (int k = 0; k < c_out; k++) {
            output_matrix[m * c_out + k] = (bias != NULL) ? bias[k] : (fp16)0.0f;
        }

        // Decode linear matrix row index back into 3D spatial output coordinates
        int w_out_idx = m % w_out;
        int h_out_idx = (m / w_out) % h_out;
        int d_out_idx = m / (w_out * h_out);

        int out_col = 0;

        // Unroll the 3D patch for this specific output pixel location
        for (int c = 0; c < c_in; c++) {
            for (int kz = 0; kz < kd; kz++) {
                // Apply low-side padding offset to map back to the raw input tensor space
                int in_d = d_out_idx * stride_d - pad_d_low + kz;

                for (int ky = 0; ky < kh; ky++) {
                    int in_h = h_out_idx * stride_h - pad_h_low + ky;

                    for (int kx = 0; kx < kw; kx++) {
                        int in_w = w_out_idx * stride_w - pad_w_low + kx;

                        // Check if the current spatial coordinate falls inside the padded bounds
                        if (in_d >= 0 && in_d < d_in &&
                            in_h >= 0 && in_h < h_in &&
                            in_w >= 0 && in_w < w_in) {

                            // Valid input space pixel: calculate linear index (CDHW layout)
                            int input_idx = ((c * d_in + in_d) * h_in + in_h) * w_in + in_w;
                            im2col_buf[m * n_size + out_col] = input[input_idx];
                        } else {
                            // Padded space pixel: fill with a structural zero
                            im2col_buf[m * n_size + out_col] = (fp16)0.0f;
                        }
                        out_col++;
                    }
                }
            }
        }
    }

    synch_barrier();

    redmule_fp16_gemm(
        im2col_buf,
        weights,
        output_matrix,
        m_size, n_size, k_size
    );

    synch_barrier();

    if (apply_relu) {
        // Cores process their assigned output matrix rows in-place within L1 TCDM
        for (int m = start_m; m < end_m; m++) {
            for (int k = 0; k < c_out; k++) {
                int idx = m * c_out + k;
                if (output_matrix[idx] < (fp16)0.0f) {
                    output_matrix[idx] = (fp16)0.0f;
                }
            }
        }
        // Final sync ensures the completed, activated layer data is ready for subsequent layers
        synch_barrier();
    }
}

void redmule_fp16_gemm_software_reference(
    const fp16 *matrix_x,
    const fp16 *matrix_w,
    fp16 *matrix_z,
    int M, int N, int K
) {
    // Loop over the rows of Matrix X (and Matrix Z)
    for (int m = 0; m < M; m++) {
        // Loop over the columns of Matrix W (and Matrix Z)
        for (int k = 0; k < K; k++) {
            
            // Use 32-bit float for accumulation to preserve accuracy 
            // and prevent intermediate underflow/overflow during long dot products
            fp16 acc = 0.0f;

            // Dot product of row 'm' from X and column 'k' from W
            for (int n = 0; n < N; n++) {
                // Matrix X: Row-major indexing -> (row * total_columns) + col
                int idx_x = m * N + n;
                
                // Matrix W: Row-major indexing -> (row * total_columns) + col
                int idx_w = n * K + k;

                acc += matrix_x[idx_x] * matrix_w[idx_w];
            }

            // Write back the final accumulated result to Matrix Z
            int idx_z = m * K + k;
            matrix_z[idx_z] = acc;
        }
    }
}

void redmule_fp16_conv3d_dhwn_rd(
    const fp16  *input,       // Input 3D tensor (Shape: D x H x W x C_in)
    const fp16  *weights,     // Pre-flattened Weight Matrix (Shape: K x N) -> [C_out x (Dk*Hk*Wk*C_in)]
    fp16  *output_matrix,     // Destination Matrix for GEMM result (Shape: M x K) -> [M x C_out]
    fp16  *im2col_buf,        // Allocated L1 TCDM buffer for vol2col (Shape: M x N) -> [M x (Dk*Hk*Wk*C_in)]
    const fp16  *bias,        // Bias vector (Shape: K) -> [C_out]
    int apply_relu,
    int c_in, int d_in, int h_in, int w_in,
    int c_out, int d_out, int h_out, int w_out,
    int kd, int kh, int kw,
    int stride_d, int stride_h, int stride_w,
    int pad_d_low, int pad_d_high,   // Depth padding (front, back)
    int pad_h_low, int pad_h_high,   // Height padding (top, bottom)
    int pad_w_low, int pad_w_high    // Width padding (left, right)
) {
    #ifdef __PROFILE_REDMULE__
    if(!tid)    cluster_timer_start();
    #endif
    uint16_t m_size = d_out * h_out * w_out;     // M: Rows of X (Total Output Pixels)
    uint16_t n_size = kd * kh * kw * c_in;       // N: Columns of X / Rows of W (Flattened Patch Size)
    uint16_t k_size = c_out;                     // K: Columns of W / Columns of Output

    int chunk_size = (m_size + nthreads - 1) / nthreads;
    int start_m = tid * chunk_size;
    int end_m = (start_m + chunk_size > m_size) ? m_size : (start_m + chunk_size);

    for (int m = start_m; m < end_m; m++) {
        #if __CONV_FILL_BIAS_WITH_IM2COL__
        for (int k = 0; k < c_out; k++) {
            output_matrix[m * c_out + k] = (bias != NULL) ? bias[k] : (fp16)0.0f;
        }
        #endif

        // Decode linear matrix row index back into 3D spatial output coordinates
        int w_out_idx = m % w_out;
        int h_out_idx = (m / w_out) % h_out;
        int d_out_idx = m / (w_out * h_out);

        int out_col = 0;

        // CRITICAL FIX: Loop over Spatial Kernel (kz, ky, kx) FIRST, and C_in LAST 
        // to match the DHWC_in memory layout.
        for (int kz = 0; kz < kd; kz++) {
            int in_d = d_out_idx * stride_d - pad_d_low + kz;

            for (int ky = 0; ky < kh; ky++) {
                int in_h = h_out_idx * stride_h - pad_h_low + ky;

                for (int kx = 0; kx < kw; kx++) {
                    int in_w = w_out_idx * stride_w - pad_w_low + kx;

                    // Check if the spatial coordinate falls inside the bounds
                    if (in_d >= 0 && in_d < d_in &&
                        in_h >= 0 && in_h < h_in &&
                        in_w >= 0 && in_w < w_in) {

                        // Calculate the base index for the start of the C_in channel block
                        // Layout: D -> H -> W -> C_in
                        int base_input_idx = ((in_d * h_in + in_h) * w_in + in_w) * c_in;
                        
                        // Vectorized-friendly copy of all continuous C_in channels
                        for (int c = 0; c < c_in; c++) {
                            im2col_buf[m * n_size + out_col] = input[base_input_idx + c];
                            out_col++;
                        }
                    } else {
                        // Padded space: fill the entire C_in block for this spatial position with zeros
                        for (int c = 0; c < c_in; c++) {
                            im2col_buf[m * n_size + out_col] = (fp16)0.0f;
                            out_col++;
                        }
                    }
                }
            }
        }
    }

    // Synchronize cluster cores before calling RedMulE hardware
    synch_barrier();
    
    #ifdef __PROFILE_REDMULE__
    if(!tid){
        mini_printf("[REDMULE CONV3D] Im2col filling took %d cycles\r\n", cluster_timer_stop());
        cluster_timer_start();
    }
    #endif

    // Core 0 (or all cores depending on your runtime driver) triggers RedMulE
    if (tid == 0) {
        redmule_fp16_gemm(
            im2col_buf,
            weights,
            output_matrix,
            m_size, n_size, k_size
        );
        #ifdef __PROFILE_REDMULE__
        mini_printf("[REDMULE CONV3D] GEMM Op took %d cycles\r\n", cluster_timer_stop());
        #endif
    }

    // Sync to wait for hardware accelerator completion
    synch_barrier();

    #if __CONV_FILL_BIAS_WITH_IM2COL__
    if (apply_relu) {
    #endif
        #ifdef __PROFILE_REDMULE__
        if(!tid)    cluster_timer_start();
        #endif
        // Cores process their assigned output matrix rows in-place within L1 TCDM
        for (int m = start_m; m < end_m; m++) {
            for (int k = 0; k < c_out; k++) {
                int idx = m * c_out + k;
                #if !__CONV_FILL_BIAS_WITH_IM2COL__
                output_matrix[idx] += (bias != NULL) ? bias[k] : (fp16)0.0f;
                #endif
                if (
                    #if !__CONV_FILL_BIAS_WITH_IM2COL__
                    apply_relu &&
                    #endif
                     output_matrix[idx] < (fp16)0.0f) {
                    output_matrix[idx] = (fp16)0.0f;
                }
            }
        }
        synch_barrier();
        #ifdef __PROFILE_REDMULE__
        if(!tid)    mini_printf("[REDMULE CONV3D] Relu Op took %d cycles\r\n", cluster_timer_stop());
        #endif
    #if __CONV_FILL_BIAS_WITH_IM2COL__
    }
    #endif
}

void redmule_fp16_conv3d_pc(
    const fp16  *input,       // Input 3D tensor (Shape: C_in x D x H x W)
    const fp16  *weights,     // Pre-flattened Weight Matrix (Shape: N x K)
    fp16  *output_matrix,     // Destination Matrix for GEMM result (Shape: M x K)
    fp16  *im2col_buf,        // Allocated L1 TCDM buffer for vol2col (Shape: M x N)
    const fp16  *bias,              // Bias vector (Shape: K)
    int apply_relu,
    uint16_t c_in, uint16_t d_in, uint16_t h_in, uint16_t w_in,
    uint16_t c_out, uint16_t d_out, uint16_t h_out, uint16_t w_out,
    uint16_t kd, uint16_t kh, uint16_t kw,
    uint16_t stride_d, uint16_t stride_h, uint16_t stride_w,
    uint16_t pad_d_low, uint16_t pad_d_high,   // Depth padding (front, back)
    uint16_t pad_h_low, uint16_t pad_h_high,   // Height padding (top, bottom)
    uint16_t pad_w_low, uint16_t pad_w_high    // Width padding (left, right)
) {
    // int iters = 0;
    for(uint16_t ch_out_idx = tid; ch_out_idx<c_out; ch_out_idx = ch_out_idx + nthreads){
        for(uint16_t d_out_idx = 0; d_out_idx<d_out; d_out_idx++){
            for(uint16_t h_out_idx = 0; h_out_idx<h_out; h_out_idx++){
                for(uint16_t w_out_idx = 0; w_out_idx<w_out; w_out_idx++){
                    fp16 acc = bias? bias[ch_out_idx]:0.0f;
                    for(uint16_t d_ker_idx = 0; d_ker_idx<kd; d_ker_idx++){
                        uint16_t d_in_idx = -pad_d_low + d_out_idx*stride_d + d_ker_idx;
                        if(d_in_idx<0 || d_in_idx>d_in) continue;
                        for(uint16_t h_ker_idx = 0; h_ker_idx<kh; h_ker_idx++){
                            uint16_t h_in_idx = -pad_h_low + h_out_idx*stride_h + h_ker_idx;
                            if(h_in_idx<0 || h_in_idx>h_in) continue;
                            for(uint16_t w_ker_idx = 0; w_ker_idx<kw; w_ker_idx++){
                                uint16_t w_in_idx = -pad_w_low + w_out_idx*stride_w + w_ker_idx;
                                if(w_in_idx<0 || w_in_idx>w_in) continue;
                                for(uint16_t ch_in_idx = 0; ch_in_idx<c_in; ch_in_idx++){
                                    uint16_t weights_idx = ch_out_idx * (c_in*kd*kh*kw)
                                        + d_ker_idx * (c_in*kh*kw) + h_ker_idx * c_in * kw 
                                        + w_ker_idx * c_in + ch_in_idx;
                                    uint16_t act_idx = d_in_idx * (h_in*w_in*c_in)
                                        + h_in_idx * w_in * c_in + w_in_idx * c_in + ch_in_idx;
                                    acc += input[act_idx] * weights[weights_idx];
                                    // mini_printf("actweight %x acc %x iter %d act %x weight %x\n",
                                    //     actweight, acc, iters, input[act_idx], weights[weights_idx]);
                                    // iters++;
                                    // if(iters>10) return;
                                }
                            }
                        }
                    }
                    if(apply_relu && (acc < 0.0f)){
                        acc = 0.0f;
                    }
                    uint16_t out_idx = d_out_idx*(h_out*w_out*c_out) + h_out_idx*w_out*c_out +
                     w_out_idx * c_out + ch_out_idx;
                    output_matrix[out_idx] = acc;
                }
            }
        }
    }
}


#endif // __pulp_cluster__