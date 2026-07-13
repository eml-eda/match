#define __STDC_WANT_IEC_60559_TYPES_EXT__
#include <float.h>

#include <match/utils.h>

int match_strcmp(const char* s1, const char* s2) {
    // Align to word size for more efficient comparison
    const unsigned char* p1 = (const unsigned char*)s1;
    const unsigned char* p2 = (const unsigned char*)s2;

    // Compare bytes until mismatch or null terminator
    while (*p1 && *p1 == *p2) {
        ++p1;
        ++p2;
    }

    // Return difference of mismatched characters (or 0 if equal)
    return *p1 - *p2;
}

int match_byte_checksum_check(const char* data, int size, int checksum) {
    // Calculate checksum
    int sum = 0;
    for (int i = 0; i < size; ++i) {
        sum += (unsigned char)data[i];
    }

    // Check if checksum matches
    return sum - checksum;
}

float match_float_checksum_check(void* data, int size, double checksum) {
    // Calculate checksum
    double sum = 0.0;
    int size_f = size / sizeof(float);
    /*
    if(print_value){
        // Print array data for debugging
        printf("[LAYER OUTPUT] Values:\n");
        for (int i = 0; i < size_f; ++i) {
            printf("%f, ", data[i]);
        }
        printf("\n");
    }
    */

    // compute and return the checksum
    for (int i = 0; i < size_f; ++i) {        
        sum += ((float*)data)[i];
    } 
    /*  
    if(print_value){
        printf("[LAYER OUTPUT] Computed Checksum: %f\n", sum);
    }
    */
    // printf("[LAYER OUTPUT] Computed Checksum: %f expected one %f size %d\n", sum, checksum_f, size_f);
    // Compute the relative error (1e-20 gives numerical stability)
    return sum - checksum;
}

static float half_to_float(uint16_t h) {
    uint32_t sign = (h >> 15) & 0x1;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    uint32_t f;

    if (exp == 0) {
        if (mant == 0) {
            f = sign << 31;
        } else {
            // subnormal -> normalize
            while ((mant & 0x400) == 0) {
                mant <<= 1;
                exp--;
            }
            exp += 1;
            mant &= 0x3FF;
            exp = (uint32_t)(exp + (127 - 15));
            f = (sign << 31) | (exp << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        // Inf or NaN
        f = (sign << 31) | 0x7F800000u | (mant << 13);
    } else {
        exp = (uint32_t)(exp + (127 - 15));
        f = (sign << 31) | (exp << 23) | (mant << 13);
    }

    union { uint32_t u; float f; } conv;
    conv.u = f;
    return conv.f;
}

float match_fp16_checksum_check(void* data, int size, double checksum) {
    // Calculate checksum
    double sum = 0.0;

    // compute and return the checksum
    for (int i = 0; i < size/2; ++i) {        
        sum += half_to_float(((uint16_t*)data)[i]);
    }
    // Compute the relative error (1e-20 gives numerical stability)
    return (float)(sum - checksum);
}

double match_fp16_checksum_comp(void* data, int size, double checksum) {
    // Calculate checksum
    double sum = 0.0;

    // compute and return the checksum
    for (int i = 0; i < size/2; ++i) {        
        sum += half_to_float(((uint16_t*)data)[i]);
    }
    // Compute the relative error (1e-20 gives numerical stability)
    return sum;
}

void handle_int_classifier(int* output_pt, int classes, int runtime_status){
    int max_idx = 0;
    int max_val = output_pt[0];
    printf("[MATCH OUTPUT] Values:  %d, ", max_val);
    for (int idx = 1; idx < classes; idx++) {
        printf("%d, ", output_pt[idx]);
        if (output_pt[idx] > max_val) {
            max_val = output_pt[idx];
            max_idx = idx;
        }
    }
    printf("\r\n[MATCH OUTPUT] Label predicted %d with value %d\r\n", max_idx, max_val);
}


void handle_fp32_classifier(float *output_pt, int classes, int runtime_status) {
    int max_idx = 0;
    float max_val = output_pt[0];
    printf("[MATCH OUTPUT] Values:  %f, ", max_val);
    for (int idx = 1; idx < classes; idx++) {
        printf("%f, ", output_pt[idx]);
        if (output_pt[idx] > max_val) {
            max_val = output_pt[idx];
            max_idx = idx;
        }
    }
    printf("\r\n[MATCH OUTPUT] Label predicted %d with value %f\r\n", max_idx, max_val);
}

#ifdef FLT16_MIN
void handle_fp16_classifier(_Float16 *output_pt, int classes, int runtime_status) {
    int max_idx = 0;
    _Float16 max_val = output_pt[0];
    printf("[MATCH OUTPUT] Values:  %f, ", (float)max_val);
    for (int idx = 1; idx < classes; idx++) {
        printf("%f, ", (float)output_pt[idx]);
        if (output_pt[idx] > max_val) {
            max_val = output_pt[idx];
            max_idx = idx;
        }
    }
    printf("\r\n[MATCH OUTPUT] Label predicted %d with value %f\r\n", max_idx, (float)max_val);
}
#endif