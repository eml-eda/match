#ifndef __MATCH_UTILS_H__
#define __MATCH_UTILS_H__

#define __STDC_WANT_IEC_60559_TYPES_EXT__
#include <float.h>

#include <match/types.h>

int match_strcmp(const char *s1, const char *s2);

int match_byte_checksum_check(const char *data, int size, int checksum);

void handle_int_classifier(int *output_pt, int classes, int runtime_status);

void handle_fp32_classifier(float *output_pt, int classes, int runtime_status);

#ifdef FLT16_MIN
void handle_fp16_classifier(_Float16 *output_pt, int classes, int runtime_status);
#endif

#endif