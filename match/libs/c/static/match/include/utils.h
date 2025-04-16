#ifndef __MATCH_UTILS_H__
#define __MATCH_UTILS_H__
#include <match/types.h>

int match_strcmp(const char* s1, const char* s2);

int match_byte_checksum_check(const char* data, int size, int checksum);

void handle_int_classifier(int* output_pt, int classes, int runtime_status);

#endif