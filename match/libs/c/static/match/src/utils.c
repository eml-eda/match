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

void handle_int_classifier(int* output_pt, int classes, int runtime_status){
    int max_idx = 0;
    int max_val = output_pt[0];
    printf("[MATCH OUTPUT] Values:\n%d, ", max_val);
    for(int idx=1; idx<classes; idx++){
        printf("%d, ", output_pt[idx]);
        if(output_pt[idx]>max_val){
            max_val = output_pt[idx];
            max_idx = idx;
        }
    }
    printf("\n[MATCH OUTPUT] Label predicted %d with value %d\n", max_idx, max_val);
}