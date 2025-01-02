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