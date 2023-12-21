#ifndef _MATCH_TILE_INDEXES_H
#define _MATCH_TILE_INDEXES_H
typedef struct tile_indexes_O_t
{
    int tile_K;
    int tile_OY;
    int tile_OX;
}tile_indexes_O;

typedef struct tile_indexes_W_t
{
    int tile_K;
    int tile_C;
    int tile_FY;
    int tile_FX;
}tile_indexes_W;

typedef struct tile_indexes_I_t
{
    int tile_C;
    int tile_IY;
    int tile_IX;
}tile_indexes_I;

typedef tile_indexes_I tile_indexes_X;
typedef tile_indexes_I tile_indexes_Y;

void substract_relative_idxs_O(tile_indexes_O* rel_tile_idxs,tile_indexes_O* abs_tile_idxs);

void add_relative_idxs_O(tile_indexes_O* rel_tile_idxs,tile_indexes_O* abs_tile_idxs);

void substract_relative_idxs_I(tile_indexes_I* rel_tile_idxs,tile_indexes_I* abs_tile_idxs);

void add_relative_idxs_I(tile_indexes_I* rel_tile_idxs,tile_indexes_I* abs_tile_idxs);

void substract_relative_idxs_X(tile_indexes_X* rel_tile_idxs,tile_indexes_X* abs_tile_idxs);

void add_relative_idxs_X(tile_indexes_X* rel_tile_idxs,tile_indexes_X* abs_tile_idxs);

void substract_relative_idxs_Y(tile_indexes_Y* rel_tile_idxs,tile_indexes_Y* abs_tile_idxs);

void add_relative_idxs_Y(tile_indexes_Y* rel_tile_idxs,tile_indexes_Y* abs_tile_idxs);

void substract_relative_idxs_W(tile_indexes_W* rel_tile_idxs,tile_indexes_W* abs_tile_idxs);

void add_relative_idxs_W(tile_indexes_W* rel_tile_idxs,tile_indexes_W* abs_tile_idxs);

#endif