#include <match_tile_indexes.h>

void substract_relative_idxs_O(tile_indexes_O* rel_tile_idxs,tile_indexes_O* abs_tile_idxs){
    abs_tile_idxs->tile_K-=rel_tile_idxs->tile_K;
    abs_tile_idxs->tile_OY-=rel_tile_idxs->tile_OY;
    abs_tile_idxs->tile_OX-=rel_tile_idxs->tile_OX;
}

void add_relative_idxs_O(tile_indexes_O* rel_tile_idxs,tile_indexes_O* abs_tile_idxs){
    abs_tile_idxs->tile_K+=rel_tile_idxs->tile_K;
    abs_tile_idxs->tile_OY+=rel_tile_idxs->tile_OY;
    abs_tile_idxs->tile_OX+=rel_tile_idxs->tile_OX;
}

void substract_relative_idxs_I(tile_indexes_I* rel_tile_idxs,tile_indexes_I* abs_tile_idxs){
    abs_tile_idxs->tile_C-=rel_tile_idxs->tile_C;
    abs_tile_idxs->tile_IY-=rel_tile_idxs->tile_IY;
    abs_tile_idxs->tile_IX-=rel_tile_idxs->tile_IX;
}

void add_relative_idxs_I(tile_indexes_I* rel_tile_idxs,tile_indexes_I* abs_tile_idxs){
    abs_tile_idxs->tile_C+=rel_tile_idxs->tile_C;
    abs_tile_idxs->tile_IY+=rel_tile_idxs->tile_IY;
    abs_tile_idxs->tile_IX+=rel_tile_idxs->tile_IX;
}

void substract_relative_idxs_X(tile_indexes_X* rel_tile_idxs,tile_indexes_X* abs_tile_idxs){
    abs_tile_idxs->tile_C-=rel_tile_idxs->tile_C;
    abs_tile_idxs->tile_IY-=rel_tile_idxs->tile_IY;
    abs_tile_idxs->tile_IX-=rel_tile_idxs->tile_IX;
}

void add_relative_idxs_X(tile_indexes_X* rel_tile_idxs,tile_indexes_X* abs_tile_idxs){
    abs_tile_idxs->tile_C+=rel_tile_idxs->tile_C;
    abs_tile_idxs->tile_IY+=rel_tile_idxs->tile_IY;
    abs_tile_idxs->tile_IX+=rel_tile_idxs->tile_IX;
}

void substract_relative_idxs_Y(tile_indexes_Y* rel_tile_idxs,tile_indexes_Y* abs_tile_idxs){
    abs_tile_idxs->tile_C-=rel_tile_idxs->tile_C;
    abs_tile_idxs->tile_IY-=rel_tile_idxs->tile_IY;
    abs_tile_idxs->tile_IX-=rel_tile_idxs->tile_IX;
}

void add_relative_idxs_Y(tile_indexes_Y* rel_tile_idxs,tile_indexes_Y* abs_tile_idxs){
    abs_tile_idxs->tile_C+=rel_tile_idxs->tile_C;
    abs_tile_idxs->tile_IY+=rel_tile_idxs->tile_IY;
    abs_tile_idxs->tile_IX+=rel_tile_idxs->tile_IX;
}

void substract_relative_idxs_W(tile_indexes_W* rel_tile_idxs,tile_indexes_W* abs_tile_idxs){
    abs_tile_idxs->tile_K-=rel_tile_idxs->tile_K;
    abs_tile_idxs->tile_C-=rel_tile_idxs->tile_C;
    abs_tile_idxs->tile_FY-=rel_tile_idxs->tile_FY;
    abs_tile_idxs->tile_FX-=rel_tile_idxs->tile_FX;
}

void add_relative_idxs_W(tile_indexes_W* rel_tile_idxs,tile_indexes_W* abs_tile_idxs){
    abs_tile_idxs->tile_K+=rel_tile_idxs->tile_K;
    abs_tile_idxs->tile_C+=rel_tile_idxs->tile_C;
    abs_tile_idxs->tile_FY+=rel_tile_idxs->tile_FY;
    abs_tile_idxs->tile_FX+=rel_tile_idxs->tile_FX;
}
