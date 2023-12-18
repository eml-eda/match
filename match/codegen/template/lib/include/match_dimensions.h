#include <match_params.h>
// output dimension
typedef struct dimension_O_t
{
  int dim_K_size;
  int size_K[NUM_MEMORY_LEVELS];
  int dim_OY_size;
  int size_OY[NUM_MEMORY_LEVELS];
  int dim_OX_size;
  int size_OX[NUM_MEMORY_LEVELS];
}dimension_O;


// weight dimension
typedef struct dimension_W_t
{
  int dim_K_size;
  int size_K[NUM_MEMORY_LEVELS];
  int dim_C_size;
  int size_C[NUM_MEMORY_LEVELS];
  int dim_FY_size;
  int size_FY[NUM_MEMORY_LEVELS];
  int dim_FX_size;
  int size_FX[NUM_MEMORY_LEVELS];
}dimension_W;


// input dimension
typedef struct dimension_I_t
{
  int dim_C_size;
  int size_C[NUM_MEMORY_LEVELS];
  int dim_IY_size;
  int size_IY[NUM_MEMORY_LEVELS];
  // TODO: needed?
  int size_FY[NUM_MEMORY_LEVELS];
  int overlap_IY_x;
  int overlap_IY_y;
  int pad_IY_x;
  int pad_IY_y;
  int dim_IX_size;
  int size_IX[NUM_MEMORY_LEVELS];
  // TODO: needed?
  int size_FX[NUM_MEMORY_LEVELS];
  int overlap_IX_x;
  int overlap_IX_y;
  int pad_IX_x;
  int pad_IX_y;
}dimension_I;

typedef dimension_I dimension_X;
typedef dimension_I dimension_Y;