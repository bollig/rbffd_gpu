#ifndef VIENNACL_LINALG_KERNELS_MATRIX_PROD_ROW_ROW_COL_SOURCE_HPP_
#define VIENNACL_LINALG_KERNELS_MATRIX_PROD_ROW_ROW_COL_SOURCE_HPP_
//Automatically generated file from auxiliary-directory, do not edit manually!
namespace viennacl
{
 namespace linalg
 {
  namespace kernels
  {
const char * const matrix_prod_row_row_col_align1_prod16_AA = 
"// file automatically generated - do not edit!\n"
"// matrix-matrix multiplication C = A * B\n"
"// matrix layouts: C...col_major, A...row_major, B...row_major\n"
"__kernel void prod16_AA(\n"
"          float alpha,\n"
"          __global const float * A,\n"
"          unsigned int A_row_start,\n"
"          unsigned int A_col_start,\n"
"          unsigned int A_row_inc,\n"
"          unsigned int A_col_inc,\n"
"          unsigned int A_row_size,\n"
"          unsigned int A_col_size,\n"
"          unsigned int A_internal_rows,\n"
"          unsigned int A_internal_cols,\n"
"          __global const float * B,  \n"
"          unsigned int B_row_start,\n"
"          unsigned int B_col_start,\n"
"          unsigned int B_row_inc,\n"
"          unsigned int B_col_inc,\n"
"          unsigned int B_row_size,\n"
"          unsigned int B_col_size,\n"
"          unsigned int B_internal_rows,\n"
"          unsigned int B_internal_cols,\n"
"          float beta,\n"
"          __global float * C,\n"
"          unsigned int C_row_start,\n"
"          unsigned int C_col_start,\n"
"          unsigned int C_row_inc,\n"
"          unsigned int C_col_inc,\n"
"          unsigned int C_row_size,\n"
"          unsigned int C_col_size,\n"
"          unsigned int C_internal_rows,\n"
"          unsigned int C_internal_cols) \n"
"{ \n"
"  size_t row_block_id = get_group_id(1);\n"
"  size_t col_block_id = get_group_id(0);\n"
"  size_t row_thread_id = get_local_id(1);\n"
"  size_t col_thread_id = get_local_id(0);\n"
"  __local float As[256];\n"
"  float cv[16] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};\n"
"  size_t aBegin = (row_block_id * 16 * A_row_inc + A_row_start) * A_internal_cols + A_col_start;\n"
"  size_t aStep = 16 * A_col_inc;\n"
"  size_t aEnd = aBegin + A_col_inc * A_col_size;\n"
"  size_t bBegin = (col_block_id * 64 * B_col_inc + B_col_start) + B_row_start * B_internal_cols;\n"
"  size_t bStep = 16 * B_row_inc * B_internal_cols;\n"
"  for(size_t a = aBegin, b = bBegin; a < aEnd; a += aStep, b += bStep) { \n"
"    for(size_t i = 0; i < 4; i++)  \n"
"      As[ (i*4 + row_thread_id) + 16 * col_thread_id] = (A[a + A_internal_cols * A_row_inc * (i * 4 + row_thread_id) + A_col_inc * col_thread_id]);\n"
"    barrier(CLK_LOCAL_MEM_FENCE); \n"
"    __local  float *ap = As; \n"
"    __global float *bp = B + (b + (16 * row_thread_id + col_thread_id) * B_col_inc); \n"
"    for(size_t i = 0; i < 16; i++) { \n"
"      float bv = bp[i * B_internal_cols]; \n"
"      for(size_t k = 0; k < 16; k++)  \n"
"	    cv[k] += ap[k] * bv; \n"
"      ap += 16; \n"
"    } \n"
"    barrier(CLK_LOCAL_MEM_FENCE); \n"
"  } \n"
"  int c = C_row_inc * 16 * row_block_id + C_row_start + (64 * C_col_inc * col_block_id + C_col_start) * C_internal_rows \n"
"          + C_internal_rows * C_col_inc * (16 * row_thread_id + col_thread_id); \n"
"  for(size_t i = 0; i < 16; i++) { \n"
"    C[c] = alpha * cv[i] + beta * C[c]; \n"
"      c += C_row_inc; \n"
"  } \n"
"} \n"
; //matrix_prod_row_row_col_align1_prod16_AA

const char * const matrix_prod_row_row_col_align1_prod16_AT = 
"// file automatically generated - do not edit!\n"
"// matrix-matrix multiplication C = A * B^T\n"
"// matrix layouts: C...col_major, A...row_major, B...row_major\n"
"__kernel void prod16_AT(\n"
"          float alpha,\n"
"          __global const float * A,\n"
"          unsigned int A_row_start,\n"
"          unsigned int A_col_start,\n"
"          unsigned int A_row_inc,\n"
"          unsigned int A_col_inc,\n"
"          unsigned int A_row_size,\n"
"          unsigned int A_col_size,\n"
"          unsigned int A_internal_rows,\n"
"          unsigned int A_internal_cols,\n"
"          __global const float * B,  \n"
"          unsigned int B_row_start,\n"
"          unsigned int B_col_start,\n"
"          unsigned int B_row_inc,\n"
"          unsigned int B_col_inc,\n"
"          unsigned int B_row_size,\n"
"          unsigned int B_col_size,\n"
"          unsigned int B_internal_rows,\n"
"          unsigned int B_internal_cols,\n"
"          float beta,\n"
"          __global float * C,\n"
"          unsigned int C_row_start,\n"
"          unsigned int C_col_start,\n"
"          unsigned int C_row_inc,\n"
"          unsigned int C_col_inc,\n"
"          unsigned int C_row_size,\n"
"          unsigned int C_col_size,\n"
"          unsigned int C_internal_rows,\n"
"          unsigned int C_internal_cols) \n"
"{ \n"
"  size_t row_block_id = get_group_id(1);\n"
"  size_t col_block_id = get_group_id(0);\n"
"  size_t row_thread_id = get_local_id(1);\n"
"  size_t col_thread_id = get_local_id(0);\n"
"  __local float As[256];\n"
"  float cv[16] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};\n"
"  size_t aBegin = (row_block_id * 16 * A_row_inc + A_row_start) * A_internal_cols + A_col_start;\n"
"  size_t aStep = 16 * A_col_inc;\n"
"  size_t aEnd = aBegin + A_col_inc * A_col_size;\n"
"  size_t bBegin = (col_block_id * 64 * B_row_inc + B_row_start) * B_internal_cols + B_col_start;\n"
"  size_t bStep = 16 * B_col_inc;\n"
"  for(size_t a = aBegin, b = bBegin; a < aEnd; a += aStep, b += bStep) { \n"
"    for(size_t i = 0; i < 4; i++)  \n"
"      As[ (i*4 + row_thread_id) + 16 * col_thread_id] = (A[a + A_internal_cols * A_row_inc * (i * 4 + row_thread_id) + A_col_inc * col_thread_id]);\n"
"    barrier(CLK_LOCAL_MEM_FENCE); \n"
"    __local  float *ap = As; \n"
"    __global float *bp = B + (b + (16 * row_thread_id + col_thread_id) * B_row_inc * B_internal_cols); \n"
"    for(size_t i = 0; i < 16; i++) { \n"
"      float bv = bp[i]; \n"
"      for(size_t k = 0; k < 16; k++)  \n"
"	    cv[k] += ap[k] * bv; \n"
"      ap += 16; \n"
"    } \n"
"    barrier(CLK_LOCAL_MEM_FENCE); \n"
"  } \n"
"  int c = C_row_inc * 16 * row_block_id + C_row_start + (64 * C_col_inc * col_block_id + C_col_start) * C_internal_rows \n"
"          + C_internal_rows * C_col_inc * (16 * row_thread_id + col_thread_id); \n"
"  for(size_t i = 0; i < 16; i++) { \n"
"    C[c] = alpha * cv[i] + beta * C[c]; \n"
"      c += C_row_inc; \n"
"  } \n"
"} \n"
; //matrix_prod_row_row_col_align1_prod16_AT

const char * const matrix_prod_row_row_col_align1_prod16_TA = 
"// file automatically generated - do not edit!\n"
"// matrix-matrix multiplication C = A^T * B\n"
"// matrix layouts: C...col_major, A...row_major, B...row_major\n"
"__kernel void prod16_TA(\n"
"          float alpha,\n"
"          __global const float * A,\n"
"          unsigned int A_row_start,\n"
"          unsigned int A_col_start,\n"
"          unsigned int A_row_inc,\n"
"          unsigned int A_col_inc,\n"
"          unsigned int A_row_size,\n"
"          unsigned int A_col_size,\n"
"          unsigned int A_internal_rows,\n"
"          unsigned int A_internal_cols,\n"
"          __global const float * B,  \n"
"          unsigned int B_row_start,\n"
"          unsigned int B_col_start,\n"
"          unsigned int B_row_inc,\n"
"          unsigned int B_col_inc,\n"
"          unsigned int B_row_size,\n"
"          unsigned int B_col_size,\n"
"          unsigned int B_internal_rows,\n"
"          unsigned int B_internal_cols,\n"
"          float beta,\n"
"          __global float * C,\n"
"          unsigned int C_row_start,\n"
"          unsigned int C_col_start,\n"
"          unsigned int C_row_inc,\n"
"          unsigned int C_col_inc,\n"
"          unsigned int C_row_size,\n"
"          unsigned int C_col_size,\n"
"          unsigned int C_internal_rows,\n"
"          unsigned int C_internal_cols) \n"
"{ \n"
"  size_t row_block_id = get_group_id(1);\n"
"  size_t col_block_id = get_group_id(0);\n"
"  size_t row_thread_id = get_local_id(1);\n"
"  size_t col_thread_id = get_local_id(0);\n"
"  __local float As[256];\n"
"  float cv[16] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};\n"
"  size_t aBegin = (row_block_id * 16 * A_col_inc + A_col_start) + A_row_start * A_internal_cols;\n"
"  size_t aStep = 16 * A_internal_cols * A_row_inc;\n"
"  size_t aEnd = aBegin + A_internal_cols * A_row_inc * A_row_size;\n"
"  size_t bBegin = (col_block_id * 64 * B_col_inc + B_col_start) + B_row_start * B_internal_cols;\n"
"  size_t bStep = 16 * B_row_inc * B_internal_cols;\n"
"  for(size_t a = aBegin, b = bBegin; a < aEnd; a += aStep, b += bStep) { \n"
"    for(size_t i = 0; i < 4; i++)  \n"
"      As[ (i*4 + row_thread_id) + 16 * col_thread_id] = (A[a + A_col_inc * (i * 4 + row_thread_id) + A_internal_cols * A_row_inc * col_thread_id]);\n"
"    barrier(CLK_LOCAL_MEM_FENCE); \n"
"    __local  float *ap = As; \n"
"    __global float *bp = B + (b + (16 * row_thread_id + col_thread_id) * B_col_inc); \n"
"    for(size_t i = 0; i < 16; i++) { \n"
"      float bv = bp[i * B_internal_cols]; \n"
"      for(size_t k = 0; k < 16; k++)  \n"
"	    cv[k] += ap[k] * bv; \n"
"      ap += 16; \n"
"    } \n"
"    barrier(CLK_LOCAL_MEM_FENCE); \n"
"  } \n"
"  int c = C_row_inc * 16 * row_block_id + C_row_start + (64 * C_col_inc * col_block_id + C_col_start) * C_internal_rows \n"
"          + C_internal_rows * C_col_inc * (16 * row_thread_id + col_thread_id); \n"
"  for(size_t i = 0; i < 16; i++) { \n"
"    C[c] = alpha * cv[i] + beta * C[c]; \n"
"      c += C_row_inc; \n"
"  } \n"
"} \n"
; //matrix_prod_row_row_col_align1_prod16_TA

const char * const matrix_prod_row_row_col_align1_prod_AT = 
"// file automatically generated - do not edit!\n"
"// matrix-matrix multiplication C = A * B^T\n"
"// matrix layouts: C...col_major, A...row_major, B...row_major\n"
"__kernel void prod_AT(\n"
"          float alpha,\n"
"          __global const float * A,\n"
"          unsigned int A_row_start,\n"
"          unsigned int A_col_start,\n"
"          unsigned int A_row_inc,\n"
"          unsigned int A_col_inc,\n"
"          unsigned int A_row_size,\n"
"          unsigned int A_col_size,\n"
"          unsigned int A_internal_rows,\n"
"          unsigned int A_internal_cols,\n"
"          __global const float * B,  \n"
"          unsigned int B_row_start,\n"
"          unsigned int B_col_start,\n"
"          unsigned int B_row_inc,\n"
"          unsigned int B_col_inc,\n"
"          unsigned int B_row_size,\n"
"          unsigned int B_col_size,\n"
"          unsigned int B_internal_rows,\n"
"          unsigned int B_internal_cols,\n"
"          float beta,\n"
"          __global float * C,\n"
"          unsigned int C_row_start,\n"
"          unsigned int C_col_start,\n"
"          unsigned int C_row_inc,\n"
"          unsigned int C_col_inc,\n"
"          unsigned int C_row_size,\n"
"          unsigned int C_col_size,\n"
"          unsigned int C_internal_rows,\n"
"          unsigned int C_internal_cols) \n"
"{ \n"
"  __local float bufA[272];\n"
"  __local float bufB[272];\n"
"  size_t block_size = 16;//get_local_size(0);\n"
"  size_t row_block_id = get_group_id(0);\n"
"  size_t col_block_id = get_group_id(1);\n"
"  size_t row_thread_id = get_local_id(0);\n"
"  size_t col_thread_id = get_local_id(1);\n"
"  size_t aBegin = (row_block_id * block_size * A_row_inc + A_row_start) * A_internal_cols + A_col_start;\n"
"  size_t aStep = block_size * A_col_inc;\n"
"  size_t bBegin = (col_block_id * block_size * B_row_inc + B_row_start) * B_internal_cols + B_col_start;\n"
"  size_t bStep = block_size * B_col_inc;\n"
"  size_t block_num = (A_col_size + block_size - 1) / block_size;\n"
"  float Csub = 0;\n"
"  size_t aOffset = row_thread_id * A_row_inc + col_thread_id * A_col_inc * A_internal_cols;\n"
"  size_t bOffset = row_thread_id * B_row_inc + col_thread_id * B_col_inc * B_internal_cols;\n"
"  size_t row_thread_id_times_block_size = row_thread_id * (block_size + 1);\n"
"  size_t col_thread_id_times_block_size = col_thread_id * (block_size + 1);\n"
"  for (size_t block = 0;\n"
"           block < block_num;\n"
"           ++block)\n"
"  {\n"
"    bufA[col_thread_id_times_block_size + row_thread_id] = ((block * block_size + row_thread_id < A_col_size) && (row_block_id * block_size + col_thread_id < A_row_size)) ? A[aBegin + aOffset] : 0;\n"
"    bufB[col_thread_id_times_block_size + row_thread_id] = ((block * block_size + row_thread_id < B_col_size) && (col_block_id * block_size + col_thread_id < B_row_size)) ? B[bBegin + bOffset] : 0;\n"
"    barrier(CLK_LOCAL_MEM_FENCE);\n"
"    __local float * bufAptr = bufA + row_thread_id_times_block_size;\n"
"    __local float * bufBptr = bufB + col_thread_id_times_block_size;\n"
"      for(int i = 0; i < 4; i++) {\n"
"      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;\n"
"      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;\n"
"      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;\n"
"      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;\n"
"     }\n"
"    barrier(CLK_LOCAL_MEM_FENCE);\n"
"    aBegin += aStep;\n"
"    bBegin += bStep;\n"
"  }\n"
"  if (get_global_id(0) < A_row_size && get_global_id(1) < B_row_size)\n"
"    C[get_global_id(0) * C_row_inc + C_row_start + (get_global_id(1) * C_col_inc + C_col_start) * C_internal_rows] = alpha * Csub + beta * C[get_global_id(0) * C_row_inc + C_row_start + (get_global_id(1) * C_col_inc + C_col_start) * C_internal_rows];\n"
"}\n"
; //matrix_prod_row_row_col_align1_prod_AT

const char * const matrix_prod_row_row_col_align1_prod16_TT = 
"// file automatically generated - do not edit!\n"
"// matrix-matrix multiplication C = A^T * B^T\n"
"// matrix layouts: C...col_major, A...row_major, B...row_major\n"
"__kernel void prod16_TT(\n"
"          float alpha,\n"
"          __global const float * A,\n"
"          unsigned int A_row_start,\n"
"          unsigned int A_col_start,\n"
"          unsigned int A_row_inc,\n"
"          unsigned int A_col_inc,\n"
"          unsigned int A_row_size,\n"
"          unsigned int A_col_size,\n"
"          unsigned int A_internal_rows,\n"
"          unsigned int A_internal_cols,\n"
"          __global const float * B,  \n"
"          unsigned int B_row_start,\n"
"          unsigned int B_col_start,\n"
"          unsigned int B_row_inc,\n"
"          unsigned int B_col_inc,\n"
"          unsigned int B_row_size,\n"
"          unsigned int B_col_size,\n"
"          unsigned int B_internal_rows,\n"
"          unsigned int B_internal_cols,\n"
"          float beta,\n"
"          __global float * C,\n"
"          unsigned int C_row_start,\n"
"          unsigned int C_col_start,\n"
"          unsigned int C_row_inc,\n"
"          unsigned int C_col_inc,\n"
"          unsigned int C_row_size,\n"
"          unsigned int C_col_size,\n"
"          unsigned int C_internal_rows,\n"
"          unsigned int C_internal_cols) \n"
"{ \n"
"  size_t row_block_id = get_group_id(1);\n"
"  size_t col_block_id = get_group_id(0);\n"
"  size_t row_thread_id = get_local_id(1);\n"
"  size_t col_thread_id = get_local_id(0);\n"
"  __local float As[256];\n"
"  float cv[16] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};\n"
"  size_t aBegin = (row_block_id * 16 * A_col_inc + A_col_start) + A_row_start * A_internal_cols;\n"
"  size_t aStep = 16 * A_internal_cols * A_row_inc;\n"
"  size_t aEnd = aBegin + A_internal_cols * A_row_inc * A_row_size;\n"
"  size_t bBegin = (col_block_id * 64 * B_row_inc + B_row_start) * B_internal_cols + B_col_start;\n"
"  size_t bStep = 16 * B_col_inc;\n"
"  for(size_t a = aBegin, b = bBegin; a < aEnd; a += aStep, b += bStep) { \n"
"    for(size_t i = 0; i < 4; i++)  \n"
"      As[ (i*4 + row_thread_id) + 16 * col_thread_id] = (A[a + A_col_inc * (i * 4 + row_thread_id) + A_internal_cols * A_row_inc * col_thread_id]);\n"
"    barrier(CLK_LOCAL_MEM_FENCE); \n"
"    __local  float *ap = As; \n"
"    __global float *bp = B + (b + (16 * row_thread_id + col_thread_id) * B_row_inc * B_internal_cols); \n"
"    for(size_t i = 0; i < 16; i++) { \n"
"      float bv = bp[i]; \n"
"      for(size_t k = 0; k < 16; k++)  \n"
"	    cv[k] += ap[k] * bv; \n"
"      ap += 16; \n"
"    } \n"
"    barrier(CLK_LOCAL_MEM_FENCE); \n"
"  } \n"
"  int c = C_row_inc * 16 * row_block_id + C_row_start + (64 * C_col_inc * col_block_id + C_col_start) * C_internal_rows \n"
"          + C_internal_rows * C_col_inc * (16 * row_thread_id + col_thread_id); \n"
"  for(size_t i = 0; i < 16; i++) { \n"
"    C[c] = alpha * cv[i] + beta * C[c]; \n"
"      c += C_row_inc; \n"
"  } \n"
"} \n"
; //matrix_prod_row_row_col_align1_prod16_TT

const char * const matrix_prod_row_row_col_align1_prod_TT = 
"// file automatically generated - do not edit!\n"
"// matrix-matrix multiplication C = A^T * B^T\n"
"// matrix layouts: C...col_major, A...row_major, B...row_major\n"
"__kernel void prod_TT(\n"
"          float alpha,\n"
"          __global const float * A,\n"
"          unsigned int A_row_start,\n"
"          unsigned int A_col_start,\n"
"          unsigned int A_row_inc,\n"
"          unsigned int A_col_inc,\n"
"          unsigned int A_row_size,\n"
"          unsigned int A_col_size,\n"
"          unsigned int A_internal_rows,\n"
"          unsigned int A_internal_cols,\n"
"          __global const float * B,  \n"
"          unsigned int B_row_start,\n"
"          unsigned int B_col_start,\n"
"          unsigned int B_row_inc,\n"
"          unsigned int B_col_inc,\n"
"          unsigned int B_row_size,\n"
"          unsigned int B_col_size,\n"
"          unsigned int B_internal_rows,\n"
"          unsigned int B_internal_cols,\n"
"          float beta,\n"
"          __global float * C,\n"
"          unsigned int C_row_start,\n"
"          unsigned int C_col_start,\n"
"          unsigned int C_row_inc,\n"
"          unsigned int C_col_inc,\n"
"          unsigned int C_row_size,\n"
"          unsigned int C_col_size,\n"
"          unsigned int C_internal_rows,\n"
"          unsigned int C_internal_cols) \n"
"{ \n"
"  __local float bufA[272];\n"
"  __local float bufB[272];\n"
"  size_t block_size = 16;//get_local_size(0);\n"
"  size_t row_block_id = get_group_id(0);\n"
"  size_t col_block_id = get_group_id(1);\n"
"  size_t row_thread_id = get_local_id(0);\n"
"  size_t col_thread_id = get_local_id(1);\n"
"  size_t aBegin = (row_block_id * block_size * A_col_inc + A_col_start) + A_row_start * A_internal_cols;\n"
"  size_t aStep = block_size * A_row_inc * A_internal_cols;\n"
"  size_t bBegin = (col_block_id * block_size * B_row_inc + B_row_start) * B_internal_cols + B_col_start;\n"
"  size_t bStep = block_size * B_col_inc;\n"
"  size_t block_num = (A_row_size + block_size - 1) / block_size;\n"
"  float Csub = 0;\n"
"  size_t aOffset = row_thread_id * A_row_inc + col_thread_id * A_col_inc * A_internal_cols;\n"
"  size_t bOffset = row_thread_id * B_row_inc + col_thread_id * B_col_inc * B_internal_cols;\n"
"  size_t row_thread_id_times_block_size = row_thread_id * (block_size + 1);\n"
"  size_t col_thread_id_times_block_size = col_thread_id * (block_size + 1);\n"
"  for (size_t block = 0;\n"
"           block < block_num;\n"
"           ++block)\n"
"  {\n"
"    bufA[row_thread_id_times_block_size + col_thread_id] = ((block * block_size + col_thread_id < A_row_size) && (row_block_id * block_size + row_thread_id < A_col_size)) ? A[aBegin + aOffset] : 0;\n"
"    bufB[col_thread_id_times_block_size + row_thread_id] = ((block * block_size + row_thread_id < B_col_size) && (col_block_id * block_size + col_thread_id < B_row_size)) ? B[bBegin + bOffset] : 0;\n"
"    barrier(CLK_LOCAL_MEM_FENCE);\n"
"    __local float * bufAptr = bufA + row_thread_id_times_block_size;\n"
"    __local float * bufBptr = bufB + col_thread_id_times_block_size;\n"
"      for(int i = 0; i < 4; i++) {\n"
"      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;\n"
"      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;\n"
"      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;\n"
"      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;\n"
"     }\n"
"    barrier(CLK_LOCAL_MEM_FENCE);\n"
"    aBegin += aStep;\n"
"    bBegin += bStep;\n"
"  }\n"
"  if (get_global_id(0) < A_col_size && get_global_id(1) < B_row_size)\n"
"    C[get_global_id(0) * C_row_inc + C_row_start + (get_global_id(1) * C_col_inc + C_col_start) * C_internal_rows] = alpha * Csub + beta * C[get_global_id(0) * C_row_inc + C_row_start + (get_global_id(1) * C_col_inc + C_col_start) * C_internal_rows];\n"
"}\n"
; //matrix_prod_row_row_col_align1_prod_TT

const char * const matrix_prod_row_row_col_align1_prod_TA = 
"// file automatically generated - do not edit!\n"
"// matrix-matrix multiplication C = A^T * B\n"
"// matrix layouts: C...col_major, A...row_major, B...row_major\n"
"__kernel void prod_TA(\n"
"          float alpha,\n"
"          __global const float * A,\n"
"          unsigned int A_row_start,\n"
"          unsigned int A_col_start,\n"
"          unsigned int A_row_inc,\n"
"          unsigned int A_col_inc,\n"
"          unsigned int A_row_size,\n"
"          unsigned int A_col_size,\n"
"          unsigned int A_internal_rows,\n"
"          unsigned int A_internal_cols,\n"
"          __global const float * B,  \n"
"          unsigned int B_row_start,\n"
"          unsigned int B_col_start,\n"
"          unsigned int B_row_inc,\n"
"          unsigned int B_col_inc,\n"
"          unsigned int B_row_size,\n"
"          unsigned int B_col_size,\n"
"          unsigned int B_internal_rows,\n"
"          unsigned int B_internal_cols,\n"
"          float beta,\n"
"          __global float * C,\n"
"          unsigned int C_row_start,\n"
"          unsigned int C_col_start,\n"
"          unsigned int C_row_inc,\n"
"          unsigned int C_col_inc,\n"
"          unsigned int C_row_size,\n"
"          unsigned int C_col_size,\n"
"          unsigned int C_internal_rows,\n"
"          unsigned int C_internal_cols) \n"
"{ \n"
"  __local float bufA[272];\n"
"  __local float bufB[272];\n"
"  size_t block_size = 16;//get_local_size(0);\n"
"  size_t row_block_id = get_group_id(0);\n"
"  size_t col_block_id = get_group_id(1);\n"
"  size_t row_thread_id = get_local_id(0);\n"
"  size_t col_thread_id = get_local_id(1);\n"
"  size_t aBegin = (row_block_id * block_size * A_col_inc + A_col_start) + A_row_start * A_internal_cols;\n"
"  size_t aStep = block_size * A_row_inc * A_internal_cols;\n"
"  size_t bBegin = (col_block_id * block_size * B_col_inc + B_col_start) + B_row_start * B_internal_cols;\n"
"  size_t bStep = block_size * B_internal_cols * B_row_inc;\n"
"  size_t block_num = (A_row_size + block_size - 1) / block_size;\n"
"  float Csub = 0;\n"
"  size_t aOffset = row_thread_id * A_row_inc + col_thread_id * A_col_inc * A_internal_cols;\n"
"  size_t bOffset = row_thread_id * B_row_inc + col_thread_id * B_col_inc * B_internal_cols;\n"
"  size_t row_thread_id_times_block_size = row_thread_id * (block_size + 1);\n"
"  size_t col_thread_id_times_block_size = col_thread_id * (block_size + 1);\n"
"  for (size_t block = 0;\n"
"           block < block_num;\n"
"           ++block)\n"
"  {\n"
"    bufA[row_thread_id_times_block_size + col_thread_id] = ((block * block_size + col_thread_id < A_row_size) && (row_block_id * block_size + row_thread_id < A_col_size)) ? A[aBegin + aOffset] : 0;\n"
"    bufB[row_thread_id_times_block_size + col_thread_id] = ((block * block_size + col_thread_id < B_row_size) && (col_block_id * block_size + row_thread_id < B_col_size)) ? B[bBegin + bOffset] : 0;\n"
"    barrier(CLK_LOCAL_MEM_FENCE);\n"
"    __local float * bufAptr = bufA + row_thread_id_times_block_size;\n"
"    __local float * bufBptr = bufB + col_thread_id_times_block_size;\n"
"      for(int i = 0; i < 4; i++) {\n"
"      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;\n"
"      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;\n"
"      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;\n"
"      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;\n"
"     }\n"
"    barrier(CLK_LOCAL_MEM_FENCE);\n"
"    aBegin += aStep;\n"
"    bBegin += bStep;\n"
"  }\n"
"  if (get_global_id(0) < A_col_size && get_global_id(1) < B_col_size)\n"
"    C[get_global_id(0) * C_row_inc + C_row_start + (get_global_id(1) * C_col_inc + C_col_start) * C_internal_rows] = alpha * Csub + beta * C[get_global_id(0) * C_row_inc + C_row_start + (get_global_id(1) * C_col_inc + C_col_start) * C_internal_rows];\n"
"}\n"
; //matrix_prod_row_row_col_align1_prod_TA

const char * const matrix_prod_row_row_col_align1_prod_AA = 
"// file automatically generated - do not edit!\n"
"// matrix-matrix multiplication C = A * B\n"
"// matrix layouts: C...col_major, A...row_major, B...row_major\n"
"__kernel void prod_AA(\n"
"          float alpha,\n"
"          __global const float * A,\n"
"          unsigned int A_row_start,\n"
"          unsigned int A_col_start,\n"
"          unsigned int A_row_inc,\n"
"          unsigned int A_col_inc,\n"
"          unsigned int A_row_size,\n"
"          unsigned int A_col_size,\n"
"          unsigned int A_internal_rows,\n"
"          unsigned int A_internal_cols,\n"
"          __global const float * B,  \n"
"          unsigned int B_row_start,\n"
"          unsigned int B_col_start,\n"
"          unsigned int B_row_inc,\n"
"          unsigned int B_col_inc,\n"
"          unsigned int B_row_size,\n"
"          unsigned int B_col_size,\n"
"          unsigned int B_internal_rows,\n"
"          unsigned int B_internal_cols,\n"
"          float beta,\n"
"          __global float * C,\n"
"          unsigned int C_row_start,\n"
"          unsigned int C_col_start,\n"
"          unsigned int C_row_inc,\n"
"          unsigned int C_col_inc,\n"
"          unsigned int C_row_size,\n"
"          unsigned int C_col_size,\n"
"          unsigned int C_internal_rows,\n"
"          unsigned int C_internal_cols) \n"
"{ \n"
"  __local float bufA[272];\n"
"  __local float bufB[272];\n"
"  size_t block_size = 16;//get_local_size(0);\n"
"  size_t row_block_id = get_group_id(0);\n"
"  size_t col_block_id = get_group_id(1);\n"
"  size_t row_thread_id = get_local_id(0);\n"
"  size_t col_thread_id = get_local_id(1);\n"
"  size_t aBegin = (row_block_id * block_size * A_row_inc + A_row_start) * A_internal_cols + A_col_start;\n"
"  size_t aStep = block_size * A_col_inc;\n"
"  size_t bBegin = (col_block_id * block_size * B_col_inc + B_col_start) + B_row_start * B_internal_cols;\n"
"  size_t bStep = block_size * B_internal_cols * B_row_inc;\n"
"  size_t block_num = (A_col_size + block_size - 1) / block_size;\n"
"  float Csub = 0;\n"
"  size_t aOffset = row_thread_id * A_row_inc + col_thread_id * A_col_inc * A_internal_cols;\n"
"  size_t bOffset = row_thread_id * B_row_inc + col_thread_id * B_col_inc * B_internal_cols;\n"
"  size_t row_thread_id_times_block_size = row_thread_id * (block_size + 1);\n"
"  size_t col_thread_id_times_block_size = col_thread_id * (block_size + 1);\n"
"  for (size_t block = 0;\n"
"           block < block_num;\n"
"           ++block)\n"
"  {\n"
"    bufA[col_thread_id_times_block_size + row_thread_id] = ((block * block_size + row_thread_id < A_col_size) && (row_block_id * block_size + col_thread_id < A_row_size)) ? A[aBegin + aOffset] : 0;\n"
"    bufB[row_thread_id_times_block_size + col_thread_id] = ((block * block_size + col_thread_id < B_row_size) && (col_block_id * block_size + row_thread_id < B_col_size)) ? B[bBegin + bOffset] : 0;\n"
"    barrier(CLK_LOCAL_MEM_FENCE);\n"
"    __local float * bufAptr = bufA + row_thread_id_times_block_size;\n"
"    __local float * bufBptr = bufB + col_thread_id_times_block_size;\n"
"      for(int i = 0; i < 4; i++) {\n"
"      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;\n"
"      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;\n"
"      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;\n"
"      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;\n"
"     }\n"
"    barrier(CLK_LOCAL_MEM_FENCE);\n"
"    aBegin += aStep;\n"
"    bBegin += bStep;\n"
"  }\n"
"  if (get_global_id(0) < A_row_size && get_global_id(1) < B_col_size)\n"
"    C[get_global_id(0) * C_row_inc + C_row_start + (get_global_id(1) * C_col_inc + C_col_start) * C_internal_rows] = alpha * Csub + beta * C[get_global_id(0) * C_row_inc + C_row_start + (get_global_id(1) * C_col_inc + C_col_start) * C_internal_rows];\n"
"}\n"
; //matrix_prod_row_row_col_align1_prod_AA

  }  //namespace kernels
 }  //namespace linalg
}  //namespace viennacl
#endif
