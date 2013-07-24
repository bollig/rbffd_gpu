#ifndef VIENNACL_LINALG_KERNELS_COORDINATE_MATRIX_SOURCE_HPP_
#define VIENNACL_LINALG_KERNELS_COORDINATE_MATRIX_SOURCE_HPP_
//Automatically generated file from auxiliary-directory, do not edit manually!
/** @file coordinate_matrix_source.h
 *  @brief OpenCL kernel source file, generated automatically from scripts in auxiliary/. */
namespace viennacl
{
 namespace linalg
 {
  namespace kernels
  {
const char * const coordinate_matrix_align1_row_info_extractor = 
"__kernel void row_info_extractor(\n"
"          __global const uint2 * coords, //(row_index, column_index)\n"
"          __global const float * elements,\n"
"          __global const uint  * group_boundaries,\n"
"          __global float * result,\n"
"          unsigned int option,\n"
"          __local unsigned int * shared_rows,\n"
"          __local float * inter_results)\n"
"{\n"
"  uint2 tmp;\n"
"  float val;\n"
"  uint last_index  = get_local_size(0) - 1;\n"
"  uint group_start = group_boundaries[get_group_id(0)];\n"
"  uint group_end   = group_boundaries[get_group_id(0) + 1];\n"
"  uint k_end = (group_end > group_start) ? 1 + (group_end - group_start - 1) / get_local_size(0) : 0;   // -1 in order to have correct behavior if group_end - group_start == j * get_local_size(0)\n"
"  uint local_index = 0;\n"
"  for (uint k = 0; k < k_end; ++k)\n"
"  {\n"
"    local_index = group_start + k * get_local_size(0) + get_local_id(0);\n"
"    tmp = (local_index < group_end) ? coords[local_index] : (uint2) 0;\n"
"    val = (local_index < group_end && (option != 3 || tmp.x == tmp.y) ) ? elements[local_index] : 0;\n"
"    //check for carry from previous loop run:\n"
"    if (get_local_id(0) == 0 && k > 0)\n"
"    {\n"
"      if (tmp.x == shared_rows[last_index])\n"
"      {\n"
"        switch (option)\n"
"        {\n"
"          case 0: //inf-norm\n"
"          case 3: //diagonal entry\n"
"            val = max(val, fabs(inter_results[last_index]));\n"
"            break;\n"
"          case 1: //1-norm\n"
"            val = fabs(val) + inter_results[last_index];\n"
"            break;\n"
"          case 2: //2-norm\n"
"            val = sqrt(val * val + inter_results[last_index]);\n"
"            break;\n"
"          default:\n"
"            break;\n"
"        }\n"
"      }\n"
"      else\n"
"      {\n"
"        switch (option)\n"
"        {\n"
"          case 0: //inf-norm\n"
"          case 1: //1-norm\n"
"          case 3: //diagonal entry\n"
"            result[shared_rows[last_index]] = inter_results[last_index];\n"
"            break;\n"
"          case 2: //2-norm\n"
"            result[shared_rows[last_index]] = sqrt(inter_results[last_index]);\n"
"          default:\n"
"            break;\n"
"        }\n"
"      }\n"
"    }\n"
"    //segmented parallel reduction begin\n"
"    barrier(CLK_LOCAL_MEM_FENCE);\n"
"    shared_rows[get_local_id(0)] = tmp.x;\n"
"    switch (option)\n"
"    {\n"
"      case 0:\n"
"      case 3:\n"
"        inter_results[get_local_id(0)] = val;\n"
"        break;\n"
"      case 1:\n"
"        inter_results[get_local_id(0)] = fabs(val);\n"
"        break;\n"
"      case 2:\n"
"        inter_results[get_local_id(0)] = val * val;\n"
"      default:\n"
"        break;\n"
"    }\n"
"    float left = 0;\n"
"    barrier(CLK_LOCAL_MEM_FENCE);\n"
"    for (unsigned int stride = 1; stride < get_local_size(0); stride *= 2)\n"
"    {\n"
"      left = (get_local_id(0) >= stride && tmp.x == shared_rows[get_local_id(0) - stride]) ? inter_results[get_local_id(0) - stride] : 0;\n"
"      barrier(CLK_LOCAL_MEM_FENCE);\n"
"      switch (option)\n"
"      {\n"
"        case 0: //inf-norm\n"
"        case 3: //diagonal entry\n"
"          inter_results[get_local_id(0)] = max(inter_results[get_local_id(0)], left);\n"
"          break;\n"
"        case 1: //1-norm\n"
"          inter_results[get_local_id(0)] += left;\n"
"          break;\n"
"        case 2: //2-norm\n"
"          inter_results[get_local_id(0)] += left;\n"
"          break;\n"
"        default:\n"
"          break;\n"
"      }\n"
"      barrier(CLK_LOCAL_MEM_FENCE);\n"
"    }\n"
"    //segmented parallel reduction end\n"
"    if (get_local_id(0) != last_index &&\n"
"        shared_rows[get_local_id(0)] != shared_rows[get_local_id(0) + 1] &&\n"
"        inter_results[get_local_id(0)] != 0)\n"
"    {\n"
"      result[tmp.x] = (option == 2) ? sqrt(inter_results[get_local_id(0)]) : inter_results[get_local_id(0)];\n"
"    }\n"
"    barrier(CLK_LOCAL_MEM_FENCE);\n"
"  } //for k\n"
"  if (get_local_id(0) == last_index && inter_results[last_index] != 0)\n"
"    result[tmp.x] = (option == 2) ? sqrt(inter_results[last_index]) : inter_results[last_index];\n"
"}\n"
; //coordinate_matrix_align1_row_info_extractor

const char * const coordinate_matrix_align1_vec_mul = 
"__kernel void vec_mul(\n"
"          __global const uint2 * coords, //(row_index, column_index)\n"
"          __global const float * elements,\n"
"          __global const uint  * group_boundaries,\n"
"          __global const float * vector,\n"
"          __global float * result,\n"
"          __local unsigned int * shared_rows,\n"
"          __local float * inter_results)\n"
"{\n"
"  uint2 tmp;\n"
"  float val;\n"
"  uint last_index  = get_local_size(0) - 1;\n"
"  uint group_start = group_boundaries[get_group_id(0)];\n"
"  uint group_end   = group_boundaries[get_group_id(0) + 1];\n"
"  uint k_end = (group_end > group_start) ? 1 + (group_end - group_start - 1) / get_local_size(0) : 0;   // -1 in order to have correct behavior if group_end - group_start == j * get_local_size(0)\n"
"  uint local_index = 0;\n"
"  for (uint k = 0; k < k_end; ++k)\n"
"  {\n"
"    local_index = group_start + k * get_local_size(0) + get_local_id(0);\n"
"    tmp = (local_index < group_end) ? coords[local_index] : (uint2) 0;\n"
"    val = (local_index < group_end) ? elements[local_index] * vector[tmp.y] : 0;\n"
"    //check for carry from previous loop run:\n"
"    if (get_local_id(0) == 0 && k > 0)\n"
"    {\n"
"      if (tmp.x == shared_rows[last_index])\n"
"        val += inter_results[last_index];\n"
"      else\n"
"        result[shared_rows[last_index]] = inter_results[last_index];\n"
"    }\n"
"    //segmented parallel reduction begin\n"
"    barrier(CLK_LOCAL_MEM_FENCE);\n"
"    shared_rows[get_local_id(0)] = tmp.x;\n"
"    inter_results[get_local_id(0)] = val;\n"
"    float left = 0;\n"
"    barrier(CLK_LOCAL_MEM_FENCE);\n"
"    for (unsigned int stride = 1; stride < get_local_size(0); stride *= 2)\n"
"    {\n"
"      left = (get_local_id(0) >= stride && tmp.x == shared_rows[get_local_id(0) - stride]) ? inter_results[get_local_id(0) - stride] : 0;\n"
"      barrier(CLK_LOCAL_MEM_FENCE);\n"
"      inter_results[get_local_id(0)] += left;\n"
"      barrier(CLK_LOCAL_MEM_FENCE);\n"
"    }\n"
"    //segmented parallel reduction end\n"
"    if (get_local_id(0) != last_index &&\n"
"        shared_rows[get_local_id(0)] != shared_rows[get_local_id(0) + 1] &&\n"
"        inter_results[get_local_id(0)] != 0)\n"
"    {\n"
"      result[tmp.x] = inter_results[get_local_id(0)];\n"
"    }\n"
"    barrier(CLK_LOCAL_MEM_FENCE);\n"
"  } //for k\n"
"  if (get_local_id(0) == last_index && inter_results[last_index] != 0)\n"
"    result[tmp.x] = inter_results[last_index];\n"
"}\n"
; //coordinate_matrix_align1_vec_mul

  }  //namespace kernels
 }  //namespace linalg
}  //namespace viennacl
#endif

