
__kernel void unit_lower_triangular_substitute_inplace(
          __global const float * matrix,
          __global float * vector,
          unsigned int row_length,
          unsigned int size)
{
  float temp;
  for (int row = 0; row < size; ++row)
  {
    barrier(CLK_GLOBAL_MEM_FENCE);
    temp = vector[row];

    for  (int row_elim = row + get_global_id(0) + 1; row_elim < size; row_elim += get_global_size(0))
      vector[row_elim] -= temp * matrix[row_elim*row_length+row];
  }
};


