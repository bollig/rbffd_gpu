

__kernel void vec_mul(
          __global const unsigned int * row_indices,
          __global const unsigned int * column_indices, 
          __global const float * elements,
          __global const float * vector,  
          __global float * result,
          unsigned int size) 
{ 
  for (unsigned int row = get_global_id(0); row < size; row += get_global_size(0))
  {
    float dot_prod = 0.0f;
    for (unsigned int i = row_indices[row]; i < row_indices[row+1]; ++i)
      dot_prod += elements[i] * vector[column_indices[i]];
    result[row] = dot_prod;
  }
};


