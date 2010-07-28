
__kernel void lu_factorize(
          __global float * matrix,
          unsigned int matrix_row_length,
          unsigned int size) 
{ 
  float temp;
  unsigned rowi;
  unsigned rowk;
  for (unsigned int i=1; i<size; ++i)
  {
    rowi = i * matrix_row_length;
    for (unsigned int k=0; k<i; ++k)
    {
      rowk = k * matrix_row_length;
      if (get_global_id(0) == 0)
        matrix[rowi + k] /= matrix[rowk + k];

      barrier(CLK_GLOBAL_MEM_FENCE);
      temp = matrix[rowi + k];
      
      //parallel subtraction:
      for (unsigned int j=k+1 + get_global_id(0); j<size; j += get_global_size(0))
        matrix[rowi + j] -= temp * matrix[rowk + j];
    }
  }
} 


