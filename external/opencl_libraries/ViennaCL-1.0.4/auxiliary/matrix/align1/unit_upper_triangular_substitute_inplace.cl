

__kernel void unit_upper_triangular_substitute_inplace( 
          __global const float * matrix, 
          __global float * vector, 
          unsigned int row_length, 
          unsigned int size) 
{ 
  float temp; 
  for (int row = size-1; row > -1; --row) 
  { 
    barrier(CLK_GLOBAL_MEM_FENCE); 

    temp = vector[row]; 
    //eliminate column with index 'row' in parallel: 
    for  (int row_elim = get_global_id(0); row_elim < row; row_elim += get_global_size(0)) 
      vector[row_elim] -= temp * matrix[row_elim*row_length+row]; 
  } 
   
}; 

