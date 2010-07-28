
//perform a rank-1 update of the matrix, i.e. A += x * x^T
__kernel void rank1_update(
          __global float * matrix,
          __global const float * vector1,  
          __global const float * vector2,  
          unsigned int matrix_row_length,
          unsigned int row_length,
          unsigned int col_length) 
{ 
  float tmp;
  unsigned int offset;

  for (unsigned int row = get_global_id(0); row < row_length; row += get_global_size(0))
  {
    tmp = vector1[row];
    offset = row*matrix_row_length;
    for (unsigned int col = 0; col < col_length; ++col)
      matrix[offset+col] += tmp * vector2[col];
  }
};

