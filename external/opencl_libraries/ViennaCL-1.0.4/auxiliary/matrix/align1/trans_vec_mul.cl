

__kernel void trans_vec_mul(
          __global const float * matrix,
          __global const float * vector,  
          __global float * result,
          unsigned int matrix_row_length, //keep transpose operation in mind!
          unsigned int vector_length, //keep transpose operation in mind!
          unsigned int result_length) 
{ 
  for (unsigned int row = get_global_id(0); row < result_length; row += get_global_size(0))
  {
    float dot_prod2 = 0.0f;
    for (unsigned int col = 0; col < vector_length; ++col)
      dot_prod2 += matrix[row + col*matrix_row_length] * vector[col];
    result[row] = dot_prod2;
  }
};


