//helper:
void helper_norm2_parallel_reduction( __local float * tmp_buffer )
{
  for (unsigned int stride = get_global_size(0)/2; stride > 0; stride /= 2)
  {
    barrier(CLK_LOCAL_MEM_FENCE);
    if (get_global_id(0) < stride)
      tmp_buffer[get_global_id(0)] += tmp_buffer[get_global_id(0)+stride];
  }
};

////// norm_2
float float_vector1_norm_2_impl(
          __global const float * vec,
          unsigned int size,
          __local float * tmp_buffer)
{
  //step 1: fill buffer:
  float tmp = 0;
  for (unsigned int i = get_global_id(0); i < size; i += get_global_size(0))
    tmp += vec[i] * vec[i];
  tmp_buffer[get_global_id(0)] = tmp;
  
  //step 2: parallel reduction:
  helper_norm2_parallel_reduction(tmp_buffer);
  
  return tmp_buffer[0];
};

__kernel void norm_2(
          __global float * vec,
          unsigned int size,
          __local float * tmp_buffer,
          global float * result) 
{ 
  float tmp = float_vector1_norm_2_impl(vec, size, tmp_buffer);
  if (get_global_id(0) == 0) *result = sqrt(tmp);
};

