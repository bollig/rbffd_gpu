
float norm_inf_impl(
          __global const float * vec,
          unsigned int size,
          __local float * tmp_buffer)
{
  //step 1: fill buffer:
  float tmp = 0;
  for (unsigned int i = get_global_id(0); i < size; i += get_global_size(0))
    tmp = fmax(fabs(vec[i]), tmp);
  tmp_buffer[get_global_id(0)] = tmp;
  
  //step 2: parallel reduction:
  for (unsigned int stride = get_global_size(0)/2; stride > 0; stride /= 2)
  {
    barrier(CLK_LOCAL_MEM_FENCE);
    if (get_global_id(0) < stride)
      tmp_buffer[get_global_id(0)] = max(tmp_buffer[get_global_id(0)], tmp_buffer[get_global_id(0)+stride]);
  }
  
  return tmp_buffer[0];
};

__kernel void norm_inf(
          __global float * vec,
          unsigned int size,
          __local float * tmp_buffer,
          global float * result) 
{ 
  float tmp = norm_inf_impl(vec, size, tmp_buffer);
  if (get_global_id(0) == 0) *result = tmp;
};

