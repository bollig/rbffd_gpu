
__kernel void sub(
          __global const float16 * vec1,
          __global const float16 * vec2, 
          __global float16 * result,
          unsigned int size)
{ 
  for (unsigned int i = get_global_id(0); i < size/16; i += get_global_size(0))
    result[i] = vec1[i] - vec2[i];
};


