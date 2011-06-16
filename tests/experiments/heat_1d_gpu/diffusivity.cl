#define STRINGIFY_WITH_SUBS(s) STRINGIFY(s)
#define STRINGIFY(s) #s

std::string diffusivity_source = STRINGIFY_WITH_SUBS(

float getDiffusionCoefficient(float3 node, float sol, float cur_time, float prev_diffusivity)
{
    // CONSTANT DIFFUSION:
    return prev_diffusivity; 
}


__kernel void fillDiffusivity(
    __global float3* node_list, 
    __global float* cur_solution, 
             float cur_time,
             int nb_stencils,
    __global float* diffusivity)
{
    size_t i = get_global_id(0);    \n
    if(i < nb_stencils) {    \n
        diffusivity[i] = getDiffusionCoefficient(node_list[i], cur_solution[i], cur_time, diffusivity[i]);

    }
}



);
