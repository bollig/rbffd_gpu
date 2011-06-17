#define STRINGIFY_WITH_SUBS(s) STRINGIFY(s)
#define STRINGIFY(s) #s

std::string kernel_source = STRINGIFY_WITH_SUBS(

__kernel void           \n
enforceBoundaryConditions(
         __global FLOAT* solution_in_out,  \n
         __global int* bndry_indices,
                   int nb_bnd, \n
                   float cur_time \n
)  \n
{   \n
    uint i = get_global_id(0);    \n
\n
    if(i < nb_bnd) {    \n
        int j = bndry_indices[i];

        // FIXME: we should pass in the node list and be able to precisely determine our BC by node
        float4 dummy_node = (float4)(0.f, 0.f, 0.f, 0.f);
        FLOAT bc = getBoundaryCondition(dummy_node, solution_in_out[j], cur_time);

        solution_in_out[j] =  bc;
    }\n
}\n

);

