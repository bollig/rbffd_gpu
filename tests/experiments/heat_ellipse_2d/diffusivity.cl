#define STRINGIFY_WITH_SUBS(s) STRINGIFY(s)
#define STRINGIFY(s) #s

std::string diffusivity_source = STRINGIFY_WITH_SUBS(

float getDiffusionCoefficient(float4 node, float sol, float cur_time, float prev_diffusivity)
{
    // CONSTANT DIFFUSION:
    return prev_diffusivity; 
}



);
