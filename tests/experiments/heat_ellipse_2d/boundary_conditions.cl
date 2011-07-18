#define STRINGIFY_WITH_SUBS(s) STRINGIFY(s)
#define STRINGIFY(s) #s

std::string boundary_conditions_source = STRINGIFY_WITH_SUBS(

float getBoundaryCondition(float4 node, float sol, float cur_time)
{
    // CONSTANT DIRICHLET BCs for now 
    return 0.f; 
}

);

