#define STRINGIFY(A) #A

std::string kernel_source = STRINGIFY(
				  
__kernel void
computeDerivKernel()
{
	/*
	* Perform an inner product of input stencil weights and solution values
	* to get an output derivative value. 
	*/
	
    size_t i =  get_global_id(0);
    size_t j =  get_global_id(1);
}

);
