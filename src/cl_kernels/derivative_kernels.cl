#define STRINGIFY(A) #A

std::string kernel_source = STRINGIFY(
				  
__kernel void
hello()
{
	/*
	Just a stub kernel. 
	*/
	
    size_t i =  get_global_id(0);
    size_t j =  get_global_id(1);
}

);
