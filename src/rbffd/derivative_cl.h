#ifndef _DERIVATIVE_CL_H_
#define _DERIVATIVE_CL_H_

#include <stdlib.h>
#include <iostream>
#include <vector>
#include "derivative.h"
#include "utils/opencl/cl_base_class.h"
#include <CL/cl.hpp>

class DerivativeCL : public Derivative, public CLBaseClass
{
    size_t total_num_stencil_elements; 
	// These are pointers to gpu memory. We will need to allocate in the constructor and 
	// copy to/from the memory in computeDeriv
	cl::Buffer gpu_stencils; 
	cl::Buffer gpu_solution;

	cl::Buffer gpu_x_deriv_weights; 
	cl::Buffer gpu_y_deriv_weights; 
	cl::Buffer gpu_z_deriv_weights;
	cl::Buffer gpu_laplacian_weights;

	cl::Buffer gpu_derivative_out; 
public:
    DerivativeCL(std::vector<NodeType>& rbf_centers_, std::vector<StencilType>& stencil_, int nb_bnd_pts, int dim_num, int rank=0);
    ~DerivativeCL(); 

    // u : take derivative of this scalar variable (already allocated)
    // deriv : resulting derivative (already allocated)
    // which : which derivative (X, Y, LAPL)
    // This overrides the CPU equivalents to provide a GPU accelerated routine (using OpenCL)
    virtual void computeDerivatives(DerType which, double* u, double* deriv, int npts);

};

#endif
