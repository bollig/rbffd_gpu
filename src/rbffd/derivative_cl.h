#ifndef _DERIVATIVE_CL_H_
#define _DERIVATIVE_CL_H_

#include <stdlib.h>
#include <iostream>
#include <vector>
#include "derivative.h"
#include <CL/cl.hpp>

class DerivativeCL : public Derivative
{

public:
    DerivativeCL(cl::Context context, std::vector<Vec3>& rbf_centers_, std::vector<std::vector<int> >& stencil_, int nb_bnd_pts, int dim_num);
    ~DerivativeCL(); 

    // u : take derivative of this scalar variable (already allocated)
    // deriv : resulting derivative (already allocated)
    // which : which derivative (X, Y, LAPL)
    // This overrides the CPU equivalents to provide a GPU accelerated routine (using OpenCL)
    virtual void computeDeriv(DerType which, double* u, double* deriv, int npts);
};

#endif
