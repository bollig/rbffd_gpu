#ifndef _NCAR_POISSON_1_CL_H_
#define _NCAR_POISSON_1_CL_H_

//#include <vector>
//#include <ArrayT.h>
#include "gpu.h"
#include "exact_solution.h"
#include "communicator.h"
#include "derivative.h"
#include "ncar_poisson1.h"

class NCARPoisson1_CL : public NCARPoisson1
{
public:
        NCARPoisson1_CL(ExactSolution* _solution, GPU* subdomain_, Derivative* der_, int rank, int dim_num_);
        ~NCARPoisson1_CL();

        // Solve the Poisson problem
        virtual void solve(Communicator* comm_unit);
};

#endif
