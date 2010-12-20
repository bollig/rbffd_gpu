#ifndef _NCAR_POISSON_2_CL_H_
#define _NCAR_POISSON_2_CL_H_

//#include <vector>
//#include <ArrayT.h>
#include "grids/domain_decomposition/gpu.h"
#include "exact_solutions/exact_solution.h"
#include "utils/comm/communicator.h"
#include "rbffd/derivative.h"
#include "ncar_poisson1.h"

class NCARPoisson2_CL : public NCARPoisson1
{
public:
        NCARPoisson2_CL(ExactSolution* _solution, GPU* subdomain_, Derivative* der_, int rank, int dim_num_);
        ~NCARPoisson2_CL();

        // Solve the Poisson problem
        virtual void solve(Communicator* comm_unit);
};

#endif
