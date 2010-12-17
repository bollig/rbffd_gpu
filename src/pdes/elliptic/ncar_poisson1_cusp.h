#ifndef _NCAR_POISSON_1_CUSP_H_
#define _NCAR_POISSON_1_CUSP_H_

//#include <vector>
//#include <ArrayT.h>
#include "grids/domain_decomposition/gpu.h"
#include "exact_solutions/exact_solution.h"
#include "utils/comm/communicator.h"
#include "rbffd/derivative.h"
#include "ncar_poisson1.h"

class NCARPoisson1_CUSP : public NCARPoisson1
{
public:
        NCARPoisson1_CUSP(ExactSolution* _solution, GPU* subdomain_, Derivative* der_, int rank, int dim_num_);
        ~NCARPoisson1_CUSP();

        // Solve the Poisson problem
        virtual void solve(Communicator* comm_unit);
};

#endif
