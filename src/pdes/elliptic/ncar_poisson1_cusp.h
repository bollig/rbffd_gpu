#ifndef _NCAR_POISSON_1_CUSP_H_
#define _NCAR_POISSON_1_CUSP_H_

//#include <vector>
//#include <ArrayT.h>
#include "utils/comm/communicator.h"
#include "grids/domain.h"
#include "exact_solutions/exact_solution.h"
#include "rbffd/derivative.h"
#include "ncar_poisson1.h"

class NCARPoisson1_CUSP : public NCARPoisson1
{
public:
        NCARPoisson1_CUSP(ExactSolution* _solution, Domain* subdomain_, Derivative* der_, int rank, int dim_num_);
        ~NCARPoisson1_CUSP();

        // Solve the Poisson problem
        virtual void solve(Communicator* comm_unit);
};

#endif
