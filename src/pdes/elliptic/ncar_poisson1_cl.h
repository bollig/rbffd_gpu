#ifndef _NCAR_POISSON_1_CL_H_
#define _NCAR_POISSON_1_CL_H_

//#include <vector>
//#include <ArrayT.h>
#include "grids/domain_decomposition/gpu.h"
#include "exact_solutions/exact_solution.h"
#include "utils/comm/communicator.h"
#include "rbffd/derivative.h"
#include "ncar_poisson1.h"
#include "utils/conf/projectsettings.h"
#include <boost/numeric/ublas/vector.hpp>

class NCARPoisson1_CL : public NCARPoisson1
{
public:
        NCARPoisson1_CL(ExactSolution* _solution, GPU* subdomain_, Derivative* der_, int rank, int dim_num_);
        NCARPoisson1_CL(ProjectSettings* _settings, ExactSolution* _solution, GPU* subdomain_, Derivative* der_, int rank, int dim_num_);
        ~NCARPoisson1_CL();

        // Solve the Poisson problem (Overrides NCARPoisson1::solve())
        virtual void solve(Communicator* comm_unit);

        template<typename T>
        void write_to_file(boost::numeric::ublas::vector<T> vec, std::string filename);
};

#endif
