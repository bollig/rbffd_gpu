#ifndef _NonUniform_POISSON_1_CL_H_
#define _NonUniform_POISSON_1_CL_H_

//#include <vector>
//#include <ArrayT.h>
#include "gpu.h"
#include "exact_solution.h"
#include "communicator.h"
#include "derivative.h"
#include "ncar_poisson1.h"
#include "projectsettings.h"
#include <boost/numeric/ublas/vector.hpp>

// Set single or double precision here.
#if 1
typedef double FLOAT;
#else
typedef float FLOAT;
#endif

typedef boost::numeric::ublas::compressed_matrix<FLOAT> MatType;
typedef boost::numeric::ublas::vector<FLOAT>            VecType;
typedef std::vector<int>                                StencilType;
typedef std::vector<StencilType >                       StencilListType;
typedef Vec3                                            CenterType;
typedef std::vector<CenterType >                         CenterListType;

class NonUniformPoisson1_CL : public NCARPoisson1
{
public:
        NonUniformPoisson1_CL(ExactSolution* _solution, GPU* subdomain_, Derivative* der_, int rank, int dim_num_);
        NonUniformPoisson1_CL(ProjectSettings* _settings, ExactSolution* _solution, GPU* subdomain_, Derivative* der_, int rank, int dim_num_);
        ~NonUniformPoisson1_CL();

        // Solve the Poisson problem
        virtual void solve(Communicator* comm_unit);

        int fillBoundaryDirichlet(MatType& L, VecType& F, StencilType& stencil, CenterListType& centers);
        int fillBoundaryNeumann(MatType& L, VecType& F, StencilType& stencil, CenterListType& centers);
        int fillBoundaryRobin(MatType& L, VecType& F, StencilType& stencil, CenterListType& centers);


        void fillSolutionConstraint(MatType& L, VecType& F, StencilListType& stencils, CenterListType& centers, int nb, int ni);
        void fillSolutionNoConstraint(MatType& L, VecType& F, StencilListType& stencils, CenterListType& centers, int nb, int ni);

        int fillInterior(MatType& L, VecType& F, StencilType& stencil, CenterListType& centers);

protected:
        int fillInteriorLHS(MatType& L, VecType& F, StencilType& stencil, CenterListType& centers);
        void fillInteriorRHS(MatType& L, VecType& F, StencilType& stencil, CenterListType& centers);



        template<typename T>
        void write_to_file(boost::numeric::ublas::vector<T> vec, std::string filename);
};

#endif
