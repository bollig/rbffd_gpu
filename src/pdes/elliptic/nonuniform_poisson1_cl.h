#ifndef __NonUniform_POISSON_1_CL_H__
#define __NonUniform_POISSON_1_CL_H__

#include "utils/comm/communicator.h"
#include "grids/domain.h"
#include "exact_solutions/exact_solution.h"
#include "rbffd/rbffd.h"
#include "pdes/elliptic/ncar_poisson1.h"
#include "utils/conf/projectsettings.h"
#include <boost/numeric/ublas/vector.hpp>

#include "common_typedefs.h"

typedef boost::numeric::ublas::compressed_matrix<FLOAT> MatType;
typedef boost::numeric::ublas::vector<FLOAT>            VecType;
typedef std::vector<StencilType >                       StencilListType;
typedef Vec3                                            CenterType;
typedef std::vector<CenterType >                         CenterListType;

class NonUniformPoisson1_CL : public NCARPoisson1
{
    protected: 
        MatType* L_host_ptr; 
        VecType* F_host_ptr;
public:
        NonUniformPoisson1_CL(ExactSolution* _solution, Grid* subdomain_, RBFFD* der_, int rank, int dim_num_);
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
        void fillBoundaryDiscreteNeumannRHS(VecType& F, StencilType& stencil, CenterListType& centers);
        void fillBoundaryContinuousNeumannRHS(VecType& F, StencilType& stencil, CenterListType& centers);

        void fillBoundaryNeumannRHS(VecType& F, StencilType& stencil, CenterListType& centers);
        void fillBoundaryRobinRHS(VecType& F, StencilType& stencil, CenterListType& centers);



        template<typename T>
        void write_to_file(boost::numeric::ublas::vector<T> vec, std::string filename);
};

#endif
