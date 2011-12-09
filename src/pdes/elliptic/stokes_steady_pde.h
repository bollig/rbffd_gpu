#ifndef __STOKES_STEADY_PDE_H__
#define __STOKES_STEADY_PDE_H__

#include "pdes/pde.h"

#include "utils/geom/cart2sph.h"

#include <boost/numeric/ublas/vector_sparse.hpp>
#include <boost/numeric/ublas/matrix_sparse.hpp>
#include <boost/math/special_functions/spherical_harmonic.hpp>
#include <boost/geometry.hpp>

#define VIENNACL_HAVE_UBLAS 1
#include "viennacl/linalg/bicgstab.hpp"

typedef boost::numeric::ublas::compressed_matrix<double> MatType;
typedef boost::numeric::ublas::vector<double>            VecType;

class StokesSteadyPDE : public PDE
{

    MatType *L_host; 
    MatType *div_op; 
    VecType *F_host; 
    VecType *u_host; 

    public: 
        StokesSteadyPDE(Domain* grid, RBFFD* der, Communicator* comm) 
            : PDE(grid, der, comm) 
        {
            std::cout << "INSIDE STOKES CONSTRUCTOR" << std::endl;
        }

        virtual void assemble();  
       
        virtual void solve(std::vector<SolutionType>& y, std::vector<SolutionType>* f_out, unsigned int n_stencils, unsigned int n_nodes) {

}


//        virtual void solve(std::vector<SolutionType>& y, std::vector<SolutionType>* f_out, unsigned int n_stencils, unsigned int n_nodes) 
        virtual void solve()
        {
            std::cout << "Solving...." << std::endl;
            // Solve L u = F
            // Write solution to disk
            
            // Update U_G with the content from U
            // (reshape to fit into U_G. For stokes we assume we're in a vector type <u,v,w,p>)
            // SCALAR  std::copy(u_vec.begin(), u_vec.end(), U_G.begin()); 

            // Solve system using Stabilized BiConjugate Gradients from ViennaCL
//            *u_host = viennacl::linalg::solve(*L_host, *F_host, viennacl::linalg::bicgstab_tag(1.e-24, 3000));
            *u_host = viennacl::linalg::solve(*L_host, *F_host, viennacl::linalg::bicgstab_tag(1.e-2,2));
            //*u_host = viennacl::linalg::solve(*L_host, *F_host, viennacl::linalg::bicgstab_tag(1.e-6,20));
            std::cout << "Done with solve\n"; 
            
            this->write_to_file(*u_host, "u.mtx");
            std::cout << "Wrote u_host.mtx\n"; 
        }


        virtual std::string className() { return "stokes_steady"; }

        template<typename T>
            void write_to_file(boost::numeric::ublas::vector<T> vec, std::string filename);

        // Temperature profile function (Spherical Harmonic for now)
        double Temperature (int node_indx) {
            int m = 2; 
            unsigned int l = 3; 

            NodeType& node = grid_ref.getNode(node_indx);

            // Use Boost to transform coord system
            boost::geometry::model::point<double, 3, boost::geometry::cs::cartesian> p(node.x(), node.y(), node.z());
            boost::geometry::model::point<double, 3, boost::geometry::cs::spherical<boost::geometry::radian> > p_sph;
            boost::geometry::transform(p, p_sph);

            double lambdaP_j = p_sph.get<1>();
            double thetaP_j = p_sph.get<0>();

            // See equation: http://www.boost.org/doc/libs/1_35_0/libs/math/doc/sf_and_dist/html/math_toolkit/special/sf_poly/sph_harm.html
            return boost::math::spherical_harmonic_r<double>(l, m, lambdaP_j, thetaP_j); 
        }
};

#endif  // __STOKES_STEADY_PDE_H__
