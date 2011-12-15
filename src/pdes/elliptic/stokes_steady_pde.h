#ifndef __STOKES_STEADY_PDE_H__
#define __STOKES_STEADY_PDE_H__

#include "pdes/pde.h"

#include "utils/geom/cart2sph.h"

#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/vector_sparse.hpp>
#include <boost/numeric/ublas/matrix_sparse.hpp>
#include <boost/math/special_functions/spherical_harmonic.hpp>
#include <boost/geometry.hpp>
#include <boost/numeric/ublas/lu.hpp>
#include <boost/numeric/ublas/triangular.hpp>
#include <boost/graph/cuthill_mckee_ordering.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/properties.hpp>
#include <boost/graph/bandwidth.hpp>

#define VIENNACL_HAVE_UBLAS 1
#include "viennacl/linalg/bicgstab.hpp"
#include "viennacl/linalg/gmres.hpp"
//
// ViennaCL includes
//
#include "viennacl/scalar.hpp"
#include "viennacl/vector.hpp"
#include "viennacl/matrix.hpp"
#include "viennacl/linalg/ilu.hpp"
#include "viennacl/linalg/jacobi_precond.hpp"
#include "viennacl/linalg/direct_solve.hpp"
#include "viennacl/linalg/prod.hpp"       //generic matrix-vector product
#include "viennacl/linalg/norm_2.hpp"     //generic l2-norm for vectors

using namespace boost::numeric; 

typedef boost::numeric::ublas::compressed_matrix<double> MatType;
typedef boost::numeric::ublas::vector<double>            VecType;
typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::undirectedS,
        boost::property<boost::vertex_color_t, boost::default_color_type,
        boost::property<boost::vertex_degree_t, size_t> > > GraphType;
typedef boost::graph_traits<GraphType>::vertex_descriptor VertexType;
typedef boost::graph_traits<GraphType>::vertices_size_type size_type;


class StokesSteadyPDE : public PDE
{

    MatType *L_host; 
    MatType *L_reordered; 
    GraphType *L_graph; 
    MatType div_op; 
    VecType *F_host; 
    VecType *F_reordered; 
    VecType *u_host; 
    VecType *u_reordered; 
    
    // Lookup table to find our reorderings
    boost::numeric::ublas::vector<size_t> *inv_m_lookup; 
    boost::numeric::ublas::vector<size_t> *m_lookup; 

    public: 
        StokesSteadyPDE(Domain* grid, RBFFD* der, Communicator* comm) 
            : PDE(grid, der, comm) 
        {
            std::cout << "INSIDE STOKES CONSTRUCTOR" << std::endl;
        }

        virtual void assemble();  
       
        void build_graph(MatType& mat, GraphType& G);
        void get_cuthill_mckee_order(GraphType& G, boost::numeric::ublas::vector<size_t>& lookup_chart);
        void get_reordered_system(MatType& in_mat, VecType& in_vec, boost::numeric::ublas::vector<size_t>& order, MatType& out_mat, VecType& out_vec);
        void get_original_order(VecType& in_vec, VecType& out_vec);

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
            // *u_host = viennacl::linalg::solve(*L_host, *F_host, viennacl::linalg::bicgstab_tag(1.e-24, 3000));

            // NOTE: the preconditioners dont work on our current structure. We need to
            // reorder the matrix to get entries on the diagonal in order for ilut to work. Im
            // sure this is why the jacobi also fails. 
#if 0
            viennacl::linalg::ilut_precond< MatType >  ublas_ilut(*L_host, viennacl::linalg::ilut_tag());
            *u_host = viennacl::linalg::solve(*L_host, *F_host, viennacl::linalg::gmres_tag(1e-6, 20), ublas_ilut);
#else 
#if 0
            viennacl::linalg::jacobi_precond< MatType > ublas_jacobi(*L_host, viennacl::linalg::jacobi_tag());
            *u_host = viennacl::linalg::solve(*L_host, *F_host, viennacl::linalg::gmres_tag(1e-6, 20), ublas_jacobi);
#else 
#if 1
            *u_reordered = viennacl::linalg::solve(*L_reordered, *F_reordered, viennacl::linalg::gmres_tag(1e-12, 40));
#endif 
#endif 
#endif 
        #if 0
            // LU with Partial Pivoting
            *u_host = *F_host; 
            ublas::permutation_matrix<double> P1(L_host->size1());

            ublas::lu_factorize(*L_host, P1);
            ublas::lu_substitute(*L_host, P1, *u_host);
        #else 

        #endif 

            std::cout << "Done with solve\n"; 

            this->write_to_file(*u_reordered, "u_reordered.mtx");

            this->get_original_order(*u_reordered, *u_host);
            
            this->write_to_file(*u_host, "u.mtx");

            this->write_to_file(VecType(prod(div_op, *u_host)), "div.mtx"); 
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
