#ifndef __STOKES_STEADY_PDE_H__
#define __STOKES_STEADY_PDE_H__

#include "utils/spherical_harmonics.h"

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


#if 0
// This type is SLOW to assemble. The compressed works faster, but is slow in a product
typedef boost::numeric::ublas::coordinate_matrix<double>                        MatType;
#else
typedef boost::numeric::ublas::compressed_matrix<double>                        MatType;
#endif 
typedef boost::numeric::ublas::vector<double>                                   VecType;
typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::undirectedS,
        boost::property<boost::vertex_color_t, boost::default_color_type,
        boost::property<boost::vertex_degree_t, size_t> > >                     GraphType;
typedef boost::graph_traits<GraphType>::vertex_descriptor                       VertexType;
typedef boost::graph_traits<GraphType>::vertices_size_type                      size_type;


class StokesSteadyPDE : public PDE
{
    unsigned int N; 
    unsigned int nrows; 
    unsigned int ncols; 
    unsigned int NNZ; 

    // See matching variable names for use in rbf_prototypes.git
    MatType* LHS; 
    VecType* RHS_continuous; 
    VecType* RHS_discrete; 
    VecType* U_continuous; 
    VecType* U_computed; 
    VecType* eta; 
    MatType* DIV_operator; 

    // For Cuthill Mckee reordering
    MatType* LHS_reordered; 
    GraphType* LHS_graph; 
    VecType *RHS_reordered; 
    VecType *U_reordered; 

    // Lookup table to find our reorderings
    boost::numeric::ublas::vector<size_t> *inv_m_lookup; 
    boost::numeric::ublas::vector<size_t> *m_lookup; 

    bool constantViscosity;
    bool discreteRHS;
    bool cuthillMckeeReordering;

    public: 
    StokesSteadyPDE(Domain* grid, RBFFD* der, Communicator* comm) 
        : PDE(grid, der, comm), 
        constantViscosity(true), discreteRHS(true),
        cuthillMckeeReordering(false)
    {
        std::cout << "INSIDE STOKES CONSTRUCTOR" << std::endl;
        setupTimers();
    }

    virtual void setupTimers();

    virtual void assemble();  

    virtual void solve(std::vector<SolutionType>& y, std::vector<SolutionType>* f_out, unsigned int n_stencils, unsigned int n_nodes) {

    }

    virtual void solve();
    virtual void solve_reordered();
    virtual void solve_original();
    virtual void getDivOperator();
    virtual void reorder();
    virtual void writeToFile(); 

    virtual std::string className() { return "stokes_steady"; }

    template<typename T>
        void write_to_file(boost::numeric::ublas::vector<T> vec, std::string filename);

    // Temperature profile function (Spherical Harmonic for now)
    double Temperature (int node_indx) {
        int m = 2; 
        int l = 3; 

        NodeType& node = grid_ref.getNode(node_indx);

        // Use Boost to transform coord system
        boost::geometry::model::point<double, 3, boost::geometry::cs::cartesian> p(node.x(), node.y(), node.z());
        boost::geometry::model::point<double, 3, boost::geometry::cs::spherical<boost::geometry::radian> > p_sph;
        boost::geometry::transform(p, p_sph);

        double lambdaP_j = p_sph.get<1>();
        double thetaP_j = p_sph.get<0>();

        // See equation: http://www.boost.org/doc/libs/1_35_0/libs/math/doc/sf_and_diy2st/html/math_toolkit/special/sf_poly/sph_harm.html
        // NOTE: the extra scale of sqrt(2) is to be consistent with the sph(...) routine in MATLAB
        return sqrt(2)*boost::math::spherical_harmonic_r<double>(l, m, lambdaP_j, thetaP_j); 
    }

    protected:
    virtual void fillLHS_ConstViscosity();  
    //    virtual void fillLHS_VarViscosity();  

    virtual void fillRHS_ConstViscosity();  
    virtual void fillRHS_discrete();  
    //    virtual void fillRHS_VarViscosity();  

    void build_graph(MatType& mat, GraphType& G);
    void get_cuthill_mckee_order(GraphType& G, boost::numeric::ublas::vector<size_t>& lookup_chart);
    void get_reordered_system(MatType& in_mat, VecType& in_vec, boost::numeric::ublas::vector<size_t>& order, MatType& out_mat, VecType& out_vec);
    void get_original_order(VecType& in_vec, VecType& out_vec);


};

#endif  // __STOKES_STEADY_PDE_H__
