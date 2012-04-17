// Solve a very simple FD system (1-D) to match results with Matlab
// If the solution for a 10x10 system is the same as far as iterations, residual, error etc. 
//      We check ILU preconditioner
// else We know the difference in solution is from GMRES 
//
// Conclusion: 
//      Our C++ and MATLAB code converge exactly the same for this system. The
//      preconditioner is also exactly the same. Therefore, it must be
//      conditioning related issues that cause differences in our iterations
//      for RBFFD 

#include <viennacl/compressed_matrix.hpp>
#include <viennacl/coordinate_matrix.hpp>
#include <viennacl/linalg/ilu.hpp>
#include <viennacl/linalg/jacobi_precond.hpp>
#include <viennacl/linalg/row_scaling.hpp>
#if 0
// TODO: SPAI and AMG (experimental in VCL 1.2.0 and didnt work for us in CUSP
#endif 
#include <viennacl/linalg/gmres.hpp>
#include <viennacl/linalg/norm_1.hpp>
#include <viennacl/linalg/norm_2.hpp>
#include <viennacl/linalg/norm_inf.hpp>
#include <viennacl/linalg/prod.hpp> 
#include <viennacl/io/matrix_market.hpp>
#include <viennacl/matrix.hpp>
#include <viennacl/vector.hpp> 
#include <viennacl/vector_proxy.hpp> 
#include <viennacl/linalg/vector_operations.hpp> 

#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_sparse.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/numeric/ublas/banded.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/vector_proxy.hpp>
#include <boost/numeric/ublas/vector_sparse.hpp>
#include <boost/numeric/ublas/vector_of_vector.hpp>
#include <boost/numeric/ublas/vector_expression.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/operation.hpp>
#include <boost/numeric/ublas/lu.hpp>

#include "precond/ilu0.hpp"

#include <iomanip>
#include <iostream>
#include <sstream> 
#include <map>
#include <fstream> 
#include <typeinfo> 
using namespace std;


typedef std::vector< std::map< unsigned int, double> > STL_MAT_t; 
typedef boost::numeric::ublas::compressed_matrix<double> UBLAS_MAT_t; 
typedef boost::numeric::ublas::coordinate_matrix<double> UBLAS_ALT_MAT_t; 
typedef viennacl::compressed_matrix<double> VCL_MAT_t; 
typedef viennacl::coordinate_matrix<double> VCL_ALT_MAT_t; 

typedef std::vector<double> STL_VEC_t; 
typedef boost::numeric::ublas::vector<double> UBLAS_VEC_t; 
typedef viennacl::vector<double> VCL_VEC_t; 


int main(int argc, char** argv) {
    // vandermond matrix test. PASS
    UBLAS_MAT_t AA(11,11,33); 
    AA(0,0) = 2;   AA(0,1) = -1; 
    AA(1,0) = -1;   AA(1,1) = 2;   AA(1,2) = -1;  
    AA(2,1) = -1;   AA(2,2) = 2;   AA(2,3) = -1;  
    AA(3,2) = -1;   AA(3,3) = 2;   AA(3,4) = -1;  
    AA(4,3) = -1;   AA(4,4) = 2;   AA(4,5) = -1;  
    AA(5,4) = -1;   AA(5,5) = 2;   AA(5,6) = -1;  
    AA(6,5) = -1;   AA(6,6) = 2;   AA(6,7) = -1;  
    AA(7,6) = -1;   AA(7,7) = 2;   AA(7,8) = -1;  
    AA(8,7) = -1;   AA(8,8) = 2;   AA(8,9) = -1;  
    AA(9,8) = -1;   AA(9,9) = 2;   AA(9,10) = -1;  
    AA(10,9) = -1;   AA(10,10) = 2;  

    UBLAS_VEC_t FF(11, 1); 

    VCL_MAT_t AA_gpu(AA.size1(), AA.size2(), AA.nnz()); 
    VCL_VEC_t FF_gpu(FF.size()); 

    std::cout << "Copying to GPU\n";

    copy(AA, AA_gpu);
    copy(FF, FF_gpu);

    std::cout << "ABOUT TO PRECOND\n" << std::endl;

    viennacl::linalg::ilu0_precond< VCL_MAT_t > vcl_ilu( AA_gpu, viennacl::linalg::ilu0_tag() ); 
    viennacl::io::write_matrix_market_file(vcl_ilu.LU,"output/ILU.mtx"); 

    viennacl::linalg::gmres_tag tag(1e-6, 100, 5); 
    VCL_VEC_t U_gpu = viennacl::linalg::solve(AA_gpu, FF_gpu, tag);  

    std::cout << "GMRES Iterations: " << tag.iters() << std::endl;
    std::cout << "GMRES Error Estimate: " << tag.error() << std::endl;
    std::cout << "GMRES Krylov Dim: " << tag.krylov_dim() << std::endl;
    std::cout << "GMRES Max Number of Restarts (max_iter/krylov_dim): " << tag.max_restarts() << std::endl;
    std::cout << "GMRES Max Number of Iterations: " << tag.max_iterations() << std::endl;
    std::cout << "GMRES Tolerance: " << tag.tolerance() << std::endl;

    std::cout << "FF = " << FF_gpu << std::endl;
    std::cout << "UU = " << U_gpu << std::endl;

    U_gpu = viennacl::linalg::solve(AA_gpu, FF_gpu, tag, vcl_ilu);  

    std::cout << "GMRES Iterations: " << tag.iters() << std::endl;
    std::cout << "GMRES Error Estimate: " << tag.error() << std::endl;
    std::cout << "GMRES Krylov Dim: " << tag.krylov_dim() << std::endl;
    std::cout << "GMRES Max Number of Restarts (max_iter/krylov_dim): " << tag.max_restarts() << std::endl;
    std::cout << "GMRES Max Number of Iterations: " << tag.max_iterations() << std::endl;
    std::cout << "GMRES Tolerance: " << tag.tolerance() << std::endl;

    std::cout << "FF = " << FF_gpu << std::endl;
    std::cout << "UU = " << U_gpu << std::endl;


    // ----------- TO COMPARE WITH MATLAB -----------
    //  % Construct System
    //  e = ones(11, 1)
    //  AA = spdiags([-e 2*e -e], -1:1, 11)
    //  FF = ones(11, 1)
    //  
    //  % Find the desired solution
    //  UU = AA \ FF;
    //
    //  % Check Convergence without preconditioner
    //  UU = gmres(AA, FF)
    //
    //  % Specify same constraints as C++ and make sure we converge in SAME number of iterations
    //  % NOTE: total iterations = (outer_iter-1)*restart + inner_iter
    //  UU = gmres(AA, FF, 5, 1e-6, 100)
    //
    //  % Make sure the preconditioner is the same (ill conditioning would cause this to vary
    //  options.type='nofill'
    //  options.milu='off'
    //  [L1, U1] = ilu(AA, options)
    //
    //  % Make sure preconditioned solve converges in SAME number of iters
    //  UU = gmres(AA, FF, 5, 1e-6, 100, L1, U1)
    //

    return 0; 
}
