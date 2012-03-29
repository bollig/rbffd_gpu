// TODO : test this: 
//#define CUSP_USE_TEXTURE_MEMORY

// THIS IS adapted from verbose_monitor.cu
// PROVIDED BY THE CUSP v0.1 EXAMPLES

#include "grids/grid_reader.h"
#include "rbffd/rbffd.h"
#include "timer_eb.h" 

#include <cusp/hyb_matrix.h>
#include <cusp/ell_matrix.h>
#include <cusp/csr_matrix.h>
#include <cusp/coo_matrix.h>
#include <cusp/monitor.h>
#include <cusp/krylov/cg.h>
#include <cusp/krylov/gmres.h>
#include <cusp/gallery/poisson.h>
#include <cusp/print.h>
#include <cusp/array2d.h>
#include <cusp/multiply.h>
#include <cusp/blas.h>
#include <cusp/io/matrix_market.h>
#include <cusp/precond/diagonal.h>
#include <cusp/precond/ainv.h>
#include <cusp/precond/smoothed_aggregation.h>
#include <cusp/precond/aggregate.h>
#include <cusp/precond/smooth.h>
#include <cusp/precond/strength.h>

#include <thrust/host_vector.h> 
#include <thrust/device_vector.h>
#include <thrust/generate.h>


#include "utils/spherical_harmonics.h"

#include <iomanip>
#include <iostream>
#include <sstream> 
#include <map>
#include <fstream> 
#include <typeinfo> 
using namespace std;


typedef std::vector< std::map< unsigned int, double> > STL_MAT_t; 
typedef std::vector<double> STL_VEC_t; 


typedef cusp::array1d<double, cusp::host_memory> HOST_VEC_t; 
typedef cusp::array1d<double, cusp::device_memory> DEVICE_VEC_t; 
typedef cusp::csr_matrix<unsigned int, double, cusp::host_memory> HOST_MAT_t; 
typedef cusp::csr_matrix<unsigned int, double, cusp::device_memory> DEVICE_MAT_t; 

namespace cusp
{

    class StokesSteady //: public PDE 
    {

        EB::TimerList tm;

        //---------------------------------
        public: 

        // Perform GMRES on GPU
        void GMRES_Device(DEVICE_MAT_t& A, DEVICE_VEC_t& F, DEVICE_VEC_t& U_exact, DEVICE_VEC_t& U_approx_gpu) ;
        //---------------------------------

        void assemble_System_Stokes( RBFFD& der, Grid& grid, HOST_MAT_t& A, HOST_VEC_t& F, HOST_VEC_t& U_exact);

        template <typename VecT>
            void write_to_file(VecT vec, std::string filename){
                std::ofstream fout;
                fout.open(filename.c_str());
                for (size_t i = 0; i < vec.size(); i++) {
                    fout << std::setprecision(10) << vec[i] << std::endl;
                }
                fout.close();
                std::cout << "Wrote " << filename << std::endl;
            }


        void write_System ( HOST_MAT_t& A, HOST_VEC_t& F, HOST_VEC_t& U_exact ); 
        void write_Solution( Grid& grid, HOST_VEC_t& U_exact, DEVICE_VEC_t& U_approx_gpu );

        //---------------------------------

        void gpuTest(RBFFD& der, Grid& grid, int primeGPU=0);
    }; 

}; 

