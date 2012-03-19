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

EB::TimerList tm;

//---------------------------------

// Perform GMRES on CPU
void GMRES_Host(HOST_MAT_t& A, HOST_VEC_t& F, HOST_VEC_t& U_exact) {
#if 0
    HOST_VEC_t U_approx(U_exact.size());
    viennacl::linalg::gmres_tag tag(1e-8, 100); 
    U_approx = viennacl::linalg::solve(A, U_exact, tag); 

    std::cout << "Rel l1   Norm: " << boost::numeric::ublas::norm_1(U_approx - U_exact) / boost::numeric::ublas::norm_1(U_exact) << std::endl;  
    std::cout << "Rel l2   Norm: " << boost::numeric::ublas::norm_2(U_approx - U_exact) / boost::numeric::ublas::norm_2(U_exact) << std::endl;  
    std::cout << "Rel linf Norm: " << boost::numeric::ublas::norm_inf(U_approx - U_exact) / boost::numeric::ublas::norm_inf(U_exact) << std::endl;  
#endif 
}

// Perform GMRES on GPU
void GMRES_Device(DEVICE_MAT_t& A, DEVICE_VEC_t& F, DEVICE_VEC_t& U_exact, DEVICE_VEC_t& U_approx_gpu) {
    unsigned int restart = 300; 
    unsigned int max_iters = 10000; 
    double rel_tol = 1e-8; 
    cusp::verbose_monitor<double> monitor( F, max_iters, rel_tol ); 
    // 1e-8, 10000, 300); 

    // Solve Au = F
    cusp::krylov::gmres(A, U_approx_gpu, F, restart, monitor); 

    std::cout << "DONE\n"; 
#if 0 
std::cout << "GMRES Iterations: " << tag.iters() << std::endl;
    std::cout << "GMRES Error Estimate: " << tag.error() << std::endl;
    std::cout << "GMRES Krylov Dim: " << tag.krylov_dim() << std::endl;
    std::cout << "GMRES Max Number of Restarts (max_iter/krylov_dim): " << tag.max_restarts() << std::endl;
    std::cout << "GMRES Max Number of Iterations: " << tag.max_iterations() << std::endl;
    std::cout << "GMRES Tolerance: " << tag.tolerance() << std::endl;

    viennacl::vector_range<DEVICE_VEC_t> U_exact_view(U_exact, viennacl::range(U_exact.size() - F.size(),U_exact.size()));

    DEVICE_VEC_t diff(F.size()); 

    viennacl::linalg::sub(U_approx_gpu, U_exact_view, diff); 

    std::cout << "Rel l1   Norm: " << viennacl::linalg::norm_1(diff) / viennacl::linalg::norm_1(U_exact) << std::endl;  
    std::cout << "Rel l2   Norm: " << viennacl::linalg::norm_2(diff) / viennacl::linalg::norm_2(U_exact) << std::endl;  
    std::cout << "Rel linf Norm: " << viennacl::linalg::norm_inf(diff) / viennacl::linalg::norm_inf(U_exact) << std::endl;  
    #endif 
}

//---------------------------------

// Assemble the LHS matrix with the Identity for boundary nodes. Assume solver
// is intelligent enough to use information and converge
// 
void assemble_System( RBFFD& der, Grid& grid, HOST_MAT_t& A, HOST_VEC_t& F, HOST_VEC_t& U_exact){
    unsigned int N = grid.getNodeListSize(); 
    unsigned int n = grid.getMaxStencilSize(); 

    unsigned int nb_bnd = grid.getBoundaryIndicesSize();

    std::cout << "Boundary nodes: " << nb_bnd << std::endl;
        
    //------ RHS ----------

    SphericalHarmonic::Sph105 UU; 

    std::vector<NodeType>& nodes = grid.getNodeList(); 

    for (unsigned int i = 0; i < N; i++) {
        NodeType& node = nodes[i]; 
        double Xx = node.x(); 
        double Yy = node.y(); 
        double Zz = node.z(); 

        U_exact[i] = UU.eval(Xx, Yy, Zz) + 2*M_PI; 
        // Solving -lapl(u + const) = f = -lapl(u) + 0
        // of course the lapl(const) is 0, so we will have a test to verify
        // that our null space is closed. 
        F[i] = -UU.lapl(Xx, Yy, Zz); 
    }
    
    
    //------ LHS ----------
    unsigned ind = 0; 
    for (unsigned int i = 0; i < N; i++) {
        StencilType& sten = grid.getStencil(i); 
        double* lapl = der.getStencilWeights(RBFFD::LSFC, i); 

        A.row_offsets[i] = ind; 

        for (unsigned int j = 0; j < n; j++) {
            A.column_indices[ind] = sten[j]; 
            A.values[ind] = -lapl[j]; 
            ind++; 
        }
    }

    // VERY IMPORTANT. UNSPECIFIED LAUNCH FAILURES ARE CAUSED BY FORGETTING THIS!
    A.row_offsets[N] = ind; 
}

template <typename VecT>
void write_to_file(VecT vec, std::string filename)
{
    std::ofstream fout;
    fout.open(filename.c_str());
    for (size_t i = 0; i < vec.size(); i++) {
        fout << std::setprecision(10) << vec[i] << std::endl;
    }
    fout.close();
    std::cout << "Wrote " << filename << std::endl;
}


void write_System ( HOST_MAT_t& A, HOST_VEC_t& F, HOST_VEC_t& U_exact )
{
    write_to_file(F, "output/F.mtx"); 
    write_to_file(U_exact, "output/U_exact.mtx"); 
    cusp::io::write_matrix_market_file(A,"output/LHS.mtx"); 
    std::cout << "Wrote output/LHS.mtx\n"; 
}

void write_Solution( Grid& grid, HOST_VEC_t& U_exact, DEVICE_VEC_t& U_approx_gpu ) 
{
    unsigned int nb_bnd = grid.getBoundaryIndicesSize();

    // IF we want to write details we need to copy back to host. 
    HOST_VEC_t U_approx(U_exact.size());

    //thrust::copy(U_exact.begin(), U_exact.begin()+nb_bnd, U_approx.begin());
    thrust::copy(U_approx_gpu.begin(), U_approx_gpu.end(), U_approx.begin());

    write_to_file(U_approx, "output/U_gpu.mtx"); 
}


//---------------------------------

void gpuTest(RBFFD& der, Grid& grid, int primeGPU=0) {
    unsigned int N = grid.getNodeListSize(); 
    unsigned int n = grid.getMaxStencilSize(); 

    char test_name[256]; 
    char assemble_timer_name[256]; 
    char copy_timer_name[512]; 
    char test_timer_name[256]; 

    if (primeGPU) {
        sprintf(test_name, "%u PRIMING THE GPU", N);  
        sprintf(assemble_timer_name, "%u Primer Assemble", N);
        sprintf(copy_timer_name,     "%u Primer Copy To VCL_CSR", N); 
        sprintf(test_timer_name, "%u Primer GMRES test", N); 
    } else { 
        sprintf(test_name, "%u GMRES GPU (VCL_CSR)", N);  
        sprintf(assemble_timer_name, "%u UBLAS_CSR Assemble", N);
        sprintf(copy_timer_name,     "%u UBLAS_CSR Copy To VCL_CSR", N); 
        sprintf(test_timer_name, "%u GPU GMRES test", N); 
    }

    if (!tm.contains(assemble_timer_name)) { tm[assemble_timer_name] = new EB::Timer(assemble_timer_name); } 
    if (!tm.contains(copy_timer_name)) { tm[copy_timer_name] = new EB::Timer(copy_timer_name); } 
    if (!tm.contains(test_timer_name)) { tm[test_timer_name] = new EB::Timer(test_timer_name); } 


    std::cout << test_name << std::endl;


    // ----- ASSEMBLE -----
    tm[assemble_timer_name]->start(); 
    // Keep rows in system for boundary
    HOST_MAT_t* A = new HOST_MAT_t(N, N, n*N); 
    HOST_VEC_t* F = new HOST_VEC_t(N, 1);
    HOST_VEC_t* U_exact = new HOST_VEC_t(N, 1);
    assemble_System(der, grid, *A, *F, *U_exact); 
    tm[assemble_timer_name]->stop(); 
    
    if (!primeGPU) {
        write_System(*A, *F, *U_exact); 
    }

    // ----- SOLVE -----

    std::cout << "COPYING\n"; 
    tm[copy_timer_name]->start();

    DEVICE_MAT_t* A_gpu = new DEVICE_MAT_t(*A); 
    DEVICE_VEC_t* F_gpu = new DEVICE_VEC_t(*F); 
    DEVICE_VEC_t* U_exact_gpu = new DEVICE_VEC_t(*U_exact); 
    DEVICE_VEC_t* U_approx_gpu = new DEVICE_VEC_t(*F);

    tm[copy_timer_name]->stop();

    std::cout << "COMPUTING GMRES\n";
    tm[test_timer_name]->start();
    // Use GMRES to solve A*u = F
    GMRES_Device(*A_gpu, *F_gpu, *U_exact_gpu, *U_approx_gpu);
    tm[test_timer_name]->stop();

    write_Solution(grid, *U_exact, *U_approx_gpu); 

    // Cleanup
    delete(A);
    delete(F);
    delete(U_exact);
    delete(A_gpu);
    delete(F_gpu);
    delete(U_exact_gpu);
    delete(U_approx_gpu);
}


int main(void)
{
    bool writeIntermediate = true; 
    bool primed = false; 

    std::vector<std::string> grids; 

    //grids.push_back("~/GRIDS/md/md005.00036"); 

    grids.push_back("~/GRIDS/md/md165.27556"); 
#if 0 
    grids.push_back("~/GRIDS/md/md031.01024"); 
    grids.push_back("~/GRIDS/md/md050.02601"); 
    grids.push_back("~/GRIDS/md/md063.04096"); 
    grids.push_back("~/GRIDS/md/md089.08100"); 
    grids.push_back("~/GRIDS/md/md127.16384"); 
    grids.push_back("~/GRIDS/md/md165.27556"); 
#endif 
#if 0
    grids.push_back("~/GRIDS/geoff/scvtmesh_100k_nodes.ascii"); 
    grids.push_back("~/GRIDS/geoff/scvtmesh_500k_nodes.ascii"); 
    grids.push_back("~/GRIDS/geoff/scvtmesh_1m_nodes.ascii"); 
#endif 
    //grids.push_back("~/GRIDS/geoff/scvtmesh_1m_nodes.ascii"); 

    for (size_t i = 0; i < grids.size(); i++) {
        std::string& grid_name = grids[i]; 

        std::string weight_timer_name = grid_name + " Calc Weights";  

        tm[weight_timer_name] = new EB::Timer(weight_timer_name.c_str()); 

        // Get contours from rbfzone.blogspot.com to choose eps_c1 and eps_c2 based on stencil_size (n)
        unsigned int stencil_size = 40;
        double eps_c1 = 0.027;
        double eps_c2 = 0.274;


        GridReader* grid = new GridReader(grid_name, 4); 
        grid->setMaxStencilSize(stencil_size); 
        // We do not read until generate is called: 

        Grid::GridLoadErrType err = grid->loadFromFile(); 
        if (err == Grid::NO_GRID_FILES) 
        {
            grid->generate();
            // NOTE: We force at least one node in the domain to be a boundary. 
            //-----------------------------
            // We will set the first node as a boundary/ground point. We know
            // the normal because we're on teh sphere centered at (0,0,0)
            unsigned int nodeIndex = 0; 
            NodeType& node = grid->getNode(nodeIndex); 
            Vec3 nodeNormal = node - Vec3(0,0,0); 
            grid->appendBoundaryIndex(nodeIndex, nodeNormal); 
            //-----------------------------
            if (writeIntermediate) {
                grid->writeToFile(); 
            }
        } 
        std::cout << "Generate Stencils\n";
        Grid::GridLoadErrType st_err = grid->loadStencilsFromFile(); 
        if (st_err == Grid::NO_STENCIL_FILES) {
            //            grid->generateStencils(Grid::ST_BRUTE_FORCE);   
#if 1
            grid->generateStencils(Grid::ST_KDTREE);   
#else 
            grid->setNSHashDims(50, 50,50);  
            grid->generateStencils(Grid::ST_HASH);   
#endif 
            if (writeIntermediate) {
                grid->writeToFile(); 
            }
        }


        std::cout << "Generate RBFFD Weights\n"; 
        tm[weight_timer_name]->start(); 
        RBFFD der(RBFFD::LSFC | RBFFD::XSFC | RBFFD::YSFC | RBFFD::ZSFC, grid, 3, 0); 
        der.setEpsilonByParameters(eps_c1, eps_c2);
        int der_err = der.loadAllWeightsFromFile(); 
        if (der_err) {
            der.computeAllWeightsForAllStencils(); 

            tm[weight_timer_name]->start(); 
            if (writeIntermediate) {
                der.writeAllWeightsToFile(); 
            }
        }

        if (!primed)  {
            std::cout << "\n\n"; 
            cout << "Priming GPU with dummy operations (removes compile from benchmarks)\n";
            gpuTest(der,*grid, 1);
            gpuTest(der,*grid, 1);
            primed = true; 
            std::cout << "\n\n"; 
        } 

        // No support for GMRES on the CPU yet. 
        //cpuTest(der,*grid);  
        gpuTest(der,*grid);  

        delete(grid); 
    }

    tm.printAll();
    tm.writeToFile();
    return EXIT_SUCCESS;
}

