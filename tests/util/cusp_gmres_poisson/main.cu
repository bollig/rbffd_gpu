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

// Perform GMRES on GPU
void GMRES_Device(DEVICE_MAT_t& A, DEVICE_VEC_t& F, DEVICE_VEC_t& U_exact, DEVICE_VEC_t& U_approx_gpu) {
#if 1
    size_t restart = 300; 
    int max_iters = 10000; 
    double rel_tol = 1e-8; 
#else 
    size_t restart = 50; 
    int max_iters = 100; 
    double rel_tol = 1e-8; 
#endif 

    try {

        //    cusp::convergence_monitor<double> monitor( F, max_iters, 0, 1e-3); 
        cusp::default_monitor<double> monitor( F, max_iters, rel_tol);// , 1e-3); 

        cudaThreadSynchronize();
        std::cout << "Generated monitor\n";
        // 1e-8, 10000, 300); 

        // Solve Au = F
        cusp::krylov::gmres(A, U_approx_gpu, F, restart, monitor); 
        cudaThreadSynchronize(); 

        //    monitor.print();

        if (monitor.converged())
        {
            std::cout << "\n[+++] Solver converged to " << monitor.relative_tolerance() << " relative tolerance";       
            std::cout << " after " << monitor.iteration_count() << " iterations" << std::endl << std::endl;
        }
        else
        {
            std::cout << "\n[XXX] Solver reached iteration limit " << monitor.iteration_limit() << " before converging";
            std::cout << " to " << monitor.relative_tolerance() << " relative tolerance " << std::endl << std::endl;
        }

        std::cout << "GMRES Iterations: " << monitor.iteration_count() << std::endl;
        std::cout << "GMRES Iteration Limit: " << monitor.iteration_limit() << std::endl;
        std::cout << "GMRES Residual Norm: " << monitor.residual_norm() << std::endl;
        std::cout << "GMRES Relative Tol: " << monitor.relative_tolerance() << std::endl;
        std::cout << "GMRES Absolute Tol: " << monitor.absolute_tolerance() << std::endl;
        std::cout << "GMRES Target Residual (Abs + Rel*norm(F)): " << monitor.tolerance() << std::endl;
    }
    catch(std::bad_alloc &e)
    {
        std::cerr << "Ran out of memory trying to compute GMRES: " << e.what() << std::endl;
        exit(-1);
    }
    catch(thrust::system_error &e)
    {
        std::cerr << "Some other error happened during GMRES: " << e.what() << std::endl;
        exit(-1);
    }


    try {

        typedef cusp::array1d<double, DEVICE_VEC_t>::view DEVICE_VEC_VIEW_t; 

        DEVICE_VEC_VIEW_t U_approx_view(U_exact.begin()+(U_exact.size() - F.size()), U_exact.end()); 

        DEVICE_VEC_t diff(U_approx_gpu); 

        //cusp::blas::axpy(U_exact.begin()+(U_exact.size() - F.size()), U_exact.end(), diff.begin(),  -1); 
        cusp::blas::axpy(U_approx_view, diff, -1); 

        std::cout << "Rel l1   Norm: " << cusp::blas::nrm1(diff) / cusp::blas::nrm1(U_exact) << std::endl;  
        std::cout << "Rel l2   Norm: " << cusp::blas::nrm2(diff) / cusp::blas::nrm2(U_exact) << std::endl;  
        std::cout << "Rel linf Norm: " << cusp::blas::nrmmax(diff) / cusp::blas::nrmmax(U_exact) << std::endl;  
    }
    catch(std::bad_alloc &e)
    {
        std::cerr << "Ran out of memory trying to compute Error Norms: " << e.what() << std::endl;
        exit(-1);
    }
    catch(thrust::system_error &e)
    {
        std::cerr << "Some other error happened during Error Norms: " << e.what() << std::endl;
        exit(-1);
    }
}

//---------------------------------

// Assemble the LHS matrix with the Identity for boundary nodes. Assume solver
// is intelligent enough to use information and converge
// 
void assemble_System_Compressed( RBFFD& der, Grid& grid, HOST_MAT_t& A, HOST_VEC_t& F, HOST_VEC_t& U_exact){
    unsigned int N = grid.getNodeListSize(); 
    unsigned int n = grid.getMaxStencilSize(); 

    unsigned int nb_bnd = grid.getBoundaryIndicesSize();

    std::cout << "Boundary nodes: " << nb_bnd << std::endl;


    //------ RHS ----------

    SphericalHarmonic::Sph105 UU; 

    std::vector<NodeType>& nodes = grid.getNodeList(); 

    // We want U_exact to have the FULL solution. 
    // F should only have the compressed problem. 
    for (unsigned int i = 0; i < nb_bnd; i++) {
        NodeType& node = nodes[i]; 
        double Xx = node.x(); 
        double Yy = node.y(); 
        double Zz = node.z(); 

        U_exact[i] = UU.eval(Xx, Yy, Zz) + 2*M_PI; 
    }

    for (unsigned int i = nb_bnd; i < N; i++) {
        NodeType& node = nodes[i]; 
        double Xx = node.x(); 
        double Yy = node.y(); 
        double Zz = node.z(); 

        U_exact[i] = UU.eval(Xx, Yy, Zz) + 2*M_PI; 
        // Solving -lapl(u + const) = f = -lapl(u) + 0
        // of course the lapl(const) is 0, so we will have a test to verify
        // that our null space is closed. 
        F[i-nb_bnd] = -UU.lapl(Xx, Yy, Zz); 
    }

    //------ LHS ----------

    unsigned int ind = 0; 
    // NOTE: assumes the boundary is sorted to the top of the node indices
    for (unsigned int i = nb_bnd; i < N; i++) {
        StencilType& sten = grid.getStencil(i); 
        double* lapl = der.getStencilWeights(RBFFD::LSFC, i); 

        A.row_offsets[i-nb_bnd] = ind;

        for (unsigned int j = 0; j < n; j++) {
            if (sten[j] < (int)nb_bnd) { 
                // Subtract the solution*weight from the element of the RHS. 
                F[i-nb_bnd] -= (U_exact[sten[j]] * ( -lapl[j] )); 
                // std::cout << "Node " << i << " depends on boundary\n"; 
            } else {
                // Offset by nb_bnd so we crop off anything related to the boundary
                A.column_indices[ind] = sten[j]-nb_bnd; 
                A.values[ind] = -lapl[j]; 
                ind++; 
            }
        }
    }    

    // VERY IMPORTANT. UNSPECIFIED LAUNCH FAILURES ARE CAUSED BY FORGETTING THIS!
    A.row_offsets[N-nb_bnd] = ind; 
}



// Assemble the LHS matrix with the Identity for boundary nodes. Assume solver
// is intelligent enough to use information and converge
// 
void assemble_System_Bnd_Eye( RBFFD& der, Grid& grid, HOST_MAT_t& A, HOST_VEC_t& F, HOST_VEC_t& U_exact){
    unsigned int N = grid.getNodeListSize(); 
    unsigned int n = grid.getMaxStencilSize(); 

    unsigned int nb_bnd = grid.getBoundaryIndicesSize();

    std::cout << "Boundary nodes: " << nb_bnd << std::endl;

    //------ RHS ----------

    SphericalHarmonic::Sph105 UU; 

    std::vector<NodeType>& nodes = grid.getNodeList(); 

    for (unsigned int i = 0; i < nb_bnd; i++) {
        NodeType& node = nodes[i]; 
        double Xx = node.x(); 
        double Yy = node.y(); 
        double Zz = node.z(); 

        U_exact[i] = UU.eval(Xx, Yy, Zz) + 2*M_PI; 
        F[i] = U_exact[i]; 
    }

    for (unsigned int i = nb_bnd; i < N; i++) {
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
    for (unsigned int i = 0; i < nb_bnd; i++) {
        A.row_offsets[i] = ind; 
        A.column_indices[ind] = i; 
        A.values[ind] = 1; 
        ind++; 
    }

    for (unsigned int i = nb_bnd; i < N; i++) {
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
}

void write_Solution( Grid& grid, HOST_VEC_t& U_exact, DEVICE_VEC_t& U_approx_gpu ) 
{
    unsigned int nb_bnd = grid.getBoundaryIndicesSize();

    // IF we want to write details we need to copy back to host. 
    HOST_VEC_t U_approx(U_exact.size());

    if (U_approx_gpu.size() == U_exact.size()) {
        thrust::copy(U_approx_gpu.begin(), U_approx_gpu.end(), U_approx.begin());
    } else {
        thrust::copy(U_exact.begin(), U_exact.begin()+nb_bnd, U_approx.begin());
        thrust::copy(U_approx_gpu.begin(), U_approx_gpu.end(), U_approx.begin()+nb_bnd);
    }

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
        sprintf(copy_timer_name,     "%u Primer Copy To CUSP_DEVICE_CSR", N); 
        sprintf(test_timer_name, "%u Primer GMRES test", N); 
    } else { 
        sprintf(test_name, "%u GMRES GPU (CUSP_DEVICE_CSR)", N);  
        sprintf(assemble_timer_name, "%u CUSP_HOST_CSR Assemble", N);
        sprintf(copy_timer_name,     "%u CUSP_HOST_CSR Copy To CUSP_DEVICE_CSR", N); 
        sprintf(test_timer_name, "%u GPU GMRES test", N); 
    }

    if (!tm.contains(assemble_timer_name)) { tm[assemble_timer_name] = new EB::Timer(assemble_timer_name); } 
    if (!tm.contains(copy_timer_name)) { tm[copy_timer_name] = new EB::Timer(copy_timer_name); } 
    if (!tm.contains(test_timer_name)) { tm[test_timer_name] = new EB::Timer(test_timer_name); } 


    std::cout << test_name << std::endl;


    // ----- ASSEMBLE -----
    tm[assemble_timer_name]->start(); 
#if 0
    // Keep rows in system for boundary
    HOST_MAT_t* A = new HOST_MAT_t(N, N, n*N); 
    HOST_VEC_t* F = new HOST_VEC_t(N, 1);
    HOST_VEC_t* U_exact = new HOST_VEC_t(N, 1);
    assemble_System_Bnd_Eye(der, grid, *A, *F, *U_exact); 
#else 
    // Compress system to remove boundary rows
    unsigned int nb_bnd = grid.getBoundaryIndicesSize();
    HOST_MAT_t* A = new HOST_MAT_t(N-nb_bnd, N-nb_bnd, n*(N-nb_bnd)); 
    HOST_VEC_t* F = new HOST_VEC_t(N-nb_bnd, 1);
    HOST_VEC_t* U_exact = new HOST_VEC_t(N, 1);
    assemble_System_Compressed(der, grid, *A, *F, *U_exact); 
#endif 
    tm[assemble_timer_name]->stop(); 

    if (!primeGPU) {
        //write_System(*A, *F, *U_exact); 
    }
    // ----- SOLVE -----

    tm[copy_timer_name]->start();

    DEVICE_MAT_t* A_gpu = new DEVICE_MAT_t(*A); 
    DEVICE_VEC_t* F_gpu = new DEVICE_VEC_t(*F); 
    DEVICE_VEC_t* U_exact_gpu = new DEVICE_VEC_t(*U_exact); 
    DEVICE_VEC_t* U_approx_gpu = new DEVICE_VEC_t(F->size(), 0);

    tm[copy_timer_name]->stop();

    tm[test_timer_name]->start();
    // Use GMRES to solve A*u = F
    GMRES_Device(*A_gpu, *F_gpu, *U_exact_gpu, *U_approx_gpu);
    tm[test_timer_name]->stop();

    if (!primeGPU) {
        write_Solution(grid, *U_exact, *U_approx_gpu); 
    }
    // Cleanup
    delete(A);
    delete(A_gpu);
    delete(F);
    delete(U_exact);
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

//    grids.push_back("~/GRIDS/md/md165.27556"); 
#if 1 
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
            tm[weight_timer_name]->stop(); 

            #if 0
            // Im finding that its more efficient to compute the weights than write and load from disk. 
            if (writeIntermediate) {
                der.writeAllWeightsToFile(); 
            }
            #endif 
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

