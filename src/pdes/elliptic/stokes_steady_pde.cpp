#include "stokes_steady_pde.h"

#include <boost/numeric/ublas/io.hpp>
#include "viennacl/io/matrix_market.hpp"



void StokesSteadyPDE::assemble() {

    std::cout << "Assembling...." << std::endl;

    double eta = 1.;

    unsigned int nb_stencils = grid_ref.getStencilsSize();
    unsigned int nb_nodes = grid_ref.getNodeListSize(); 
    unsigned int max_stencil_size = grid_ref.getMaxStencilSize();

    unsigned int nrows = 4 * nb_stencils; 
    unsigned int ncols = 4 * nb_nodes; 

    unsigned int N = nb_nodes;

    unsigned int num_nonzeros = 9*max_stencil_size*N+2*(4*N)+2*(3*N);  

    L_host = new MatType(nrows, ncols, num_nonzeros); 
    F_host = new VecType(ncols);
#if 1
    // Fill L


    // U (block)  row
    for (unsigned int i = 0; i < nb_stencils; i++) {
        StencilType& st = grid_ref.getStencil(i);

        // TODO: change these to *SFC weights (when computed)
        double* ddx = der_ref.getStencilWeights(RBFFD::XSFC, i);
        double* lapl = der_ref.getStencilWeights(RBFFD::LSFC, i); 

        for (unsigned int j = 0; j < st.size(); j++) {
            unsigned int diag_row_ind = i + 0*N;
            unsigned int diag_col_ind = st[j] + 0*N;

            (*L_host)(diag_row_ind, diag_col_ind) = -eta * lapl[j];  
        }
        for (unsigned int j = 0; j < st.size(); j++) {
            unsigned int diag_row_ind = i + 0*N;
            unsigned int diag_col_ind = st[j] + 3*N;

            (*L_host)(diag_row_ind, diag_col_ind) = ddx[j];  
        }
    }

    // V (block)  row
    for (unsigned int i = 0; i < nb_stencils; i++) {
        StencilType& st = grid_ref.getStencil(i);

        // TODO: change these to *SFC weights (when computed)
        double* ddy = der_ref.getStencilWeights(RBFFD::YSFC, i);
        double* lapl = der_ref.getStencilWeights(RBFFD::LSFC, i); 

        for (unsigned int j = 0; j < st.size(); j++) {
            unsigned int diag_row_ind = i + 1*N;
            unsigned int diag_col_ind = st[j] + 1*N;

            (*L_host)(diag_row_ind, diag_col_ind) = -eta * lapl[j];  
        }
        for (unsigned int j = 0; j < st.size(); j++) {
            unsigned int diag_row_ind = i + 1*N;
            unsigned int diag_col_ind = st[j] + 3*N;

            (*L_host)(diag_row_ind, diag_col_ind) = ddy[j];  
        }
    }

    // W (block)  row
    for (unsigned int i = 0; i < nb_stencils; i++) {
        StencilType& st = grid_ref.getStencil(i);

        // TODO: change these to *SFC weights (when computed)
        double* ddz = der_ref.getStencilWeights(RBFFD::ZSFC, i);
        double* lapl = der_ref.getStencilWeights(RBFFD::LSFC, i); 

        for (unsigned int j = 0; j < st.size(); j++) {
            unsigned int diag_row_ind = i + 2*N;
            unsigned int diag_col_ind = st[j] + 2*N;

            (*L_host)(diag_row_ind, diag_col_ind) = -eta * lapl[j];  
        }
        for (unsigned int j = 0; j < st.size(); j++) {
            unsigned int diag_row_ind = i + 2*N;
            unsigned int diag_col_ind = st[j] + 3*N;

            (*L_host)(diag_row_ind, diag_col_ind) = ddz[j];  
        }
    }


    // P (block)  row
    for (unsigned int i = 0; i < nb_stencils; i++) {
        StencilType& st = grid_ref.getStencil(i);

        // TODO: change these to *SFC weights (when computed)
        double* ddx = der_ref.getStencilWeights(RBFFD::XSFC, i);
        double* ddy = der_ref.getStencilWeights(RBFFD::YSFC, i);
        double* ddz = der_ref.getStencilWeights(RBFFD::ZSFC, i);

        for (unsigned int j = 0; j < st.size(); j++) {
            unsigned int diag_row_ind = i + 3*N;
            unsigned int diag_col_ind = st[j] + 0*N;

            (*L_host)(diag_row_ind, diag_col_ind) = ddx[j];  
        }
        for (unsigned int j = 0; j < st.size(); j++) {
            unsigned int diag_row_ind = i + 3*N;
            unsigned int diag_col_ind = st[j] + 1*N;

            (*L_host)(diag_row_ind, diag_col_ind) = ddy[j];  
        }
        for (unsigned int j = 0; j < st.size(); j++) {
            unsigned int diag_row_ind = i + 3*N;
            unsigned int diag_col_ind = st[j] + 2*N;

            (*L_host)(diag_row_ind, diag_col_ind) = ddz[j];  
        }
    }

    // Fill F
    // Write both to disk (spy them in Matlab)

    viennacl::io::write_matrix_market_file(*L_host, "L_host.mtx");
    std::cout << "Wrote L_host.mtx\n"; 
    
#endif
}

