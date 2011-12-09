#include "stokes_steady_pde.h"

#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include "viennacl/io/matrix_market.hpp"



void StokesSteadyPDE::assemble() {

    std::cout << "Assembling...." << std::endl;

    double eta = 1.;
    double Ra = 1.e6;

    unsigned int nb_stencils = grid_ref.getStencilsSize();
    unsigned int nb_nodes = grid_ref.getNodeListSize(); 
    unsigned int max_stencil_size = grid_ref.getMaxStencilSize();

    // Add 4 for extra constraint cols and rows to close nullspace
    unsigned int nrows = 4 * nb_stencils + 4; 
    unsigned int ncols = 4 * nb_nodes + 4; 

    unsigned int N = nb_nodes;

    unsigned int num_nonzeros = 9*max_stencil_size*N+2*(4*N)+2*(3*N);  

    L_host = new MatType(nrows, ncols, num_nonzeros); 
//    div_op = new MatType(nb_stencils+4, ncols, 3*max_stencil_size*N + 4*N + 3*N); 
    F_host = new VecType(ncols);
    u_host = new VecType(ncols);

    // -----------------  Fill LHS --------------------
    //
    // U (block)  row
    for (unsigned int i = 0; i < nb_stencils; i++) {
        StencilType& st = grid_ref.getStencil(i);

        // TODO: change these to *SFC weights (when computed)
        double* ddx = der_ref.getStencilWeights(RBFFD::XSFC, i);
        double* lapl = der_ref.getStencilWeights(RBFFD::LSFC, i); 

        unsigned int diag_row_ind = i + 0*N;

        for (unsigned int j = 0; j < st.size(); j++) {
            unsigned int diag_col_ind = st[j] + 0*N;

            (*L_host)(diag_row_ind, diag_col_ind) = -eta * lapl[j];  
        }
        for (unsigned int j = 0; j < st.size(); j++) {
            unsigned int diag_col_ind = st[j] + 3*N;

            (*L_host)(diag_row_ind, diag_col_ind) = ddx[j];  
        }

        // Added constraint to square mat and close nullspace
        (*L_host)(diag_row_ind, 4*N+0) = 1.; 
    }

    // V (block)  row
    for (unsigned int i = 0; i < nb_stencils; i++) {
        StencilType& st = grid_ref.getStencil(i);

        // TODO: change these to *SFC weights (when computed)
        double* ddy = der_ref.getStencilWeights(RBFFD::YSFC, i);
        double* lapl = der_ref.getStencilWeights(RBFFD::LSFC, i); 

        unsigned int diag_row_ind = i + 1*N;

        for (unsigned int j = 0; j < st.size(); j++) {
            unsigned int diag_col_ind = st[j] + 1*N;

            (*L_host)(diag_row_ind, diag_col_ind) = -eta * lapl[j];  
        }
        for (unsigned int j = 0; j < st.size(); j++) {
            unsigned int diag_col_ind = st[j] + 3*N;

            (*L_host)(diag_row_ind, diag_col_ind) = ddy[j];  
        }

        // Added constraint to square mat and close nullspace
        (*L_host)(diag_row_ind, 4*N+1) = 1.; 
    }

    // W (block)  row
    for (unsigned int i = 0; i < nb_stencils; i++) {
        StencilType& st = grid_ref.getStencil(i);

        // TODO: change these to *SFC weights (when computed)
        double* ddz = der_ref.getStencilWeights(RBFFD::ZSFC, i);
        double* lapl = der_ref.getStencilWeights(RBFFD::LSFC, i); 

        unsigned int diag_row_ind = i + 2*N;

        for (unsigned int j = 0; j < st.size(); j++) {
            unsigned int diag_col_ind = st[j] + 2*N;

            (*L_host)(diag_row_ind, diag_col_ind) = -eta * lapl[j];  
        }
        for (unsigned int j = 0; j < st.size(); j++) {
            unsigned int diag_col_ind = st[j] + 3*N;

            (*L_host)(diag_row_ind, diag_col_ind) = ddz[j];  
        }
        
        // Added constraint to square mat and close nullspace
        (*L_host)(diag_row_ind, 4*N+2) = 1.; 
    }


    // P (block)  row
    for (unsigned int i = 0; i < nb_stencils; i++) {
        StencilType& st = grid_ref.getStencil(i);

        // TODO: change these to *SFC weights (when computed)
        double* ddx = der_ref.getStencilWeights(RBFFD::XSFC, i);
        double* ddy = der_ref.getStencilWeights(RBFFD::YSFC, i);
        double* ddz = der_ref.getStencilWeights(RBFFD::ZSFC, i);

        unsigned int diag_row_ind = i + 3*N;

        for (unsigned int j = 0; j < st.size(); j++) {
            unsigned int diag_col_ind = st[j] + 0*N;

            (*L_host)(diag_row_ind, diag_col_ind) = ddx[j];  
        }
        for (unsigned int j = 0; j < st.size(); j++) {
            unsigned int diag_col_ind = st[j] + 1*N;

            (*L_host)(diag_row_ind, diag_col_ind) = ddy[j];  
        }
        for (unsigned int j = 0; j < st.size(); j++) {
            unsigned int diag_col_ind = st[j] + 2*N;

            (*L_host)(diag_row_ind, diag_col_ind) = ddz[j];  
        }

        // Added constraint to square mat and close nullspace
        (*L_host)(diag_row_ind, 4*N+3) = 1.;  
    }
    
    // ------ EXTRA CONSTRAINT ROWS -----
    unsigned int diag_row_ind = 4*N;
    // U
    for (unsigned int j = 0; j < N; j++) {
        unsigned int diag_col_ind = j + 0*N;

        (*L_host)(diag_row_ind, diag_col_ind) = 1.;  
    }

    diag_row_ind++; 
    // V
    for (unsigned int j = 0; j < N; j++) {
        unsigned int diag_col_ind = j + 1*N;

        (*L_host)(diag_row_ind, diag_col_ind) = 1.;  
    }

    diag_row_ind++; 
    // W
    for (unsigned int j = 0; j < N; j++) {
        unsigned int diag_col_ind = j + 2*N;

        (*L_host)(diag_row_ind, diag_col_ind) = 1.;  
    }

    diag_row_ind++; 
    // P
    for (unsigned int j = 0; j < N; j++) {
        unsigned int diag_col_ind = j + 3*N;

        (*L_host)(diag_row_ind, diag_col_ind) = 1.;  
    }





    //------------- Fill F -------------
    // U
    for (unsigned int j = 0; j < N; j++) {
        unsigned int row_ind = j + 0*N;
        NodeType& node = grid_ref.getNode(j); 
        double rr = sqrt(node.x()*node.x() + node.y()*node.y() + node.z()*node.z());
        double dir = node.x();

        (*F_host)(row_ind) = (Ra * Temperature(j) * dir) / rr;  
    }

    // V
    for (unsigned int j = 0; j < N; j++) {
        unsigned int row_ind = j + 1*N;
        NodeType& node = grid_ref.getNode(j); 
        double rr = sqrt(node.x()*node.x() + node.y()*node.y() + node.z()*node.z());
        double dir = node.y();

        (*F_host)(row_ind) = (Ra * Temperature(j) * dir) / rr;  
    }
    
    // W
    for (unsigned int j = 0; j < N; j++) {
        unsigned int row_ind = j + 2*N;
        NodeType& node = grid_ref.getNode(j); 
        double rr = sqrt(node.x()*node.x() + node.y()*node.y() + node.z()*node.z());
        double dir = node.z();

        (*F_host)(row_ind) = (Ra * Temperature(j) * dir) / rr;  
    }

    // P
    for (unsigned int j = 0; j < N; j++) {
        unsigned int row_ind = j + 3*N;

        (*F_host)(row_ind) = 0;  
    }

    // Sum of U
    (*F_host)(4*N+0) = 0.;

    // Sum of V
    (*F_host)(4*N+1) = 0.;

    // Sum of W
    (*F_host)(4*N+2) = 0.;

    // Sum of P
    (*F_host)(4*N+3) = 0.;

    // Write both to disk (spy them in Matlab)

    viennacl::io::write_matrix_market_file(*L_host, "L_host.mtx");
    std::cout << "Wrote L_host.mtx\n"; 

    this->write_to_file(*F_host, "F.mtx");
    std::cout << "Wrote F_host.mtx\n"; 


    div_op = MatType(boost::numeric::ublas::project(*L_host, boost::numeric::ublas::range(3*N,4*N+4), boost::numeric::ublas::range(0,4*N+4)));

    viennacl::io::write_matrix_market_file(div_op, "DIV_operator.mtx"); 

}


//----------------------------------------------------------------------

    template<typename T>
void StokesSteadyPDE::write_to_file(boost::numeric::ublas::vector<T> vec, std::string filename)
{
    std::ofstream fout;
    fout.open(filename.c_str());
    for (int i = 0; i < vec.size(); i++) {
        fout << std::scientific << vec[i] << std::endl;
    }
    fout.close();
}

