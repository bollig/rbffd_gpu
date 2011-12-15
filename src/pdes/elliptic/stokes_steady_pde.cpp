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
    L_graph = new GraphType( ncols );
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

    this->build_graph(*L_host, *L_graph); 
    boost::numeric::ublas::vector<size_t> r( ncols );
    this->get_cuthill_mckee_order(*L_graph, r); 

    this->write_to_file(r, "CuthillMckeeOrder.mtx");


    div_op = MatType(boost::numeric::ublas::project(*L_host, boost::numeric::ublas::range(3*N,4*N+4), boost::numeric::ublas::range(0,4*N+4)));

    viennacl::io::write_matrix_market_file(div_op, "DIV_operator.mtx"); 

}


// Fill an adjacency graph based on the matrix
void StokesSteadyPDE::build_graph(MatType& mat, GraphType& G) {

    for (MatType::const_iterator1 row_it = mat.begin1();
            row_it != mat.end1();
            ++row_it) {
        for (MatType::const_iterator2 col_it = row_it.begin();
                col_it != row_it.end();
                ++col_it) {
            boost::add_edge( col_it.index1(), col_it.index2(), G);
        }
    }
}


void StokesSteadyPDE::get_cuthill_mckee_order(GraphType& G, boost::numeric::ublas::vector<size_t>& lookup_chart) {

    boost::property_map<GraphType,boost::vertex_index_t>::type
        index_map = get(boost::vertex_index, G);

    std::cout << "-> CutHill McKee Starting Bandwidth: " << boost::bandwidth(G) << "\n";

    // Solving for CutHill McKee
    std::vector<VertexType> inv_perm(boost::num_vertices(G));
    std::vector<size_type> perm(boost::num_vertices(G));
    cuthill_mckee_ordering(G, inv_perm.rbegin(), get(boost::vertex_color,G), make_degree_map(G));

    // Building new lookup chart
    for ( size_t i = 0; i < inv_perm.size(); i++ )
        lookup_chart[i] = index_map[inv_perm[i]];

    // Finding new bandwidth for debug purposes
    for (size_type c = 0; c != inv_perm.size(); ++c )
        perm[index_map[inv_perm[c]]] = c;
    std::cout << "-> CutHill McKee Ending Bandwidth: " << boost::bandwidth(G, make_iterator_property_map(&perm[0], index_map, perm[0])) << "\n";

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

