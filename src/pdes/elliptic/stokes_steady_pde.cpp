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
    L_reordered = new MatType(nrows, ncols, num_nonzeros); 
    //    div_op = new MatType(nb_stencils+4, ncols, 3*max_stencil_size*N + 4*N + 3*N); 
    F_host = new VecType(ncols);
    F_reordered = new VecType(ncols);
    u_host = new VecType(ncols);
    u_reordered = new VecType(ncols);

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

#if 1
    L_graph = new GraphType( ncols-4 );

    // Reorder within the standard blocks (exclude constraints)
    MatType submat = MatType(boost::numeric::ublas::project(*L_host, boost::numeric::ublas::range(0,nrows-4), boost::numeric::ublas::range(0,ncols-4)));
    this->build_graph(submat, *L_graph);

    m_lookup = new boost::numeric::ublas::vector<size_t>( ncols );
    std::cout << "Reordering interior of matrix (excluding the 4 extra constraints)\n";
    this->get_cuthill_mckee_order(*L_graph, *m_lookup ); 

    for (int i = 0; i < 4; i++) {
        (*m_lookup)((ncols-4) + i) = (ncols-4)+i; 
    }
    this->get_reordered_system(*L_host, *F_host, *m_lookup, *L_reordered, *F_reordered); 

    viennacl::io::write_matrix_market_file(*L_reordered, "L_reordered.mtx");
    std::cout << "Wrote L_reordered.mtx\n"; 
    this->write_to_file(*F_reordered, "F_reordered.mtx");

    this->write_to_file(*m_lookup, "CuthillMckeeOrder.mtx");
#endif    


    // TODO: figure out ordering here
    div_op = MatType(boost::numeric::ublas::project(*L_host, boost::numeric::ublas::range(3*N,4*N+4), boost::numeric::ublas::range(0,4*N+4)));
    viennacl::io::write_matrix_market_file(div_op, "DIV_operator.mtx"); 

}


void StokesSteadyPDE::solve() 
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
    *u_reordered = viennacl::linalg::solve(*L_reordered, *F_reordered, viennacl::linalg::gmres_tag(1e-8, 100));
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




void StokesSteadyPDE::get_reordered_system(MatType& in_mat, VecType& in_vec, boost::numeric::ublas::vector<size_t>& order, MatType& out_mat, VecType& out_vec) {
    inv_m_lookup = new boost::numeric::ublas::vector<size_t>( order.size() );

    boost::numeric::ublas::vector<size_t>& inv_lookup = *inv_m_lookup;     


    // We construct an inverse lookup table so we can iterate over ONLY the nonzero elements
    for (size_t i = 0; i < order.size(); i++) {
        inv_lookup(order(i)) = i; 
    }

    for (MatType::const_iterator1 row_it = in_mat.begin1();
            row_it != in_mat.end1();
            ++row_it) {
        for (MatType::const_iterator2 col_it = row_it.begin();
                col_it != row_it.end();
                ++col_it) {
            // We use the reordering as LHS(R,R). So that means every non-zero (i,j)
            // is permuted to the new index given by (R(i),R(j))
            size_t row_ind = inv_lookup(col_it.index1()); 
            size_t col_ind = inv_lookup(col_it.index2());

            out_mat(row_ind, col_ind) = *col_it; 
        }
    }
    for (VecType::const_iterator row_it = in_vec.begin();
            row_it != in_vec.end(); 
            ++row_it) {
        size_t row_ind = inv_lookup(row_it.index());
        out_vec(row_ind) = *row_it; 
    }
}

void StokesSteadyPDE::get_original_order(VecType& in_vec, VecType& out_vec) {
    for (VecType::const_iterator row_it = in_vec.begin();
            row_it != in_vec.end(); 
            ++row_it) {
        size_t row_ind = (*m_lookup)(row_it.index());
        out_vec(row_ind) = *row_it; 
    }
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
    std::cout << "Wrote " << filename << std::endl;
}

