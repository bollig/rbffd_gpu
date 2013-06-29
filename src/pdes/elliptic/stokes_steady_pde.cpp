
// contains include mpi.h via communicator, which must be calseld before stdio.h or stdlib.h for
// mpich2

#include <mpi.h>
#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include "viennacl/io/matrix_market.hpp"

#include "stokes_steady_pde.h"


void StokesSteadyPDE::setupTimers() {
    tm["RHS_discrete"] = new EB::Timer("[StokesSteadyPDE] Fill RHS Discrete"); 
    tm["RHS_continuous"] = new EB::Timer("[StokesSteadyPDE] Fill RHS Continuous"); 
    tm["LHS"] = new EB::Timer("[StokesSteadyPDE] Fill LHS"); 
    tm["gmres"] = new EB::Timer("[StokesSteadyPDE] Compute GMRES (1e-8, 200)"); 

}


void StokesSteadyPDE::reorder() {

    LHS_reordered = new MatType(nrows, ncols, NNZ); 
    RHS_reordered = new VecType(ncols);
    U_reordered = new VecType(ncols);

    LHS_graph = new GraphType( ncols-4 );

    // Reorder within the standard blocks (exclude constraints)
    MatType submat = MatType(boost::numeric::ublas::project(*LHS, boost::numeric::ublas::range(0,nrows-4), boost::numeric::ublas::range(0,ncols-4)));
    this->build_graph(submat, *LHS_graph);

    m_lookup = new boost::numeric::ublas::vector<size_t>( ncols );

    std::cout << "Reordering interior of matrix (excluding the 4 extra constraints)\n";

    this->get_cuthill_mckee_order(*LHS_graph, *m_lookup ); 

    for (int i = 0; i < 4; i++) {
        (*m_lookup)((ncols-4) + i) = (ncols-4)+i; 
    }
    this->get_reordered_system(*LHS, *RHS_continuous, *m_lookup, *LHS_reordered, *RHS_reordered); 

    viennacl::io::write_matrix_market_file(*LHS_reordered, "LHS_reordered.mtx");
    std::cout << "Wrote LHS_reordered.mtx\n"; 
    this->write_to_file(*RHS_reordered, "RHS_reordered.mtx");

    this->write_to_file(*m_lookup, "CuthillMckeeOrder.mtx");
}

void StokesSteadyPDE::assemble() {
    std::cout << "Assembling...." << std::endl;
    if (constantViscosity) {
        this->fillLHS_ConstViscosity(); 
        this->fillRHS_ConstViscosity(); 
    } else {
//        this->fillLHS_VarViscosity(); 
//        this->fillRHS_VarViscosity(); 
    }
    std::cout << "Done." << std::endl;
}

/**************  LHS *****************/ 

void StokesSteadyPDE::fillLHS_ConstViscosity() { 

    tm["LHS"]->start(); 
    double eta = 1.;
    //double Ra = 1.e6;

    // We have different nb_stencils and nb_nodes when we parallelize. The subblocks might not be full
    unsigned int nb_stencils = grid_ref.getStencilsSize();
    unsigned int nb_nodes = grid_ref.getNodeListSize(); 
    unsigned int max_stencil_size = grid_ref.getMaxStencilSize();

    // Add 4 for extra constraint cols and rows to close nullspace
    // --------- THESE 4 VARS ARE CLASS PROPS ------------
    nrows = 4 * nb_stencils + 4; 
    ncols = 4 * nb_nodes + 4; 
    N = nb_nodes;
    NNZ = 9*max_stencil_size*N+2*(4*N)+2*(3*N);  
    // ---------------------------------------------------

    LHS = new MatType(nrows, ncols, NNZ); 

    std::cout << "Filling LHS\n";

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

            (*LHS)(diag_row_ind, diag_col_ind) = -eta * lapl[j];  
        }
        for (unsigned int j = 0; j < st.size(); j++) {
            unsigned int diag_col_ind = st[j] + 3*N;

            (*LHS)(diag_row_ind, diag_col_ind) = ddx[j];  
        }

        // Added constraint to square mat and close nullspace
        (*LHS)(diag_row_ind, 4*N+0) = 1.; 
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

            (*LHS)(diag_row_ind, diag_col_ind) = -eta * lapl[j];  
        }
        for (unsigned int j = 0; j < st.size(); j++) {
            unsigned int diag_col_ind = st[j] + 3*N;

            (*LHS)(diag_row_ind, diag_col_ind) = ddy[j];  
        }

        // Added constraint to square mat and close nullspace
        (*LHS)(diag_row_ind, 4*N+1) = 1.; 
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

            (*LHS)(diag_row_ind, diag_col_ind) = -eta * lapl[j];  
        }
        for (unsigned int j = 0; j < st.size(); j++) {
            unsigned int diag_col_ind = st[j] + 3*N;

            (*LHS)(diag_row_ind, diag_col_ind) = ddz[j];  
        }

        // Added constraint to square mat and close nullspace
        (*LHS)(diag_row_ind, 4*N+2) = 1.; 
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

            (*LHS)(diag_row_ind, diag_col_ind) = ddx[j];  
        }
        for (unsigned int j = 0; j < st.size(); j++) {
            unsigned int diag_col_ind = st[j] + 1*N;

            (*LHS)(diag_row_ind, diag_col_ind) = ddy[j];  
        }
        for (unsigned int j = 0; j < st.size(); j++) {
            unsigned int diag_col_ind = st[j] + 2*N;

            (*LHS)(diag_row_ind, diag_col_ind) = ddz[j];  
        }

        // Added constraint to square mat and close nullspace
        (*LHS)(diag_row_ind, 4*N+3) = 1.;  
    }

    // ------ EXTRA CONSTRAINT ROWS -----
    unsigned int diag_row_ind = 4*N;
    // U
    for (unsigned int j = 0; j < N; j++) {
        unsigned int diag_col_ind = j + 0*N;

        (*LHS)(diag_row_ind, diag_col_ind) = 1.;  
    }

    diag_row_ind++; 
    // V
    for (unsigned int j = 0; j < N; j++) {
        unsigned int diag_col_ind = j + 1*N;

        (*LHS)(diag_row_ind, diag_col_ind) = 1.;  
    }

    diag_row_ind++; 
    // W
    for (unsigned int j = 0; j < N; j++) {
        unsigned int diag_col_ind = j + 2*N;

        (*LHS)(diag_row_ind, diag_col_ind) = 1.;  
    }

    diag_row_ind++; 
    // P
    for (unsigned int j = 0; j < N; j++) {
        unsigned int diag_col_ind = j + 3*N;

        (*LHS)(diag_row_ind, diag_col_ind) = 1.;  
    }

    tm["LHS"]->stop(); 
}

/**************  RHS *****************/ 

void StokesSteadyPDE::fillRHS_ConstViscosity() {

    tm["RHS_continuous"]->start(); 
    // This is our manufactured solution:
    SphericalHarmonic::SphericalHarmonicBase* UU = new SphericalHarmonic::Sph32(); 
    SphericalHarmonic::SphericalHarmonicBase* VV = new SphericalHarmonic::Sph32105(); 
    SphericalHarmonic::SphericalHarmonicBase* WW = new SphericalHarmonic::Sph32(); 
    SphericalHarmonic::SphericalHarmonicBase* PP = new SphericalHarmonic::Sph32(); 

    std::vector<NodeType>& nodes = grid_ref.getNodeList(); 

    //------------- Fill F -------------
    RHS_continuous = new VecType(ncols);
    U_continuous = new VecType(ncols);

    std::cout << "Filling continuous RHS\n";
    // U
    for (unsigned int j = 0; j < N; j++) {
        unsigned int row_ind = j + 0*N;
        NodeType& node = nodes[j]; 
        double Xx = node.x(); 
        double Yy = node.y(); 
        double Zz = node.z(); 

        (*U_continuous)(row_ind) = UU->eval(Xx,Yy,Zz); 
        (*RHS_continuous)[row_ind] = -UU->lapl(Xx,Yy,Zz) + PP->d_dx(Xx,Yy,Zz);  
    }
#if 1

    // V
    for (unsigned int j = 0; j < N; j++) {
        unsigned int row_ind = j + 1*N;
        NodeType& node = nodes[j]; 
        double Xx = node.x(); 
        double Yy = node.y(); 
        double Zz = node.z(); 
        //double rr = sqrt(node.x()*node.x() + node.y()*node.y() + node.z()*node.z());
        //double dir = node.y();

        // (*RHS_continuous)(row_ind) = (Ra * Temperature(j) * dir) / rr;  
        (*U_continuous)(row_ind) = VV->eval(Xx,Yy,Zz); 
        (*RHS_continuous)(row_ind) = -VV->lapl(Xx,Yy,Zz) + PP->d_dy(Xx,Yy,Zz);  
    }

    // W
    for (unsigned int j = 0; j < N; j++) {
        unsigned int row_ind = j + 2*N;
        NodeType& node = nodes[j];
        double Xx = node.x(); 
        double Yy = node.y(); 
        double Zz = node.z(); 

        (*U_continuous)(row_ind) = WW->eval(Xx,Yy,Zz); 
        (*RHS_continuous)(row_ind) = -WW->lapl(Xx,Yy,Zz) + PP->d_dz(Xx,Yy,Zz);  
    }

    // P
    for (unsigned int j = 0; j < N; j++) {
        unsigned int row_ind = j + 3*N;
        NodeType& node = nodes[j]; 
        double Xx = node.x(); 
        double Yy = node.y(); 
        double Zz = node.z(); 

        (*U_continuous)(row_ind) = PP->eval(Xx,Yy,Zz); 
        (*RHS_continuous)(row_ind) = UU->d_dx(Xx,Yy,Zz) + VV->d_dy(Xx,Yy,Zz) + WW->d_dz(Xx,Yy,Zz);  
    }
#endif
    // Sum of U
    (*RHS_continuous)(4*N+0) = 0.;

    // Sum of V
    (*RHS_continuous)(4*N+1) = 0.;

    // Sum of W
    (*RHS_continuous)(4*N+2) = 0.;

    // Sum of P
    (*RHS_continuous)(4*N+3) = 0.;
    tm["RHS_continuous"]->stop();

    if (discreteRHS) {
        this->fillRHS_discrete();
    }
}
    
void StokesSteadyPDE::fillRHS_discrete() {
    std::cout << "Filling discrete RHS\n";
    RHS_discrete = new VecType(ncols);
    tm["RHS_discrete"]->start();
    // This takes 21832.6 ms for N=10201.  Slow. But its the same for ublas and viennacl.
    *RHS_discrete = prod(*LHS, *U_continuous); 
    tm["RHS_discrete"]->stop();
}

void StokesSteadyPDE::writeToFile() 
{
    // Write both to disk (spy them in Matlab)
    viennacl::io::write_matrix_market_file(*LHS, "LHS.mtx");
    std::cout << "Wrote LHS.mtx\n"; 

    this->write_to_file(*RHS_discrete, "RHS_discrete.mtx");
    this->write_to_file(*RHS_continuous, "RHS_continuous.mtx");
    this->write_to_file(*U_continuous, "U_continuous.mtx");

    std::cout << "*** NOTE: If you want to visualize RHS or Solution (U) you need to load the reordered grid produced by the stencil generator ***\n"; 
}

void StokesSteadyPDE::getDivOperator() {

    // TODO: figure out ordering here
    DIV_operator = new MatType(boost::numeric::ublas::project(*LHS, boost::numeric::ublas::range(3*N,4*N+4), boost::numeric::ublas::range(0,4*N+4)));
    viennacl::io::write_matrix_market_file(*DIV_operator, "DIV_operatorerator.mtx"); 

}


void StokesSteadyPDE::solve() 
{
    if (cuthillMckeeReordering) {
        this->solve_reordered(); 
    } else {
        this->solve_original();
    }
}


void StokesSteadyPDE::solve_original() 
{
#if 0
    U_computed = new VecType(ncols);
    std::cout << "Solving...." << std::endl;
    // Solve L u = F
    // Write solution to disk

    // Update U_G with the content from U
    // (reshape to fit into U_G. For stokes we assume we're in a vector type <u,v,w,p>)
    // SCALAR  std::copy(U_vec.begin(), U_vec.end(), U_G.begin()); 

    // Solve system using Stabilized BiConjugate Gradients from ViennaCL
    tm["gmres"]->start(); 

    viennacl::linalg::gmres_tag custom_gmres(1e-8, 100, 30);
    
    *U_computed = viennacl::linalg::solve(*LHS, *RHS_continuous, custom_gmres);
    tm["gmres"]->stop(); 
    
    std::cout << "No. of iters: " << custom_gmres.iters() << std::endl;
    std::cout << "Est. error: " << custom_gmres.error() << std::endl;


    // NOTE: the preconditioners dont work on our current structure. We need to
    // reorder the matrix to get entries on the diagonal in order for ilut to work. Im
    // sure this is why the jacobi also fails. 
#if 0
    viennacl::linalg::ilut_precond< MatType >  ublas_ilut(*LHS, viennacl::linalg::ilut_tag());
    *U_computed = viennacl::linalg::solve(*LHS, *RHS_discrete, viennacl::linalg::gmres_tag(1e-6, 20), ublas_ilut);
#else 
#if 0
    viennacl::linalg::jacobi_precond< MatType > ublas_jacobi(*LHS, viennacl::linalg::jacobi_tag());
    *U_computed = viennacl::linalg::solve(*LHS, *RHS_discrete, viennacl::linalg::gmres_tag(1e-6, 20), ublas_jacobi);
#else 
#if 0
    *U_reordered = viennacl::linalg::solve(*LHS_reordered, *RHS_reordered, viennacl::linalg::gmres_tag(1e-8, 100));
    this->write_to_file(*U_reordered, "U_reordered.mtx");
    this->get_original_order(*U_reordered, *U_computed);
#endif 
#endif 
#endif 
#if 0
    // LU with Partial Pivoting
    *U_computed = *RHS_discrete; 
    ublas::permutation_matrix<double> P1(LHS->size1());

    ublas::lU_factorize(*LHS, P1);
    ublas::lU_substitute(*LHS, P1, *U_computed);
#else 

#endif 

    std::cout << "Done with solve\n"; 

    this->write_to_file(*U_computed, "U_computed.mtx");

//    this->write_to_file(VecType(prod(*DIV_operator, *U_computed)), "div.mtx"); 
#endif
}


void StokesSteadyPDE::solve_reordered() 
{
    U_computed = new VecType(ncols);
    std::cout << "Solving...." << std::endl;
    // Solve L u = F
    // Write solution to disk

    // Update U_G with the content from U
    // (reshape to fit into U_G. For stokes we assume we're in a vector type <u,v,w,p>)
    // SCALAR  std::copy(U_vec.begin(), U_vec.end(), U_G.begin()); 

    // Solve system using Stabilized BiConjugate Gradients from ViennaCL
    // *U_computed = viennacl::linalg::solve(*LHS, *RHS_discrete, viennacl::linalg::bicgstab_tag(1.e-24, 3000));

    // NOTE: the preconditioners dont work on our current structure. We need to
    // reorder the matrix to get entries on the diagonal in order for ilut to work. Im
    // sure this is why the jacobi also fails. 
#if 0
    viennacl::linalg::ilut_precond< MatType >  ublas_ilut(*LHS, viennacl::linalg::ilut_tag());
    *U_computed = viennacl::linalg::solve(*LHS, *RHS_discrete, viennacl::linalg::gmres_tag(1e-6, 20), ublas_ilut);
#else 
#if 0
    viennacl::linalg::jacobi_precond< MatType > ublas_jacobi(*LHS, viennacl::linalg::jacobi_tag());
    *U_computed = viennacl::linalg::solve(*LHS, *RHS_discrete, viennacl::linalg::gmres_tag(1e-6, 20), ublas_jacobi);
#else 
#if 0
    *U_reordered = viennacl::linalg::solve(*LHS_reordered, *RHS_reordered, viennacl::linalg::gmres_tag(1e-8, 200));
    this->write_to_file(*U_reordered, "U_reordered.mtx");
    this->get_original_order(*U_reordered, *U_computed);
#endif 
#endif 
#endif 
#if 0
    // LU with Partial Pivoting
    *U_computed = *RHS_discrete; 
    ublas::permutation_matrix<double> P1(LHS->size1());

    ublas::lU_factorize(*LHS, P1);
    ublas::lU_substitute(*LHS, P1, *U_computed);
#else 

#endif 

    std::cout << "Done with solve\n"; 

    this->write_to_file(*U_computed, "U_computed.mtx");

//    this->write_to_file(VecType(prod(*DIV_operator, *U_computed)), "div.mtx"); 
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
    for (size_t i = 0; i < vec.size(); i++) {
        fout << std::scientific << vec[i] << std::endl;
    }
    fout.close();
    std::cout << "Wrote " << filename << std::endl;
}

