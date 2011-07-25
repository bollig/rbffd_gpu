
#include "vortex_rollup.h"
#include "utils/geom/cart2sph.h"

// This should assemble a matrix L of weights which can be used to solve the PDE
void VortexRollup::assemble() {
    if (!weightsPrecomputed) {
        der_ref.computeAllWeightsForAllStencils();
    }

    static int assembled = 0; 

#if 0
    if (!assembled) {

        unsigned int n_stencils = grid_ref.getStencilsSize();

        boost::numeric::ublas::compressed_matrix<double> L_host(n_stencils, n_stencils);
    if (this->useHyperviscosity) {
        std::cout << "PDE USING HYPERVISCOSITY\n";
        for (unsigned int j = 0; j < n_stencils; j++) {

            NodeType& x_j = grid_ref.getNode(j);

            StencilType& st = grid_ref.getStencil(j); 

            // Get just the dPhi/dr weights
            double* dPhi_dlambda = der_ref.getStencilWeights(RBFFD::LAMBDA, j);
            double* hv_filter = der_ref.getStencilWeights(RBFFD::HV, j);
            // Match the weights with the scalars
            for (unsigned int i = 0; i < st.size(); i++) {
                L_host(st[0], st[i]) = dPhi_dlambda[i] + hv_filter[i];
                //  printf ("lhost(%d, st[%d]) = w[%d] = %g\n", st[0], st[i], i, val);
            }
        }
    } else {
        for (unsigned int j = 0; j < n_stencils; j++) {
            NodeType& x_j = grid_ref.getNode(j);
            StencilType& st = grid_ref.getStencil(j); 
            // Get just the dPhi/dr weights
            double* dPhi_dlambda = der_ref.getStencilWeights(RBFFD::LAMBDA, j);

            // Match the weights with the scalars
            for (unsigned int i = 0; i < st.size(); i++) {
                L_host(st[0], st[i]) = dPhi_dlambda[i];
                //  printf ("lhost(%d, st[%d]) = w[%d] = %g\n", st[0], st[i], i, val);
            }
        }
    }
#if 0
        std::ofstream fout;
        fout.open("L_host.mtx"); 
        for (unsigned int i =0; i < n_stencils; i++) {
            for (unsigned int j =0; j < n_stencils-1; j++) {
                fout << L_host(i,j) << ",";
            }
            fout << L_host(i,n_stencils-1);
            fout << std::endl;
        }
        fout.close();
        //std::cout << L_host << std::endl;
#endif 
        this->D_N = L_host; 
        assembled = 1;
    }
#endif
}


// This will apply the weights appropriately for an explicit (del_u = L*u) or implicit (u = L^-1 del_u)
void VortexRollup::solve(std::vector<SolutionType>& u_t, std::vector<SolutionType>* f_out, unsigned int n_stencils, unsigned int n_nodes, double t) {

    // Fill DM with interpolation weights for each stencil
    // The stencils will interpolate the function value at the stencil center
    // using a weighted combination of values in the stencil. Seems
    // counter-intuitive, but once we can interpolate our values at the stencil
    // centers we can start approximating derivatives for the stencil center
    // with weighted combinations. 

    
    // dh/dt = - W * D * h
    // where W = diag(w(theta_p)) and 
    //       D = B A^-1
    // with B_i,j = cos(theta_p_i) cos(theta_p_j) * sin(lambda_i - lambda_j) (1/r)(d theta / dr)
    //
    double rho0 = 3.;
    double gamma = 5.;

    std::vector<SolutionType> dh_d_lambda(n_stencils);  
    
    der_ref.applyWeightsForDeriv(RBFFD::LAMBDA, u_t, dh_d_lambda, true); 

#if 0
    boost::numeric::ublas::vector<double> dh_d_lambda(n_stencils);
    boost::numeric::ublas::vector<double> h(n_stencils);
    for (unsigned int i = 0; i < n_stencils; i++) {
        h(i) = u_t[i]; 
    }
    dh_d_lambda = boost::numeric::ublas::prod(this->D_N, h); 
#endif 

    for (unsigned int i = 0; i < n_stencils; i++) {
        NodeType& v = grid_ref.getNode(i);
        
        sph_coords_type spherical_coords = cart2sph(v.x(), v.y(), v.z());
        double theta_p = spherical_coords.phi; 
        double phi_p = spherical_coords.theta; 
        double temp = spherical_coords.r; 

        // From Natasha's email: 
        // 7/7/11 4:46 pm
        double rho_p = rho0 * cos(theta_p); 

        // NOTE: The sqrt(2) is written as sqrt(3.) in the paper with Grady.
        // Natasha verified sqrt(3) is required
        // Also, for whatever reason sech is not defined in the C standard
        // math. I provide it in the cart2sph header. 
        double Vt = (3.* sqrt(3.) / 2.) * (sech(rho_p) * sech(rho_p)) * tanh(rho_p); 
        double w_theta_P = (fabs(rho_p) < 4. * DBL_EPSILON) ? 0. : Vt / rho_p;

        // dh/dt + u' / cos(theta') * dh/d(lambda) = 0
        // u' = w(theta') cos(theta')
        // So, dh/dt = - w(theta_P) * dh/d(lambda)
        (*f_out)[i] = - (w_theta_P) * dh_d_lambda[i];
    }
    if (useHyperviscosity) {
        std::cout << "Using Hyperviscosity Filter\n";
        // Filter is ONLY applied after the rest of the RHS is evaluated
        std::vector<SolutionType> hv_filter; 
        der_ref.applyWeightsForDeriv(RBFFD::HV, u_t, hv_filter, true); 
        for (unsigned int i =0; i < n_stencils; i++) {
            (*f_out)[i] += hv_filter[i]; 
        }
    }
}

#if 0
void VortexRollup::advance(TimeScheme which, double delta_t) {

    unsigned int nb_stencils = grid_ref.getStencilsSize(); 
    unsigned int nb_nodes = grid_ref.getNodeListSize(); 
    std::vector<NodeType>& nodes = grid_ref.getNodeList();

    // backup the current solution so we can perform intermediate steps
    std::vector<double> original_solution = this->U_G; 
    std::vector<double>& s = this->U_G; 
    std::vector<SolutionType> feval1(nb_stencils);  

    // If we need to assemble a matrix L for solving implicitly, this is the routine to do that. 
    // For explicit schemes we can just solve for our weights and have them stored in memory.
    this->assemble(); 

    // This routine will apply our weights to "s" in however many intermediate steps are required
    // and store the results in feval1
    this->solve(s, &feval1, nb_stencils, nb_nodes, cur_time); 

    // compute u^* = u^n + dt*lapl(u^n)
    for (unsigned int i = 0; i < nb_nodes; i++) {
        NodeType& v = nodes[i];
        //printf("dt= %f, time= %f\n", dt, time);
        // FIXME: allow the use of a forcing term 
        double f = 0.;//force(i, v, time*dt);
        s[i] = s[i] + delta_t* ( feval1[i] + f);
        double e1 = exact_ptr->at(grid_ref.getNode(i), cur_time); 
        printf("%f %f %f\n", s[i], e1, s[i] - e1);
   }

//    cur_time = start_time;
    cur_time += delta_t; 
}
#endif 


void VortexRollup::setupTimers() {

}



