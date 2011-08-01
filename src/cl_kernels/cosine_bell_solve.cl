FLOAT solve(__global FLOAT* u_t,
            unsigned int n_stencils,
            unsigned int n_nodes,
            double t)
{
        FLOAT dh_dlambda= applyWeightsForDeriv(RBFFD::LAMBDA, u_t);
        FLOAT dh_dtheta = applyWeightsForDeriv(RBFFD::THETA, u_t);
        FLOAT hv_filter = applyWeightsForDeriv(RBFFD::HV, u_t);

        NodeType& v = grid_ref.getNode(i);

        sph_coords_type spherical_coords = cart2sph(v.x(), v.y(), v.z());
        // longitude, latitude respectively:
        double lambda = spherical_coords.theta;
        double theta = spherical_coords.phi;

        double vel_u =   u0 * (cos(theta) * cos(alpha) + sin(theta) * cos(lambda) * sin(alpha));
        //double vel_v = - u0 * (cos(lambda) * sin(alpha));
        double vel_v = - u0 * (sin(lambda) * sin(alpha));

        // dh/dt + u / cos(theta) * dh/d(lambda) + v * dh/d(theta) = 0
        // dh/dt = - [diag(u/cos(theta)) * D_LAMBDA * h + diag(v/a) * D_THETA * h] + H
     FLOAT f_out = -((vel_u/(a * cos(theta))) * dh_dlambda[i] + (vel_v/a) * dh_dtheta[i]);

    if (useHyperviscosity) {
        // Filter is ONLY applied after the rest of the RHS is evaluated
        for (unsigned int i =0; i < n_stencils; i++) {
            f_out += hv_filter;
        }
    }
}
