#ifndef __COSINE_BELL_VCL_H__
#define __COSINE_BELL_VCL_H__

#include "pdes/time_dependent_pde_vcl.h"
#include "utils/geom/cart2sph.h"

#include <iomanip>
#include <iostream>
#include <sstream> 
#include <map>
#include <fstream> 
#include <typeinfo> 

// ----------------  Snipped from ViennaCL 1.3.1 examples (custom-kernels.cpp) ------------------------
//
// Custom compute kernels which compute an elementwise product/division of two vectors
// Input: v1 ... vector
//        v2 ... vector
// Output: result ... vector
//
// Algorithm: set result[i] <- - (v1[i] * v2[i])
//            or  result[i] <- v1[i] * v2[i]
//            (in MATLAB notation this is something like 'result = -(v1 .* v2)' and 'result = v1 .* v2');
// NOTE: need to allow support for double manually, no elegant way to my knowledge to do this in VCL
const char * my_compute_program = 
"#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n\n"
"__kernel void elementwise_prod_neg(\n"
"          __global const double * vec1,\n"
"          __global const double * vec2, \n"
"          __global double * result,\n"
"          unsigned int size) \n"
"{ \n"
"  for (unsigned int i = get_global_id(0); i < size; i += get_global_size(0))\n"
"    result[i] = - vec1[i] * vec2[i];\n"
"};\n\n"
"__kernel void elementwise_prod(\n"
"          __global const double * vec1,\n"
"          __global const double * vec2, \n"
"          __global double * result,\n"
"          unsigned int size) \n"
"{ \n"
"  for (unsigned int i = get_global_id(0); i < size; i += get_global_size(0))\n"
"    result[i] = vec1[i] * vec2[i];\n"
"};\n\n";

// ---------------------------------------------- END Snip





class CosineBell_VCL : public TimeDependentPDE_VCL
{
    protected: 
        int useHyperviscosity; 
        // RADIUS OF THE SPHERE: 
        double a;
        // RADIUS OF THE BELL
        double R;
        // ANGLE OF ADVECTION AROUND SPHERE (from Equator and Greenwhich Mean Time) 
        double alpha;
        // The initial velocity (NOTE: scalar in denom is 12 days in seconds)
        double u0;
        double time_for_revolution; 

        VCL_VEC_t* vel_u; 
        VCL_VEC_t* vel_v;

        viennacl::ocl::kernel * my_kernel_mul; 

    public:
        CosineBell_VCL(Domain* grid, RBFFD_VCL* der, Communicator* comm, double earth_radius, double velocity_angle, double one_revolution_in_seconds, int gpuType, int useHyperviscosity, bool weightsComputed=false) 
            :
                TimeDependentPDE_VCL(grid, der, comm, gpuType, weightsComputed), 

                useHyperviscosity(useHyperviscosity), 
                a(1.), //6.37122*10^6; % radius of earth in meters
                // ANGLE OF ADVECTION AROUND SPHERE (from Equator and Greenwhich Mean Time) 
                alpha(M_PI/2.),
                // Time in seconds to complete one revolution
                time_for_revolution(1036800.)

                {

                    // RADIUS OF THE BELL
                    R = a/3.;
                    // The initial velocity (NOTE: scalar in denom is 12 days in seconds)
                    u0 = (2.*M_PI*a)/time_for_revolution; 

                    // Fill in constants
                    // Allocate GPU buffers for velocity
                    // load solve kernel
                    //    std::string solve_str = #include "cosine_bell_solve.cl"
                    // initialize the TimeDependentPDE_VCL superclass
                    this->initialize();
                    this->assembleDM();

#if 0
                    VCL_VEC_t dh_dlambda(n_stencils);  
                    VCL_VEC_t dh_dtheta(n_stencils);  
                    VCL_VEC_t hv_filter(n_stencils);  
#endif 

                    unsigned int n_stencils = grid_ref.getStencilsSize(); 
                    this->vel_u = new VCL_VEC_t(n_stencils); 
                    this->vel_v = new VCL_VEC_t(n_stencils); 
                }

        virtual void assembleDM() {

            unsigned int n_stencils = grid_ref.getStencilsSize(); 
            unsigned int n_nodes = grid_ref.getNodeListSize(); 

            //
            // Set up the OpenCL program given in my_compute_kernel:
            // A program is one compilation unit and can hold many different compute kernels.
            //
            viennacl::ocl::program & my_prog = viennacl::ocl::current_context().add_program(my_compute_program, "my_compute_program");
            my_prog.add_kernel("elementwise_prod_neg");  //register elementwise product kernel
            my_prog.add_kernel("elementwise_prod");  //register elementwise product kernel

            // Now we can get the kernels from the program 'my_program'.
            // (Note that first all kernels need to be registered via add_kernel() before get_kernel() can be called,
            // otherwise existing references might be invalidated)
            //
            my_kernel_mul = &(my_prog.get_kernel("elementwise_prod_neg"));

            std::cout << "----------> ViennaCL registered elementwise_prod\n";

            // Need to assemble a diagonal matrix containing velocity 
            UBLAS_VEC_t vel_u_cpu(n_stencils); 
            UBLAS_VEC_t vel_v_cpu(n_stencils); 

            for (unsigned int i = 0; i < n_stencils; i++) {
                NodeType& v = grid_ref.getNode(i);

                sph_coords_type spherical_coords = cart2sph(v.x(), v.y(), v.z());
                // longitude, latitude respectively:
                double lambda = spherical_coords.theta; 
                double theta = spherical_coords.phi; 

                vel_u_cpu[i] =   u0 * (cos(theta) * cos(alpha) + sin(theta) * cos(lambda) * sin(alpha)); 
                vel_u_cpu[i] /= a; 
                vel_v_cpu[i] = - u0 * (sin(lambda) * sin(alpha));
                vel_v_cpu[i] /= a; 
            }

            this->vel_u = new VCL_VEC_t(vel_u_cpu.size());
            viennacl::copy(vel_u_cpu.begin(), vel_u_cpu.end(), this->vel_u->begin()); 
            this->vel_v = new VCL_VEC_t(vel_v_cpu.size());
            viennacl::copy(vel_v_cpu.begin(), vel_v_cpu.end(), this->vel_v->begin()); 
        }

        virtual void enforceBoundaryConditions(std::vector<SolutionType>& y_t, double t) { 
            //DO NOTHING;
        }

        virtual double getMaxVelocity(double at_time) { 
            // The angular velocity is constant and will never be more than this. 
            // (At theta = pi/2, lambda = 0 and alpha = pi/2, the velocity is u0
            return u0; 
        }

        virtual void solve(VCL_VEC_t& y_t, VCL_VEC_t& f_out, unsigned int n_stencils, unsigned int n_nodes, double t)
        {    

            // Apply DM as product
            VCL_VEC_t dh_dlambda = viennacl::linalg::prod(*(der_ref_gpu.getGPUWeights(RBFFD::LAMBDA_i)), y_t);
            VCL_VEC_t dh_dtheta  = viennacl::linalg::prod(*(der_ref_gpu.getGPUWeights(RBFFD::THETA_i)), y_t);
            VCL_VEC_t hv_filter  = viennacl::linalg::prod(*(der_ref_gpu.getGPUWeights(RBFFD::HV_i)), y_t);

            VCL_VEC_t result_mul1(dh_dlambda.size());
            VCL_VEC_t result_mul2(dh_dlambda.size());

            //
            // Launch the kernel with 'vector_size' threads in one work group
            // Note that size_t might differ between host and device. Thus, a cast to cl_uint is necessary for the forth argument.
            //
            viennacl::ocl::enqueue((*my_kernel_mul)(*vel_u, dh_dlambda, result_mul1, static_cast<cl_uint>(dh_dlambda.size())));  
            viennacl::ocl::enqueue((*my_kernel_mul)(*vel_v, dh_dtheta, result_mul2, static_cast<cl_uint>(dh_dlambda.size())));  

            // Compute the dh/dt
            // dh/dt = -(vel_u * dh_dlambda) - (vel_v * dh_dtheta) 
            f_out = (result_mul1 + result_mul2); 

            // Optionally add HV filter
            if (useHyperviscosity) {
                //f_out += hv_filter; 
            }

            // IF we want to write details we need to copy back to host. 
            UBLAS_VEC_t U_approx(f_out.size());
            //copy(f_out.begin(), f_out.end(), U_approx.begin());
            copy(f_out.begin(), f_out.end(), U_approx.begin());

            write_to_file(U_approx, "output/U_gpu.mtx"); 
            std::cout << "ERROR\n";
            exit(-1);   
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


    protected:
        virtual std::string className() {return "cosine_bell_vcl";}
        virtual void solve() {;}
}; 
#endif 

//,  public CosineBell 
//CosineBell(grid, der, comm, earth_radius, velocity_angle, one_revolution_in_seconds, useHyperviscosity, weightsComputed), 
