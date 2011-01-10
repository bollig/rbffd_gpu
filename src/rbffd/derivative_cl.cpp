#include <stdlib.h>
#include <math.h>
#include "derivative_cl.h"

using namespace std;

//----------------------------------------------------------------------
DerivativeCL::DerivativeCL(cl::Context context, vector<Vec3>& rbf_centers_, vector<vector<int> >& stencil_, int nb_bnd_, int dimensions)
: Derivative(rbf_centers_, stencil_, nb_bnd_, dimensions)	
{
	cout << "Inside DerivativeCL constructor" << endl;

    std::cout<<"Loading and compiling OpenCL source for DerivativeCL\n";


	streamsdk::SDKFile file;
    if (!file.open("HelloCL_Kernels.cl")) {
         std::cerr << "We couldn't load CL source code\n";
         return SDK_FAILURE;
    }
	cl::Program::Sources sources(1, std::make_pair(file.source().data(), file.source().size()));

	cl::Program program = cl::Program(context, sources, &err);
	if (err != CL_SUCCESS) {
        std::cerr << "Program::Program() failed (" << err << ")\n";
        return SDK_FAILURE;
    }

    err = program.build(devices);
    if (err != CL_SUCCESS) {

		if(err == CL_BUILD_PROGRAM_FAILURE)
        {
            cl::string str = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[0]);

            std::cout << " \n\t\t\tBUILD LOG\n";
            std::cout << " ************************************************\n";
			std::cout << str.c_str() << std::endl;
            std::cout << " ************************************************\n";
        }

        std::cerr << "Program::build() failed (" << err << ")\n";
        return SDK_FAILURE;
    }

 cl::Kernel kernel(program, "hello", &err);
    if (err != CL_SUCCESS) {
        std::cerr << "Kernel::Kernel() failed (" << err << ")\n";
        return SDK_FAILURE;
    }
    if (err != CL_SUCCESS) {
        std::cerr << "Kernel::setArg() failed (" << err << ")\n";
        return SDK_FAILURE;
    }

        cl::CommandQueue queue(context, devices[0], 0, &err);
    if (err != CL_SUCCESS) {
        std::cerr << "CommandQueue::CommandQueue() failed (" << err << ")\n";
    }



    // NOW WE HAVE A KERNEL PREPPED AND READY TO BE CALLED

	cout << "Allocating GPU memory for " << stencil_.size() << " stencil weights" << endl;

	int solution_size = rbf_centers_.size();
	cout << "Allocating GPU memory for " << solution_size << " solution values" << endl;
}

DerivativeCL::~DerivativeCL() {
	cout << "Freeing OpenCL memory for solution" << endl;
	cout << "Freeing OpenCL memory for stencil weights" << endl;
}

//----------------------------------------------------------------------
//
//	This needs to be offloaded to an OpenCL Kernel
//	
//
void DerivativeCL::computeDeriv(DerType which, double* u, double* deriv, int npts)
{

    cout << "COMPUTING DERIVATIVE: ";
    vector<double*>* weights_p;

    switch(which) {
    case X:
        weights_p = &x_weights;
        //printf("weights_p= %d\n", weights_p);
        //exit(0);
        cout << "X" << endl;
        break;

    case Y:
        weights_p = &y_weights;
        cout << "Y" << endl;
        break;

    case Z:
        //vector<mat>& weights = z_weights;
        weights_p = &z_weights;
        cout << "Z" << endl;
        break;

    case LAPL:
        weights_p = &lapl_weights;
        cout << "LAPL" << endl;
        break;

    default:
        cout << "???" << endl;
        printf("Wrong derivative choice\n");
        printf("Should not happen\n");
        exit(EXIT_FAILURE);
    }

    vector<double*>& weights = *weights_p;

	cout << "Sending " << rbf_centers.size() << " solution updates to GPU" << endl;


    double der;

    for (int i=0; i < stencil.size(); i++) {
        double* w = weights[i];
        vector<int>& st = stencil[i];
        der = 0.0;
        int n = st.size();
        for (int s=0; s < n; s++) {
            der += w[s] * u[st[s]]; 
        }
        deriv[i] = der;
    }



        std::cout<<"Running CL program\n";
    err = queue.enqueueNDRangeKernel(
        kernel, cl::NullRange, cl::NDRange(4, 4), cl::NDRange(2, 2)
    );

    if (err != CL_SUCCESS) {
        std::cerr << "CommandQueue::enqueueNDRangeKernel()" \
            " failed (" << err << ")\n";
       return SDK_FAILURE;
    }

    err = queue.finish();
    if (err != CL_SUCCESS) {
        std::cerr << "Event::wait() failed (" << err << ")\n";
    }


}
//----------------------------------------------------------------------
