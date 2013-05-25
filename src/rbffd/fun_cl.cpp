#include <stdlib.h>
#include <math.h>
#include "fun_cl.h"
#include "timer_eb.h"
#include "common_typedefs.h"     // Declares type FLOAT


using namespace EB;
using namespace std;

//----------------------------------------------------------------------
//
FUN_CL::FUN_CL(DerTypes typesToCompute, Grid* grid, int dim_num, int rank)
: RBFFD_CL(typesToCompute, grid, dim_num, rank)
{
	nb_nodes = grid_ref.getNodeListSize();
	std::vector<StencilType>& stencil_map = grid_ref.getStencils();
	nodes_per_stencil = stencil_map[0].size(); // assumed constant

    this->loadKernel("computeDerivMultiWeightFunKernel", "derivative_kernels.cl");

    this->allocateGPUMem();
    //this->updateStencilsOnGPU(false);
    this->updateStencilsOnGPU(true); //GE
    std::cout << "Done copying stencils to GPU\n";

	// The GPU does not require the nodes
    //this->updateNodesOnGPU(false);
    //this->updateNodesOnGPU(true);
    //std::cout << "Done copying nodes to GPU\n";
}
//----------------------------------------------------------------------
void FUN_CL::allocateGPUMem()
{
	RBFFD_CL::allocateGPUMem();

	printf("entering allocateGPUMem in fun_cl.cpp\n");
    std::vector<StencilType>& stencil_map = grid_ref.getStencils();
    unsigned int float_size = useDouble? sizeof(double) : sizeof(float);
    unsigned int nb_stencils = stencil_map.size();

	printf("nb_stencils= %d\n", nb_stencils);
	printf("nb_nodes= %d\n", nb_nodes);
	printf("nodes_per_stencil= %d\n", nodes_per_stencil);
	printf("float_size= %d\n", float_size);

	#if 0
	SuperBuffer<double> sup_deriv;
	SuperBuffer<double> sup_deriv_x;
	SuperBuffer<double> sup_deriv_y;
	SuperBuffer<double> sup_deriv_z;
	SuperBuffer<double> sup_deriv_l;
	sup_function = SuperBuffer<double>(u);
	SuperBuffer<double> sup_weights[NUM_DERIVATIVE_TYPES];
	SuperBuffer<int>    sup_stencils;
	#endif

	sup_stencils = SuperBuffer<int>(100);
	sup_all_weights = SuperBuffer<double>(nb_nodes*nb_stencils*4);

	// CHECK that stencils were computed (NOT DONE)

	#if 0
	all_weights_bytes = nb_nodes * stencil_map[0].size() * float_size * 4;
	std::cout << "allocateGPU, stencil size: " << stencil_map[0].size() << std::endl;
	gpu_all_weights = cl::Buffer(context, CL_MEM_READ_WRITE, all_weights_bytes, NULL, &err);
	printf("*** mem_size = %d\n", getSize(gpu_all_weights));
	printf("*** all_weights_bytes= %d\n", all_weights_bytes);

	bytesAllocated += all_weights_bytes;
	#endif

	#if 0
    function_mem_bytes = nb_nodes * float_size * 4;
//printf("...nb_nodes= %d\n", nb_nodes);
//printf("...float_size= %d\n", float_size);
    gpu_function = cl::Buffer(context, CL_MEM_READ_ONLY, function_mem_bytes, NULL, &err);
//std::cout << "...function_mem_bytes= " << function_mem_bytes << "\n"; 
//printf("gpu_function size (bytes) = %d\n", getSize(gpu_function)); exit(0);
	bytesAllocated += function_mem_bytes;

	// transfer all derivatives for all four functions
    deriv_mem_bytes = 4 * nb_stencils * float_size;
    gpu_deriv_x_out = cl::Buffer(context, CL_MEM_READ_WRITE, deriv_mem_bytes, NULL, &err);
    gpu_deriv_y_out = cl::Buffer(context, CL_MEM_READ_WRITE, deriv_mem_bytes, NULL, &err);
    gpu_deriv_z_out = cl::Buffer(context, CL_MEM_READ_WRITE, deriv_mem_bytes, NULL, &err);
    gpu_deriv_l_out = cl::Buffer(context, CL_MEM_READ_WRITE, deriv_mem_bytes, NULL, &err);
    bytesAllocated += deriv_mem_bytes*4; // 4 derivatives 

    gpu_nodes = cl::Buffer(context, CL_MEM_READ_ONLY, nodes_mem_bytes, NULL, &err);
    bytesAllocated += nodes_mem_bytes;

    std::cout << "Allocated: " << bytesAllocated << " bytes (" << ((bytesAllocated / 1024.)/1024.) << "MB)" << std::endl;
	#endif
}
//----------------------------------------------------------------------
void FUN_CL::calcDerivs(SuperBuffer<double>& u, SuperBuffer<double>& deriv_x, 
			SuperBuffer<double>& deriv_y, SuperBuffer<double>& deriv_z, SuperBuffer<double>& deriv_l, bool isChangedU)
//void FUN_CL::applyWeightsForDerivDouble(unsigned int start_indx, unsigned int nb_stencils, SuperBuffer& u, SuperBuffer& deriv_x, 
			//SuperBuffer& deriv_y, SuperBuffer& deriv_z, SuperBuffer& deriv_l, bool isChangedU)
{
	printf("applyWeightsFoDerivDouble using SuperBuffer arguments\n");
	if (isChangedU) u.copyToDevice();
	sup_all_weights.copyToDevice();

    err = queue.finish(); // added by GE
    tm["applyWeights"]->start();
    unsigned int nb_stencils = nb_nodes;

    try {
        int i = 0;
        kernel.setArg(i++, sup_stencils.dev); //gpu_stencils);
        kernel.setArg(i++, sup_stencils.dev); //gpu_all_weights);
        kernel.setArg(i++, u.dev);              // 4 functions
        kernel.setArg(i++, deriv_x.dev);           // COPY_OUT
        kernel.setArg(i++, deriv_y.dev);           // COPY_OUT
        kernel.setArg(i++, deriv_z.dev);           // COPY_OUT
        kernel.setArg(i++, deriv_l.dev);           // COPY_OUT
        //FIXME: we want to pass a unsigned int for maximum array lengths, but OpenCL does not allow
        //unsigned int arguments at this time
        kernel.setArg(i++, sizeof(unsigned int), &nb_stencils);               // const
        unsigned int stencil_size = grid_ref.getMaxStencilSize();
        kernel.setArg(i++, sizeof(unsigned int), &stencil_size);            // const
        //kernel.setArg(i++, sizeof(unsigned int), &stencil_padded_size);            // const
        std::cout << "Set " << i << " kernel args\n";
    } catch (cl::Error er) {
        printf("[setKernelArg] ERROR: %s(%s)\n", er.what(), oclErrorString(er.err()));
    }

	printf("before enqueue kernel\n");
	enqueueKernel(kernel, cl::NDRange(nb_stencils), cl::NullRange, true);
	printf("after enqueue kernel\n");
    tm["applyWeights"]->end();
}
//----------------------------------------------------------------------
void FUN_CL::applyWeightsForDerivDouble(unsigned int start_indx, unsigned int nb_stencils, double* u, double* deriv_x, double* deriv_y, double* deriv_z, double* deriv_l, bool isChangedU)
{
	cout << "****** enter of FUN_CL::applyWeightsForDerivativeDouble ******\n";

    if (isChangedU) {
		//for (int i=start_indx; i < 10; i++) printf("u[%d] = %f\n", i, u[start_indx+i]); // ok
//printf("4*nb_stencils*8= %d\n", 4*nb_stencils*8);
		copyArrayToGPU(&u[start_indx], gpu_function);  // DOES NOT WORK (assumes start_indx=0)
    }

    // Will only update if necessary
    // false here implies that we should not block on the update to finish
    this->updateWeightsOnGPU(true);
	printf("after updateweights\n");

    err = queue.finish(); // added by GE
    tm["applyWeights"]->start();

	double* all_weights = new double[nb_stencils*4]; 

    try {
        int i = 0;
        kernel.setArg(i++, sup_stencils.dev); //gpu_stencils);
        kernel.setArg(i++, sup_stencils.dev); //gpu_all_weights);
        kernel.setArg(i++, gpu_function);              // 4 functions
        kernel.setArg(i++, gpu_deriv_x_out);           // COPY_OUT
        kernel.setArg(i++, gpu_deriv_y_out);           // COPY_OUT
        kernel.setArg(i++, gpu_deriv_z_out);           // COPY_OUT
        kernel.setArg(i++, gpu_deriv_l_out);           // COPY_OUT
        //FIXME: we want to pass a unsigned int for maximum array lengths, but OpenCL does not allow
        //unsigned int arguments at this time
        unsigned int nb_stencils = grid_ref.getStencilsSize();
        kernel.setArg(i++, sizeof(unsigned int), &nb_stencils);               // const
        unsigned int stencil_size = grid_ref.getMaxStencilSize();
        kernel.setArg(i++, sizeof(unsigned int), &stencil_size);            // const
        //kernel.setArg(i++, sizeof(unsigned int), &stencil_padded_size);            // const
        std::cout << "Set " << i << " kernel args\n";
    } catch (cl::Error er) {
        printf("[setKernelArg] ERROR: %s(%s)\n", er.what(), oclErrorString(er.err()));
    }

	#if 0
	// User specifies work group size
	int items_per_group = 8;
	int nn = nb_stencils % items_per_group;
	int nd = nb_stencils / items_per_group;
	int tot_items = (nn != 0) ? (nd+1)*items_per_group : nb_stencils; 
	std::cout << "** nb_stencils: " << nb_stencils << std::endl;
	std::cout << "** nb_stencils % items_per_group: " << nn << std::endl;
	std::cout << "** nb_stencils / items_per_group: " << nd << std::endl;
	std::cout << "** total number items: " << tot_items << std::endl;
	std::cout << "** items per group: " << items_per_group << std::endl;
	enqueueKernel(kernel, cl::NDRange(tot_items), cl::NDRange(items_per_group), true);
	#else
	// Let OpenCL choose the work group size
	
	printf("before enqueue kernel\n");
	enqueueKernel(kernel, cl::NDRange(nb_stencils), cl::NullRange, true);
	printf("after enqueue kernel\n");
	#endif
    tm["applyWeights"]->end();


    if (4*nb_stencils *sizeof(double) != deriv_mem_bytes) { 
        std::cout << "npts*sizeof(double) [" << nb_stencils*sizeof(double) << "] != deriv_mem_bytes [" << deriv_mem_bytes << "]" << std::endl;
        exit(EXIT_FAILURE);
    }

	exit(0);

	// not required
	//copyResultsToHost(deriv_x, deriv_y, deriv_z, deriv_l);
}
//----------------------------------------------------------------------
void FUN_CL::updateWeightsDouble(bool forceFinish)
{
// simply create a large array of zeros.
//

	std::cout << "GE enter updateWeightsDouble\n";
    if (weightsModified) {

        tm["sendWeights"]->start();
        unsigned int weights_mem_size = gpu_stencil_size * sizeof(double);

        std::cout << "[FUN_CL] Writing weights to GPU memory\n";

        unsigned int nb_stencils = grid_ref.getStencilsSize();
		unsigned int  tot_elements = gpu_stencil_size * nb_stencils * 4;

		double* all_weights = new double [tot_elements];
		for (int i=0; i < tot_elements; i++) {
			all_weights[i] = 0.0;
		}

        err = queue.enqueueWriteBuffer(gpu_all_weights, CL_TRUE, 0, tot_elements*sizeof(int), &(all_weights[0]), NULL, &event);
        if (forceFinish) {
            queue.finish();
		}


        if ((nb_stencils * stencil_padded_size) != gpu_stencil_size) {
            // Critical error between allocate and update
            std::cout << "nb_stencils*stencil_padded_size != gpu_stencil_size" << std::endl;
            exit(EXIT_FAILURE);
        }

        tm["sendWeights"]->end();

        weightsModified = false;

    } else {
        //        std::cout << "No need to update gpu_weights" << std::endl;
    }
}
//----------------------------------------------------------------------
#if 0
void RBFFD_CL::updateFunctionDouble(unsigned int start_indx, unsigned int nb_vals, double* u, bool forceFinish)
{
    //    cout << "Sending " << nb_nodes << " solution updates to GPU: (bytes)" << function_mem_bytes << endl;


    if (function_mem_bytes != nb_vals*sizeof(double)) {
        std::cout << "function_mem_bytes != nb_nodes*sizeof(double)" << std::endl;
		std::cout << "nb_vals= " << nb_vals << "\n";
		std::cout << "function_mem_bytes= " << function_mem_bytes << "\n";
		std::cout << "nb_vals= " << nb_vals << "\n";
        exit(EXIT_FAILURE);
    } else {
		std::cout << "Updating solution: " << function_mem_bytes << " bytes \n";
		std::cout << start_indx << ", " << nb_vals << "\n";
    }
    // TODO: mask off fields not update
    err = queue.enqueueWriteBuffer(gpu_function, CL_TRUE, start_indx*sizeof(double), function_mem_bytes, &u[start_indx], NULL, &event);
    //    queue.flush();

    if (forceFinish) {
        queue.finish();
    }
}
#endif
//----------------------------------------------------------------------
//
