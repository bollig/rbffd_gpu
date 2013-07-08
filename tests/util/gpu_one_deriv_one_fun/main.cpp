#include "utils/comm/communicator.h"
#include <stdlib.h>

#include <algorithm>
#include "pdes/parabolic/heat_pde.h"

#include "grids/regulargrid.h"

#include "rbffd/rbffd_cl.h"
#include "rbffd/fun_cl.h"
#include "utils/random.h"
#include "utils/norms.h"

#include "exact_solutions/exact_regulargrid.h"

#include "timer_eb.h"

vector<double> u_cpu;
RBFFD_CL::SuperBuffer<double> u_gpu;

vector<double> xderiv_cpu, uderiv_cpu;
vector<double> yderiv_cpu, vderiv_cpu;
vector<double> zderiv_cpu, wderiv_cpu;
vector<double> lderiv_cpu, pderiv_cpu;

vector<double> u,v,w,p;
vector<double> ux,uy,uz,ul;
vector<double> vx,vy,vz,vl;
vector<double> wx,wy,wz,wl;
vector<double> px,py,pz,pl;

// derivatives with respect to (x,y,z,lapl)
RBFFD_CL::SuperBuffer<double> uderiv_gpu, xderiv_gpu;
RBFFD_CL::SuperBuffer<double> vderiv_gpu, yderiv_gpu;
RBFFD_CL::SuperBuffer<double> wderiv_gpu, zderiv_gpu;
RBFFD_CL::SuperBuffer<double> pderiv_gpu, lderiv_gpu;

RBFFD_CL::SuperBuffer<double> deriv4_gpu;


// Sames types as in rbffd/fun_cl.h
enum KernelType {FUN_KERNEL, FUN_INV_KERNEL, FUN_DERIV4_KERNEL,
    FUN1_DERIV4_WEIGHT4,
    //FUN1_DERIV1_WEIGHT4,
    FUN4_DERIV4_WEIGHT4,
    FUN4_DERIV4_WEIGHT4_INV};
KernelType kernel_type;

Grid* grid;
int dim;
int stencil_size;
int nx, ny, nz;
int use_gpu;
EB::TimerList tm; 
ProjectSettings* settings;
RBFFD* der_cpu;
FUN_CL* der;
//RBFFD_CL* der;

using namespace std;

//----------------------------------------------------------------------
#if 0
double gigaFlops(int npts, int nsten, double time)
// time: one matrix/vector multiply  (ms)
{
	double gflops = (2.*npts*nsten)/(time*1.e-3)*1.e-9;
	return(gflops);
	gigaFlops(nx*ny*nz, 
}
#endif
//----------------------------------------------------------------------
void printTimes(std::vector<double>& timings, int npts, int stencil_size, int nb_derivs)
{
	double mean = 0.;
	double std = 0.;
	for (int i=0; i < timings.size(); i++) {
		mean += timings[i];
		std += timings[i]*timings[i];
	}
	mean /= timings.size();
	std = sqrt(std/timings.size()-mean*mean);
	double gflop = 2.*npts*stencil_size*nb_derivs*1.e-9 / (mean*1.e-3) ;
	printf("mean times: ");
	for (int i=0; i < timings.size(); i++) printf("%f, ", timings[i]);
	double time_per_deriv = mean/nb_derivs;
	printf("\n");
	printf("mean time= %f (ms), standard deviation= %f (ms), Gflops: %f, time per derivative: %f (ms)\n", mean, std, gflop, time_per_deriv);
}
//----------------------------------------------------------------------
typedef std::vector<double> VD;
void vectorBreakup(VD& v4_src, VD& va, VD& vb, VD& vc, VD& vd)
{
    int sz = v4_src.size() >> 2;
    va.assign(&v4_src[0],    &v4_src[1*sz]);
    vb.assign(&v4_src[1*sz], &v4_src[2*sz]);
    vc.assign(&v4_src[2*sz], &v4_src[3*sz]);
    vd.assign(&v4_src[3*sz], &v4_src[4*sz]);
}
//----------------------------------------------------------------------
typedef std::vector<double> VD;
void vectorBreakupAoS(VD& v4_src, VD& va, VD& vb, VD& vc, VD& vd)
{
    int sz = v4_src.size() >> 2;
    va.resize(sz);
    vb.resize(sz);
    vc.resize(sz);
    vd.resize(sz);

    for (int i=0; i < sz; i+=4) {
        va[i] = v4_src[i+0];
        vb[i] = v4_src[i+1];
        vc[i] = v4_src[i+2];
        vd[i] = v4_src[i+3];
    }
}
//----------------------------------------------------------------------
void vectorCombine(VD& va, VD& vb, VD& vc, VD& vd, VD& v4_target)
{
    int sz = va.size();
    v4_target.resize(sz*4);
    std::copy(va.begin(), va.end(), v4_target.begin());
    std::copy(vb.begin(), vb.end(), v4_target.begin()+sz);
    std::copy(vc.begin(), vc.end(), v4_target.begin()+2*sz);
    std::copy(vd.begin(), vd.end(), v4_target.begin()+3*sz);
}
//----------------------------------------------------------------------
void vectorCombineAoS(VD& va, VD& vb, VD& vc, VD& vd, VD& v4_target)
{
    int sz = va.size();
    v4_target.resize(sz*4);
    for (int i=0; i < sz; i++) {
        v4_target[4*i+0] = va[i];
        v4_target[4*i+1] = vb[i];
        v4_target[4*i+2] = vc[i];
        v4_target[4*i+3] = vd[i];
    }
}
//----------------------------------------------------------------------
void setupTimers(EB::TimerList& tm) {
    tm["main_total"]         = new EB::Timer("[main] Total Time");
    tm["total"]             = new EB::Timer("[main] Remaining time");
    tm["rbffd"]             = new EB::Timer("[main] RBFFD constructor");
    tm["destructor"]         = new EB::Timer("[main] Destructors");
    tm["stencils"]             = new EB::Timer("[main] Stencil computation");
    tm["cpu_tests"]         = new EB::Timer("[main] CPU tests");
    tm["gpu_tests"]         = new EB::Timer("[main] GPU tests");
    tm["compute_weights"]     = new EB::Timer("[main] Stencil weights");
    tm["deriv_accuracy"]     = new EB::Timer("[main] Derivative Accuracy");
    tm["sort+grid"]         = new EB::Timer("[main] Sort + Grid generation");
    tm["solution_check"]     = new EB::Timer("[main] Solution check");
    //tm.printAll(stdout, 60);
}
//----------------------------------------------------------------------
void initializeArrays()
{
    // Redundant initializations since we have tons of memory
    int size = grid->getNodeList().size();

    // We will work with four functions. 

    printf("size= %d\n", size);

    switch (kernel_type) {
    case FUN4_DERIV4_WEIGHT4:
        size = 4*size;
        break;
    }

    u_cpu.resize(size); 

    xderiv_cpu.resize(size);
    yderiv_cpu.resize(size);
    zderiv_cpu.resize(size);
    lderiv_cpu.resize(size);

    // d/dx(u) = u (given my weight definitions)
    // Output: dudx, dudy, dudz, dudl
    // All derivatives are equal if weight matrix is identity. 
    for (int i=0; i < u_cpu.size(); i++) {
        double rnd = randf(-1.,1.);
        if (i < 20) printf("rnd= %f\n", rnd);
        u_cpu[i] = rnd;
    }
}
//----------------------------------------------------------------------
void initializeOneDDerivatives(int size)
{
    ux.resize(size);
}
//----------------------------------------------------------------------
void computeOnGPU4()
{
    printf("Enter computerOnGPU4\n");
    // Do not overwrite xderiv_cpu, so allocate new space on host (to compare against CPU results)
    tm["gpu_tests"]->start();

    u_gpu = RBFFD_CL::SuperBuffer<double>(u_cpu, "u_cpu"); 
    u_gpu.copyToDevice();

    for (int i=0; i < 5; i++) {
        printf("[main] u_gpu[%d]= %f\n", i, u_gpu[i]);
    }

    der->convertWeights();
    printf("after convertWeights\n");

	std::vector<double> timing;
	timing.resize(0);

    // Not in in RBBF (knows nothing about SuperBuffer). Must redesign
    switch (kernel_type) {
    case FUN1_DERIV4_WEIGHT4:
        deriv4_gpu = RBFFD_CL::SuperBuffer<double>(4*xderiv_cpu.size(), "deriv4_gpu"); 
		for (int i=0; i < 3; i++) {
        	der->computeDerivs(u_gpu, deriv4_gpu, true); 
		}
		for (int i=0; i < 5; i++) {
        	der->computeDerivs(u_gpu, deriv4_gpu, true); 
			timing.push_back(der->getGpuExecutionTime());
		}
		printTimes(timing, nx*ny*nz, stencil_size, 4);
		//double gflop = gigaFlops(nx*ny*nz, stencil_size, mean_time);
        deriv4_gpu.copyToHost();
        break;
    //case FUN1_DERIV1_WEIGHT4:
    case FUN4_DERIV4_WEIGHT4:
    case FUN4_DERIV4_WEIGHT4_INV:
        vectorCombineAoS(ux,uy,uz,ul,uderiv_cpu);
        vectorCombineAoS(vx,vy,vz,vl,vderiv_cpu);
        vectorCombineAoS(wx,wy,wz,wl,wderiv_cpu);
        vectorCombineAoS(px,py,pz,pl,pderiv_cpu);

        uderiv_gpu = RBFFD_CL::SuperBuffer<double>(uderiv_cpu.size(), "uderiv_gpu"); 
        vderiv_gpu = RBFFD_CL::SuperBuffer<double>(vderiv_cpu.size(), "wderiv_gpu"); 
        wderiv_gpu = RBFFD_CL::SuperBuffer<double>(wderiv_cpu.size(), "wderiv_gpu"); 
        pderiv_gpu = RBFFD_CL::SuperBuffer<double>(pderiv_cpu.size(), "pderiv_gpu"); 

        //printf("size of deriv arrays: %d, %d, %d, %d, %d\n", u_gpu.hostSize(), uderiv_gpu.hostSize(), vderiv_gpu.hostSize(), wderiv_gpu.hostSize(), pderiv_gpu.hostSize());

		for (int i=0; i < 3; i++) {
        	der->computeDerivs(u_gpu, uderiv_gpu, vderiv_gpu, wderiv_gpu, pderiv_gpu, true); 
		}
		for (int i=0; i < 5; i++) {
        	der->computeDerivs(u_gpu, uderiv_gpu, vderiv_gpu, wderiv_gpu, pderiv_gpu, true); 
			timing.push_back(der->getGpuExecutionTime());
		}
		printTimes(timing, nx*ny*nz, stencil_size, 16);

        uderiv_gpu.copyToHost();
        vderiv_gpu.copyToHost();
        wderiv_gpu.copyToHost();
        pderiv_gpu.copyToHost();

        for (int i=0; i < 5; i++) {
            printf("(%d), cpu, u=%f, uderiv(ux,uy,uz,ul)= %f, %f, %f, %f\n", i, u_cpu[i], uderiv_cpu[4*i],uderiv_cpu[4*i+1],uderiv_cpu[4*i+2],uderiv_cpu[4*i+3]);
            printf("(%d), gpu, u=%f, uderiv(ux,uy,uz,ul)= %f, %f, %f, %f\n", i, u_gpu[i], uderiv_gpu[4*i],uderiv_gpu[4*i+1],uderiv_gpu[4*i+2],uderiv_gpu[4*i+3]);
        }

        //u_gpu.copyToHost();

        // // ERROR ON GPU???
        // WHY ARE DERIVATIVES ZERO?)

        #if 0
        for (int i=0; i < 5; i++) {
            printf("(%d), gpu, uderiv= %f, %f, %f, %f\n", i, uderiv_gpu[4*i],uderiv_gpu[4*i+1],uderiv_gpu[4*i+2],uderiv_gpu[4*i+3]);
            printf("(%d), gpu, u= %f, %f, %f, %f\n", i, u[4*i],u[4*i+1],u[4*i+2],u[4*i+3]);
        }
        break;
        #endif
    }

    //u_gpu.copyToHost();

    tm["gpu_tests"]->end();
}
//----------------------------------------------------------------------
void computeOnGPU()
{
    //printf("Enter computerOnGPU\n");
    // Do not overwrite xderiv_cpu, so allocate new space on host (to compare against CPU results)
    tm["gpu_tests"]->start();
    xderiv_gpu = RBFFD_CL::SuperBuffer<double>(xderiv_cpu.size(), "xderiv_cpu"); 

    u_gpu = RBFFD_CL::SuperBuffer<double>(u_cpu, "u_cpu"); 
    u_gpu.copyToDevice();
    //for (int i=0; i < 20; i++) {
        //printf("[main] u_gpu[%d]= %f\n", i, u_gpu[i]);
    //}

	std::vector<double> timing;
	timing.resize(0);

    der->convertWeights();
    // Not in in RBBF (knows nothing about SuperBuffer). Must redesign
	for (int i=0; i < 3; i++) {
    	der->computeDerivs(u_gpu, xderiv_gpu, true); 
	}
	for (int i=0; i < 5; i++) {
    	der->computeDerivs(u_gpu, xderiv_gpu, true); 
		timing.push_back(der->getGpuExecutionTime());
	}
	printTimes(timing, nx*ny*nz, stencil_size, 1);

    //for (int i=0; i < 10; i++) {
        //printf("GPU bef) xder(i) = %f\n", i, xderiv_gpu[i]);
    //}

    xderiv_gpu.copyToHost();
    //u_gpu.copyToHost();
    tm["gpu_tests"]->end();
}
//----------------------------------------------------------------------
void computeOnCPU4()
{
    tm["cpu_tests"]->start();
    der_cpu = new RBFFD(RBFFD::X | RBFFD::Y | RBFFD::Z | RBFFD::LAPL, grid, dim); 
    der_cpu->computeAllWeightsForAllStencilsEmpty();
    // Weights must already be computed

    switch (kernel_type) {
    #if 0
    case FUN1_DERIV1_WEIGHT4:
        der_cpu->computeDeriv(RBFFD::X,    u_cpu, xderiv_cpu, true);
        der_cpu->computeDeriv(RBFFD::Y,    u_cpu, yderiv_cpu, true);
        der_cpu->computeDeriv(RBFFD::Z,    u_cpu, zderiv_cpu, true);
        der_cpu->computeDeriv(RBFFD::LAPL, u_cpu, lderiv_cpu, true);
        break;
    #endif
    case FUN4_DERIV4_WEIGHT4:
    case FUN4_DERIV4_WEIGHT4_INV:
        int sz = u_cpu.size() >> 2;
        printf("before breakup\n");
        printf("sz= %d\n", sz);
        // solution is SoA on GPU (u1,u2,...),(v1,v2,...),...
        vectorBreakup(u_cpu, u, v, w, p);
        printf("after breakup\n");

        //initializeOneDDerivatives(u.size());

        printf("before computer Deriv\n");
        printf("u size: %d\n", u.size());
        printf("ux size: %d\n", ux.size());
        der_cpu->computeDeriv(RBFFD::X, u, ux, true); 
        printf("after computer Deriv\n");
        der_cpu->computeDeriv(RBFFD::X, v, vx, true);
        der_cpu->computeDeriv(RBFFD::X, w, wx, true);
        der_cpu->computeDeriv(RBFFD::X, p, px, true);
        for (int i=0; i < 5; i++) {
            printf("u,ux,vx,wx,px= %f, %f, %f, %f, %f\n", u[i],ux[i],vx[i],wx[i],px[i]);
        }

        der_cpu->computeDeriv(RBFFD::Y, u, uy, true);
        der_cpu->computeDeriv(RBFFD::Y, v, vy, true);
        der_cpu->computeDeriv(RBFFD::Y, w, wy, true);
        der_cpu->computeDeriv(RBFFD::Y, p, py, true);
        for (int i=0; i < 5; i++) {
            printf("u,uy,vy,wy,py= %f, %f, %f, %f, %f\n", u[i],uy[i],vy[i],wy[i],py[i]);
        }

        der_cpu->computeDeriv(RBFFD::Z, u, uz, true);
        der_cpu->computeDeriv(RBFFD::Z, v, vz, true);
        der_cpu->computeDeriv(RBFFD::Z, w, wz, true);
        der_cpu->computeDeriv(RBFFD::Z, p, pz, true);
        for (int i=0; i < 5; i++) {
            printf("u,uz,vz,wz,pz= %f, %f, %f, %f, %f\n", u[i],uz[i],vz[i],wz[i],pz[i]);
        }

        der_cpu->computeDeriv(RBFFD::LAPL, u, ul, true);
        der_cpu->computeDeriv(RBFFD::LAPL, v, vl, true);
        der_cpu->computeDeriv(RBFFD::LAPL, w, wl, true);
        der_cpu->computeDeriv(RBFFD::LAPL, p, pl, true);
        for (int i=0; i < 5; i++) {
            printf("u,ul,vl,wl,pl= %f, %f, %f, %f, %f\n", u[i],ul[i],vl[i],wl[i],pl[i]);
        }

        #if 0
        vectorCombine(ux,vx,wx,px,xderiv_cpu);
        vectorCombine(uy,vy,wy,py,yderiv_cpu);
        vectorCombine(uz,vz,wz,pz,zderiv_cpu);
        vectorCombine(ul,vl,wl,pl,lderiv_cpu);
        #endif

        #if 0
        vectorCombine(ux,uy,uz,ul,uderiv_cpu);
        vectorCombine(vx,vy,vz,vl,vderiv_cpu);
        vectorCombine(wx,wy,wz,wl,wderiv_cpu);
        vectorCombine(px,py,pz,pl,pderiv_cpu);
        #endif
        break;
    }

    tm["cpu_tests"]->end();

    //for (int i=0; i < 20; i++) {
        //printf("(%d), CPU4: u,dudx= %f, %f, %f, %f\n", i, u_cpu[i], xderiv_cpu[i], yderiv_cpu[i], zderiv_cpu[i], lderiv_cpu[i]);
    //}
}
//----------------------------------------------------------------------
void computeOnCPU()
{
//    printf("\n***** ComputeOnCPU *****\n");
    tm["cpu_tests"]->start();

    der_cpu = new RBFFD(RBFFD::X, grid, dim); 
    der_cpu->computeAllWeightsForAllStencilsEmpty();
    // Weights must already be computed

    // Verify that the CPU works
    // NOTE: we pass booleans at the end of the param list to indicate that
    // the function "u" is new (true) or same as previous calls (false). This
    // helps avoid overhead of passing "u" to the GPU.
    #if 1
    // u_cpu stores a single function. 
    //printf("size of u_cpu: %d\n", u_cpu.size());
    //printf("size of xderiv_cpu: %d\n", xderiv_cpu.size());

    der_cpu->computeDeriv(RBFFD::X, u_cpu, xderiv_cpu, true);

    tm["cpu_tests"]->end();
    #endif
    //printf("***** exit ComputeOnCPU *****\n\n");
}
//----------------------------------------------------------------------
void checkDerivativeAccuracy()
{
    tm["deriv_accuracy"]->start();
    double xnorm = linfnorm(*xderiv_gpu.host, xderiv_cpu);
    printf("*********************************************\n");
    printf("**       INF derivative error norm: %f    ***\n", xnorm);
    printf("*********************************************\n");
    double eps = 1.e-5;
    if (xnorm > eps) {
        printf("CPU and GPU derivative do not match to within %f\n", eps);
    }

    //printf("***\n***Derivatives with errors larger than 1.e-5 ***\n");

    #if 0
    for (int i=0; i < 50; i++) {
        printf("(CPU/GPU der) %f, %f\n", i, xderiv_cpu[i], xderiv_gpu[i]); 
    }
    #endif

    #if 0
    for (int i=0; i < xderiv_gpu.hostSize(); i++) {
        if (i > 20) {
            printf("too many to print ...\n"); 
            break;
        }
        if (abs(xderiv_gpu[i] - xderiv_cpu[i]) > 1.e-5) {
            printf("(GPU aft) xder[%d]=%f\n", i, xderiv_gpu[i]); 
            printf("(CPU aft) xder[%d]=%f\n", i, xderiv_cpu[i]);
        }
    }
    #endif
    tm["deriv_accuracy"]->end();
}
//----------------------------------------------------------------------
// Array of Structures to Structure of Arrays
void AoS2SoA(std::vector<double>& xderiv_cpu, std::vector<double>& yderiv_cpu, 
             std::vector<double>& zderiv_cpu, std::vector<double>& lderiv_cpu, 
             std::vector<double>& deriv4_cpu)
{
    deriv4_cpu.resize(4*xderiv_cpu.size());

    for (int j=0, i=0; i < xderiv_cpu.size(); i++) {
        deriv4_cpu[j++] = xderiv_cpu[i];
        deriv4_cpu[j++] = yderiv_cpu[i];
        deriv4_cpu[j++] = zderiv_cpu[i];
        deriv4_cpu[j++] = lderiv_cpu[i];
    }
}
//----------------------------------------------------------------------
void checkDerivativeAccuracy4()
{
    tm["deriv_accuracy"]->start();
    // deriv_gpu: (ux,uy,uz,ul)_1, (ux,uy,uz,ul)_2
    // Array of Structures to Structure of Arrays
    vector<double> deriv4_cpu;
    double xnorm=1.e6;
    double ynorm=1.e6;
    double znorm=1.e6;
    double lnorm=1.e6;

    switch (kernel_type) {
    case FUN1_DERIV4_WEIGHT4:
        AoS2SoA(xderiv_cpu, yderiv_cpu, zderiv_cpu, lderiv_cpu, deriv4_cpu);
        xnorm = linfnorm(*deriv4_gpu.host, deriv4_cpu);
        break;
    //case FUN1_DERIV1_WEIGHT4:
    case FUN4_DERIV4_WEIGHT4:
    case FUN4_DERIV4_WEIGHT4_INV:
        //printf("step 1\n");
        //printf("size of uderiv_cpu: %d\n", uderiv_cpu.size());
        //printf("size of uderiv_gpu: %d\n", uderiv_gpu.hostSize());
        xnorm = linfnorm(*uderiv_gpu.host, uderiv_cpu);
        //printf("step 2\n");
        ynorm = linfnorm(*vderiv_gpu.host, vderiv_cpu);
        //printf("step 3\n");
        znorm = linfnorm(*wderiv_gpu.host, wderiv_cpu);
        lnorm = linfnorm(*pderiv_gpu.host, pderiv_cpu);
        //printf("size of pderiv_cpu: %d\n", pderiv_cpu.size());
        //printf("size of pderiv_gpu: %d\n", pderiv_gpu.hostSize());
        printf("x,y,z,l norms: %f, %f, %f, %f\n", xnorm, ynorm, znorm, lnorm);
        xnorm = (xnorm > ynorm) ? xnorm : ynorm;
        xnorm = (xnorm > znorm) ? xnorm : znorm;
        xnorm = (xnorm > lnorm) ? xnorm : lnorm;
        break;
    //case FUN4_DERIV4_WEIGHT4:
        //xnorm = linfnorm(*xderiv_gpu.host, xderiv_cpu);
        //ynorm = linfnorm(*yderiv_gpu.host, yderiv_cpu);
        //znorm = linfnorm(*zderiv_gpu.host, lderiv_cpu);
        //lnorm = linfnorm(*lderiv_gpu.host, lderiv_cpu);
    }

    printf("*********************************************\n");
    printf("INF derivative error norm: %f\n", xnorm);
    printf("*********************************************\n");
    double eps = 1.e-5;
    if (xnorm > eps) {
        printf("CPU and GPU derivative do not match to within %f\n", eps);
    }

    switch (kernel_type) {
    case FUN4_DERIV4_WEIGHT4:
    case FUN4_DERIV4_WEIGHT4_INV:
        #if 0
        int sz = uderiv_cpu.size() / 4;
        //printf("CPU deriv size: %d\n", uderiv_cpu.size());
        //printf("GPU deriv size: %d\n", uderiv_gpu.hostSize());
        for (int i=0; i < 5; i++) {
            printf("----- i = %d -------\n", i);
            printf("SOL 0\n");
            printf("x cpu/gpu der: %f, %f\n", uderiv_cpu[i], uderiv_gpu[i]);
            printf("y cpu/gpu der: %f, %f\n", vderiv_cpu[i], vderiv_gpu[i]);
            printf("z cpu/gpu der: %f, %f\n", wderiv_cpu[i], wderiv_gpu[i]);
            printf("l cpu/gpu der: %f, %f\n", pderiv_cpu[i], pderiv_gpu[i]);
            printf("SOL 1\n");
            printf("x cpu/gpu der: %f, %f\n", uderiv_cpu[i+sz], uderiv_gpu[i+sz]);
            printf("y cpu/gpu der: %f, %f\n", vderiv_cpu[i+sz], vderiv_gpu[i+sz]);
            printf("z cpu/gpu der: %f, %f\n", wderiv_cpu[i+sz], wderiv_gpu[i+sz]);
            printf("l cpu/gpu der: %f, %f\n", pderiv_cpu[i+sz], pderiv_gpu[i+sz]);
            printf("SOL 2\n");
            printf("x cpu/gpu der: %f, %f\n", uderiv_cpu[i+2*sz], uderiv_gpu[i+2*sz]);
            printf("y cpu/gpu der: %f, %f\n", vderiv_cpu[i+2*sz], vderiv_gpu[i+2*sz]);
            printf("z cpu/gpu der: %f, %f\n", wderiv_cpu[i+2*sz], wderiv_gpu[i+2*sz]);
            printf("l cpu/gpu der: %f, %f\n", pderiv_cpu[i+2*sz], pderiv_gpu[i+2*sz]);
            printf("SOL 3\n");
            printf("x cpu/gpu der: %f, %f\n", uderiv_cpu[i+3*sz], uderiv_gpu[i+3*sz]);
            printf("y cpu/gpu der: %f, %f\n", vderiv_cpu[i+3*sz], vderiv_gpu[i+3*sz]);
            printf("z cpu/gpu der: %f, %f\n", wderiv_cpu[i+3*sz], wderiv_gpu[i+3*sz]);
            printf("l cpu/gpu der: %f, %f\n", pderiv_cpu[i+3*sz], pderiv_gpu[i+3*sz]);
        }
        #endif
        break;
    }

    //printf("***\n***Derivatives with errors larger than 1.e-5 ***\n");


    #if 0
    for (int i=0; i < 20; i++) {
        printf("(CPU/GPU der) %f, %f\n", i, xderiv_cpu[i], deriv4_gpu[4*i]); 
        printf("(CPU/GPU der) %f, %f\n", i, yderiv_cpu[i], deriv4_gpu[4*i+1]); 
        printf("(CPU/GPU der) %f, %f\n", i, zderiv_cpu[i], deriv4_gpu[4*i+2]); 
        printf("(CPU/GPU der) %f, %f\n", i, lderiv_cpu[i], deriv4_gpu[4*i+3]); 
    }
    #endif

    #if 0
    for (int i=0; i < xderiv_cpu.size(); i++) {
        if (i > 20) {
            printf("too many to print ...\n"); 
            break;
        }
        if (abs(deriv4_gpu[4*i] - xderiv_cpu[i]) > 1.e-) {
            printf("(GPU aft) xder[%d]=%f\n", i, deriv4_gpu[4*i]); 
            printf("(CPU aft) xder[%d]=%f\n", i, xderiv_cpu[i]);
        }
    }
    #endif
    printf("----\n");
    switch (kernel_type) {
    case FUN1_DERIV4_WEIGHT4:
        for (int i=0; i < 5; i++) {
            printf("(%d) GPU: x,y,z,l= %f, %f, %f, %f\n", i, deriv4_gpu[4*i], deriv4_gpu[4*i+1], deriv4_gpu[4*i+2], deriv4_gpu[4*i+3]);
        }
        break;
    #if 0
    case FUN1_DERIV1_WEIGHT4:
        for (int i=0; i < 20; i++) {
            printf("(%d) GPU: x,y,z,l= %f, %f, %f, %f\n", i, xderiv_gpu[i], yderiv_gpu[i], zderiv_gpu[i], lderiv_gpu[i]);
        }
        break;
    #endif
    }
    tm["deriv_accuracy"]->end();
}
//----------------------------------------------------------------------
void createGrid()
{
    tm["total"]->start();

    int dim = 3;
    nx = REQUIRED<int>("NB_X");
    ny = REQUIRED<int>("NB_Y");
    nz = REQUIRED<int>("NB_Z");

    // FIX: PROGRAM TO DEAL WITH SINGLE WEIGHT 

    double minX = OPTIONAL<double> ("MIN_X", "-1.");     
    double maxX = OPTIONAL<double> ("MAX_X", " 1.");     
    double minY = OPTIONAL<double> ("MIN_Y", "-1.");     
    double maxY = OPTIONAL<double> ("MAX_Y", " 1.");     
    double minZ = OPTIONAL<double> ("MIN_Z", "-1.");     
    double maxZ = OPTIONAL<double> ("MAX_Z", " 1.");     

    stencil_size = REQUIRED<int>("STENCIL_SIZE"); 
    use_gpu      = OPTIONAL<int>("USE_GPU", "1"); 


    grid = new RegularGrid(nx, ny, nz, minX, maxX, minY, maxY, minZ, maxZ); 
    tm["total"]->end();

    tm["sort+grid"]->start();
    grid->setSortBoundaryNodes(true); 
    grid->generate();
    tm["sort+grid"]->end();

    tm["stencils"]->start();
    Grid::st_generator_t stencil_type;
	std::string node_dist = REQUIRED<std::string>("NODE_DIST");
	if (node_dist == "random") {
    	stencil_type = Grid::ST_RANDOM;
	} else if (node_dist == "compact") {
    	stencil_type = Grid::ST_COMPACT;
	}
    grid->generateStencils(stencil_size, stencil_type);   // nearest nb_points
    tm["stencils"]->end();

}
//----------------------------------------------------------------------
void setupDerivativeWeights()
{
    // Might need more options for setKernelType
    //
    tm["compute_weights"]->start();
    if (use_gpu) {
        switch (kernel_type) {
        case FUN_KERNEL:
            //printf("\n\nbefore new FUN_CL\n");
            der = new FUN_CL(RBFFD::X, grid, dim); 
            //printf("  ** enter setKernelType\n");
            der->setKernelType(FUN_CL::FUN_KERNEL); // necessary
            //printf("  ** exited setKernelType\n");
            break;
        case FUN_INV_KERNEL:
            der = new FUN_CL(RBFFD::X, grid, dim); 
            der->setKernelType(FUN_CL::FUN_INV_KERNEL);
            break;
        #if 0
        case FUN1_DERIV1_WEIGHT4:
            der = new FUN_CL(RBFFD::X | RBFFD::Y | RBFFD::Z | RBFFD::LAPL, grid, dim); 
            der->setKernelType(FUN_CL::FUN1_DERIV1_WEIGHT4);
            break;
        #endif
        case FUN1_DERIV4_WEIGHT4:
            der = new FUN_CL(RBFFD::X | RBFFD::Y | RBFFD::Z | RBFFD::LAPL, grid, dim); 
            der->setKernelType(FUN_CL::FUN1_DERIV4_WEIGHT4);
            break;
        case FUN4_DERIV4_WEIGHT4:
            der = new FUN_CL(RBFFD::X | RBFFD::Y | RBFFD::Z | RBFFD::LAPL, grid, dim); 
            der->setKernelType(FUN_CL::FUN4_DERIV4_WEIGHT4);
            break;
        case FUN4_DERIV4_WEIGHT4_INV:
            der = new FUN_CL(RBFFD::X | RBFFD::Y | RBFFD::Z | RBFFD::LAPL, grid, dim); 
            der->setKernelType(FUN_CL::FUN4_DERIV4_WEIGHT4_INV);
            break;
        }
        //printf("before computeAllWeights\n");
        der->computeAllWeightsForAllStencilsEmpty(); 
        //printf("after computeAllWeights\n");
    } else {
        //printf("Routine meant to test GPU only\n");
        exit(0);
    }

    double epsilon = settings->GetSettingAs<double>("EPSILON");
    der->setEpsilon(epsilon);
    //printf("*** epsilon= %f\n", epsilon);

    //printf("*** exit setupDerivativeWeights\n");

    // weights are all in one large array (for all derivatives)
    tm["compute_weights"]->end();

}
//----------------------------------------------------------------------
void cleanup()
{
    tm["destructor"]->start();
    delete(der);
    //printf("after delete der\n");
    delete(der_cpu);
    //printf("after delete der_cpu\n");
    delete(grid);  // **** ERROR I BELIEVE (WHY?)
    //printf("after delete grid\n");
    delete(settings);
    //printf("after delete settings\n");
    cout.flush();
    tm["destructor"]->end();
}
//----------------------------------------------------------------------
int main(int argc, char** argv)
{
    setupTimers(tm);
    //tm.printAll(stdout, 60);
    //kernel_type = FUN1_DERIV4_WEIGHT4; // ux,uy,uz,up
    //kernel_type = FUN4_DERIV4_WEIGHT4;
    //kernel_type = FUN4_DERIV4_WEIGHT4_INV;

    tm["main_total"]->start();

    //Communicator* comm_unit = new Communicator(argc, argv);
    //settings = new ProjectSettings(argc, argv, comm_unit->getRank());
    settings = new ProjectSettings("test.conf");
    //
    // Parse file created by python script
    // Uncomment if not using a script
    bool use_script = REQUIRED<bool>("USE_PYTHON_SCRIPT");
    if (use_script) {
        std::string script_data_file = REQUIRED<std::string>("SCRIPT_DATA_FILE");
        settings->ParseFile(script_data_file);
    }

    std::string k_type = REQUIRED<std::string>("FUN_KERNEL");
	printf("k_type= %s\n", k_type.c_str());
    if (k_type == "FUN_KERNEL") kernel_type = FUN_KERNEL;
    else if (k_type == "FUN_INV_KERNEL") kernel_type = FUN_INV_KERNEL;
    else if (k_type == "FUN4_DERIV4_WEIGHT4") kernel_type = FUN4_DERIV4_WEIGHT4;
    else if (k_type == "FUN4_DERIV4_WEIGHT4_INV") kernel_type = FUN4_DERIV4_WEIGHT4_INV;
    else if (k_type == "FUN1_DERIV4_WEIGHT4") kernel_type = FUN1_DERIV4_WEIGHT4;
    else { printf("kernel type (%s) not found\n"); exit(0); }

    //printf("******   creeatGrid *****************\n");
    createGrid();
    //printf("******   initialize arrays *****************\n");
    initializeArrays();
    //printf("*******  setupDerivativeWeights *************\n");
    setupDerivativeWeights();
    printf("*******  exit setupDerivativeWeights *************\n");

    switch (kernel_type) {
    case FUN_KERNEL:
    case FUN_INV_KERNEL:
        //printf("**** Compute on CPU ****\n");
        computeOnCPU(); // must be called before GPU
        //printf("**** Compute on GPU ****\n");
        computeOnGPU();
        checkDerivativeAccuracy();
        break;
    case FUN4_DERIV4_WEIGHT4:
    case FUN4_DERIV4_WEIGHT4_INV:
    case FUN1_DERIV4_WEIGHT4:
    //case FUN1_DERIV1_WEIGHT4:
        printf("**** Compute on CPU4 ****\n");
        computeOnCPU4(); // must be called before GPU
        printf("**** Exit Compute on CPU4 ****\n");
    printf("----------------------------------------------------------------------------------\n");
    printf("----------------------------------------------------------------------------------\n");
        computeOnGPU4();
        checkDerivativeAccuracy4();
        break;
    }


    tm["main_total"]->end();
    tm.printAll(stdout, 60);

    cleanup();

    exit(EXIT_SUCCESS);
}
//----------------------------------------------------------------------
//
//stencils on GPU are zero. WHY? 
