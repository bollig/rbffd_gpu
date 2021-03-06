// This example demonstrates the use of MPI and communication (non-overlapped)
// for VCL/UBLAS
#include <mpi.h>
#include <getopt.h>

#include <viennacl/compressed_matrix.hpp>
#include <viennacl/coordinate_matrix.hpp>
#include <viennacl/ell_matrix.hpp>
#include <viennacl/linalg/gmres.hpp>
#include <viennacl/linalg/norm_1.hpp>
#include <viennacl/linalg/norm_2.hpp>
#include <viennacl/linalg/norm_inf.hpp>
#include <viennacl/linalg/prod.hpp>
#include <viennacl/io/matrix_market.hpp>
#include <viennacl/matrix.hpp>
#include <viennacl/vector.hpp>

#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_sparse.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/numeric/ublas/banded.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/vector_proxy.hpp>
#include <boost/numeric/ublas/vector_sparse.hpp>
#include <boost/numeric/ublas/vector_of_vector.hpp>
#include <boost/numeric/ublas/vector_expression.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/operation.hpp>
#include <boost/numeric/ublas/lu.hpp>

#include "grids/grid_reader.h"
#include "rbffd/rbffd.h"
#include "timer_eb.h"


#include "utils/spherical_harmonics.h"

#include <CL/opencl.h>

#include <iostream>
#include <sstream>
#include <map>
#include <fstream>
#include <typeinfo>
using namespace std;

#define stringify( name ) # name

// Define a couple types that we will work with.
// Then we can templatize most routines and specialize as necessary

typedef std::vector< std::map< unsigned int, double> > STL_Sparse_Mat;
typedef boost::numeric::ublas::compressed_matrix<double> UBLAS_CSR_Mat;
typedef boost::numeric::ublas::coordinate_matrix<double> UBLAS_COO_Mat;
typedef viennacl::compressed_matrix<double> VCL_CSR_Mat;
typedef viennacl::coordinate_matrix<double> VCL_COO_Mat;
typedef viennacl::ell_matrix<double> VCL_ELL_Mat;

//typedef std::vector<double> UBLAS_Vec;
typedef boost::numeric::ublas::vector<double> UBLAS_Vec;
typedef viennacl::vector<double> VCL_Vec;

enum MatrixType //: int
{
	COO_CPU=0, COO_GPU, CSR_CPU, CSR_GPU, ELL_GPU, DUMMY
};

const char* assemble_t_eStrings[] =
{
	stringify( UBLAS_COO_CPU ),
	stringify( VCL_COO_GPU ),
	stringify( UBLAS_CSR_CPU ),
	stringify( VCL_CSR_GPU ),
	stringify( VCL_ELL_GPU ),
	stringify( DUMMY )
};


// TODO:
// Sort CSR, ELL, HYB by column. (use std::pair<unsigned int, unsigned int>
// (sten[j], j) and sort on sten[j]. Then use the sorted j's to index sten[]
// and lapl[]
// NOTE: I did this sorting and benchmarked. There was no difference in timing.
// The STL maps auto sort and the assembly on the GPU sorts as well.

EB::TimerList timers;


//---------------------------------

template <typename MatT, typename VecT>
void benchmark_Multiply_Host(MatT& A, VecT& F, VecT& U_exact) {
	VecT F_discrete(F.size(), 1);
	F_discrete = boost::numeric::ublas::prod(A, U_exact);

	std::cout << "Rel l1   Norm: " << boost::numeric::ublas::norm_1(F_discrete - F) / boost::numeric::ublas::norm_1(F) << std::endl;
	std::cout << "Rel l2   Norm: " << boost::numeric::ublas::norm_2(F_discrete - F) / boost::numeric::ublas::norm_2(F) << std::endl;
	std::cout << "Rel linf Norm: " << boost::numeric::ublas::norm_inf(F_discrete - F) / boost::numeric::ublas::norm_inf(F) << std::endl;
}

template <typename MatT, typename VecT>
void benchmark_Multiply_Device(MatT& A, VecT& F, VecT& U_exact) {
	VecT F_discrete(F.size());
	F_discrete = viennacl::linalg::prod(A, U_exact);

	VecT diff = F_discrete - F;

	std::cout << "Rel l1   Norm: " << viennacl::linalg::norm_1(diff) / viennacl::linalg::norm_1(F) << std::endl;
	std::cout << "Rel l2   Norm: " << viennacl::linalg::norm_2(diff) / viennacl::linalg::norm_2(F) << std::endl;
	std::cout << "Rel linf Norm: " << viennacl::linalg::norm_inf(diff) / viennacl::linalg::norm_inf(F) << std::endl;
}

void assemble_LHS(RBFFD& der, Grid& grid, STL_Sparse_Mat& A){

	unsigned int N = grid.getNodeListSize();
	unsigned int n = grid.getMaxStencilSize();

	unsigned int n_bnd = grid.getBoundaryIndicesSize();
	std::cout << "Boundary nodes: " << n_bnd << std::endl;
	//A_ptr = new MatType( N );
	//    MatType& A     = *A_ptr;

	for (unsigned int i = n_bnd; i < N; i++) {
		StencilType& sten = grid.getStencil(i);
		double* lapl = der.getStencilWeights(RBFFD::LSFC, i);

		// Off diagonals
		for (unsigned int j = 0; j < n; j++) {
			A[i][sten[j]] = -lapl[j];
		}
	}
}

void assemble_LHS( RBFFD& der, Grid& grid, UBLAS_COO_Mat& A){

	unsigned int N = grid.getNodeListSize();
	unsigned int n = grid.getMaxStencilSize();

	unsigned int n_bnd = grid.getBoundaryIndicesSize();
	std::cout << "Boundary nodes: " << n_bnd << std::endl;
	//A_ptr = new MatType( N );
	//    MatType& A     = *A_ptr;

	for (unsigned int i = n_bnd; i < N; i++) {
		StencilType& sten = grid.getStencil(i);
		double* lapl = der.getStencilWeights(RBFFD::LSFC, i);

		// Off diagonals
		for (unsigned int j = 0; j < n; j++) {
			A.append_element(i, sten[j], -lapl[j]);
		}
	}
}

void assemble_LHS( RBFFD& der, Grid& grid, UBLAS_CSR_Mat& A){

	unsigned int N = grid.getNodeListSize();
	unsigned int n = grid.getMaxStencilSize();

	unsigned int n_bnd = grid.getBoundaryIndicesSize();
	std::cout << "Boundary nodes: " << n_bnd << std::endl;
	//A_ptr = new MatType( N );
	//    MatType& A     = *A_ptr;

	for (unsigned int i = n_bnd; i < N; i++) {
		StencilType& sten = grid.getStencil(i);
		double* lapl = der.getStencilWeights(RBFFD::LSFC, i);

		// Off diagonals
		for (unsigned int j = 0; j < n; j++) {
			A(i,sten[j]) += -lapl[j];
			//A(i,sten[j]) = -lapl[j];
		}
	}
}


void assemble_RHS ( RBFFD& der, Grid& grid, UBLAS_Vec& F, UBLAS_Vec& U_exact){
	SphericalHarmonic::Sph32 UU;

	unsigned int N = grid.getNodeListSize();
	//unsigned int n = grid.getMaxStencilSize();
	std::vector<NodeType>& nodes = grid.getNodeList();

	for (unsigned int i = 0; i < N; i++) {
		NodeType& node = nodes[i];
		double Xx = node.x();
		double Yy = node.y();
		double Zz = node.z();

		U_exact[i] = UU.eval(Xx, Yy, Zz);
		// Solving -lapl(u) = f
		F[i] = -UU.lapl(Xx, Yy, Zz);
	}
}

template <class MatType, class OpMatType, MatrixType assemble_t_e, MatrixType operate_t_e>
void run_SpMV(RBFFD& der, Grid& grid) {
	unsigned int N = grid.getNodeListSize();
	unsigned int n = grid.getMaxStencilSize();

	char test_name[256];
	char assemble_timer_name[256];
	char copy_timer_name[512];
	char test_timer_name[256];

	sprintf(test_name, "%u SpMV (%s -> %s)", N, assemble_t_eStrings[assemble_t_e], assemble_t_eStrings[operate_t_e]);
	sprintf(assemble_timer_name, "%u %s Assemble", N, assemble_t_eStrings[assemble_t_e]);
	sprintf(copy_timer_name,     "%u %s Copy To %s", N, assemble_t_eStrings[assemble_t_e], assemble_t_eStrings[operate_t_e]);
	sprintf(test_timer_name, "%u %s Multiply test", N, assemble_t_eStrings[operate_t_e]);

	if (!timers.contains(assemble_timer_name)) { timers[assemble_timer_name] = new EB::Timer(assemble_timer_name); }
	if (!timers.contains(copy_timer_name)) { timers[copy_timer_name] = new EB::Timer(copy_timer_name); }
	if (!timers.contains(test_timer_name)) { timers[test_timer_name] = new EB::Timer(test_timer_name); }


	std::cout << test_name << std::endl;

	MatType* A = NULL;
	OpMatType* A_op = NULL;

	// Assemble the matrix
	// ----------------------
	timers[assemble_timer_name]->start();
	A = new MatType(N, N, n*N);
	assemble_LHS(der, grid, *A);

	UBLAS_Vec* F = new UBLAS_Vec(N, 1);
	UBLAS_Vec* U_exact = new UBLAS_Vec(N, 1);
	assemble_RHS(der, grid, *F, *U_exact);
	timers[assemble_timer_name]->stop();

	timers[copy_timer_name]->start();
	A_op = new OpMatType(N,N);
	copy(*A, *A_op);

	VCL_Vec* F_op = new VCL_Vec(N);
	VCL_Vec* U_exact_op = new VCL_Vec(N);
	viennacl::copy(F->begin(), F->end(), F_op->begin());
	viennacl::copy(U_exact->begin(), U_exact->end(), U_exact_op->begin());
	timers[copy_timer_name]->stop();

	for (int ii=0; ii<10; ii++) {
		timers[test_timer_name]->start();
		benchmark_Multiply_Device<OpMatType>(*A_op, *F_op, *U_exact_op);
		timers[test_timer_name]->stop();
	}

	// Cleanup
	delete(A);
	delete(A_op);
	delete(F);
	delete(U_exact);
	delete(F_op);
	delete(U_exact_op);
}

template <class MatType, MatrixType assemble_t_e, MatrixType operate_t_e>
void run_SpMV(RBFFD& der, Grid& grid) {

	unsigned int N = grid.getNodeListSize();
	unsigned int n = grid.getMaxStencilSize();

	char test_name[256];
	char assemble_timer_name[256];
	char test_timer_name[256];

	sprintf(test_name, "%u SpMV (%s -> %s)", N, assemble_t_eStrings[assemble_t_e], assemble_t_eStrings[operate_t_e]);
	sprintf(assemble_timer_name, "%u %s Assemble", N, assemble_t_eStrings[assemble_t_e]);
	sprintf(test_timer_name, "%u %s Multiply test", N, assemble_t_eStrings[assemble_t_e]);

	if (!timers.contains(assemble_timer_name)) { timers[assemble_timer_name] = new EB::Timer(assemble_timer_name); }
	if (!timers.contains(test_timer_name)) { timers[test_timer_name] = new EB::Timer(test_timer_name); }

	std::cout << test_name << std::endl;

	// Assemble the matrix
	// ----------------------
	timers[assemble_timer_name]->start();
	MatType* A = new MatType(N,N, n*N);
	assemble_LHS(der, grid, *A);

	UBLAS_Vec* F = new UBLAS_Vec(N, 1);
	UBLAS_Vec* U_exact = new UBLAS_Vec(N, 1);
	assemble_RHS(der, grid, *F, *U_exact);
	timers[assemble_timer_name]->stop();

#if 0
	std::ofstream f_out("F.mtx");
	for (unsigned int i = 0; i < N; i++) {
		f_out << (*F)[i] << std::endl;
	}
	f_out.close();
#endif

	for (int ii=0; ii<10; ii++) {
		timers[test_timer_name]->start();
		benchmark_Multiply_Host<MatType>(*A, *F, *U_exact);
		timers[test_timer_name]->stop();
	}

	// Cleanup
	delete(A);
}

template <class MatType, MatrixType assemble_t_e, MatrixType operate_t_e>
void run_test(RBFFD& der, Grid& grid) {
	switch (operate_t_e) {
		case COO_GPU:
			run_SpMV<MatType, VCL_COO_Mat, assemble_t_e, operate_t_e>(der, grid);
			break;
		case CSR_GPU:
			run_SpMV<MatType, VCL_CSR_Mat, assemble_t_e, operate_t_e>(der, grid);
			break;
		case ELL_GPU:
			run_SpMV<MatType, VCL_ELL_Mat, assemble_t_e, operate_t_e>(der, grid);
			break;
		case COO_CPU:
			run_SpMV<MatType, assemble_t_e, operate_t_e>(der, grid);
			break;
		case CSR_CPU:
			run_SpMV<MatType, assemble_t_e, operate_t_e>(der, grid);
			break;
		default:
			std::cout << "ERROR! Unsupported multiply type\n";
	}
}

template <MatrixType assemble_t_e, MatrixType operate_t_e>
void run_test(RBFFD& der, Grid& grid) {
	switch (assemble_t_e) {
		case COO_CPU:
			run_test<UBLAS_COO_Mat, assemble_t_e, operate_t_e>(der, grid);
			break;
		case CSR_CPU:
			run_test<UBLAS_CSR_Mat, assemble_t_e, operate_t_e>(der, grid);
			break;
		case DUMMY:
			run_SpMV<UBLAS_CSR_Mat, VCL_COO_Mat, assemble_t_e, assemble_t_e>(der, grid);
			break;
		default:
			std::cout << "ERROR! Unsupported assembly type\n";
	}
}

int main(int argc, char** argv)
{
	int do_stencils = 0; 
	int do_partition = 0;
	int do_multiply = 0; 

	const struct option longopts[] =
	{
		{"version",   no_argument,        0, 'v'},
		{"help",      no_argument,        0, 'h'},
		// Need to get the number of stencil neighbors (n)
		{"stencils",  required_argument,  0, 's'},
		{"multiply",  no_argument,  0, 'm'},
		// Need to get number of processors to expect (P)
		{"metis",     required_argument,  0, 'g'},
		{0,0,0,0},
	};

	int index;
	int iarg=0;

	//turn off getopt error message
	opterr=1; 

	while(iarg != -1)
	{
		iarg = getopt_long(argc, argv, "s:mg:vh", longopts, &index);

		switch (iarg)
		{
			case 'h':
				std::cout << "You hit help" << std::endl;
				break;

			case 'v':
				std::cout << "You hit version" << std::endl;
				break;

			case 'm':
				std::cout << "You hit multiply" << std::endl;
				break;

			case 'g':
				std::cout << "You hit graph" << std::endl;
				break;

			case 's':
				std::cout << "You hit stencils" << std::endl;
				do_stencils = 1; 
				break;
			default: 
				exit(0); 
				break;
		}
		break;
	}

	if (do_stencils) {

		bool writeIntermediate = true;
		bool primed = false;

		std::vector<std::string> grids;
		grids.clear();

		//grids.push_back("~/GRIDS/md/md005.00036");
		grids.push_back("~/sphere_grids/md063.04096");
#if 0
		grids.push_back("~/sphere_grids/md079.06400"); 
		grids.push_back("~/sphere_grids/md089.08100");
		grids.push_back("~/sphere_grids/md100.10201");
		grids.push_back("~/sphere_grids/md127.16384");
		grids.push_back("~/sphere_grids/md141.20164");
		grids.push_back("~/sphere_grids/md165.27556");  
		grids.push_back("~/sphere_grids/scvtmesh001.100000");
		grids.push_back("~/sphere_grids/scvtmesh002.500000");
		grids.push_back("~/sphere_grids/scvtmesh003.1000000");
#endif
		for (size_t i = 0; i < grids.size(); i++) {
			std::string& grid_name = grids[i];

			std::string weight_timer_name = grid_name + " Calc Weights";

			timers[weight_timer_name] = new EB::Timer(weight_timer_name.c_str());

			// Get contours from rbfzone.blogspot.com to choose eps_c1 and eps_c2 based on stencil_size (n)
			// NOTE: for benchmarking the size matters but eps_c* do not. We can
			// get junk derivatives and benchmark the same (the FLOP count matters,
			// not the accuracy). 
			// Also, the sparsity pattern matters (KDTree vs LSH) 
#if 0
			unsigned int stencil_size = 40;
			double eps_c1 = 0.027;
			double eps_c2 = 0.274;
#else
			unsigned int stencil_size = 50;
			double eps_c1 = 0.027;
			double eps_c2 = 0.274;
#endif

			GridReader* grid = new GridReader(grid_name, 4);
			grid->setMaxStencilSize(stencil_size);
			// We do not read until generate is called:

			Grid::GridLoadErrType err = grid->loadFromFile();
			if (err == Grid::NO_GRID_FILES)
			{
				grid->generate();
				if (writeIntermediate) {
					grid->writeToFile();
				}
			}
			std::cout << "Generate Stencils\n";
			Grid::GridLoadErrType st_err = grid->loadStencilsFromFile();
			if (st_err == Grid::NO_STENCIL_FILES) {
				//            grid->generateStencils(Grid::ST_BRUTE_FORCE);
#if 1
				grid->generateStencils(Grid::ST_KDTREE);
#else
				grid->setNSHashDims(50, 50,50);
				grid->generateStencils(Grid::ST_HASH);
#endif
				if (writeIntermediate) {
					grid->writeToFile();
				}
			}
			delete(grid);
		}

		timers.printAll();
		timers.writeToFile();


	} else if (do_partition) { 

	} else if (do_multiply) { 

		MPI::Init(argc, argv);
		MPI::COMM_WORLD.Set_errhandler(MPI::ERRORS_THROW_EXCEPTIONS);

		if ( MPI::COMM_WORLD.Get_rank() ) { 

			std::cout << "Forget it my rank is: " << MPI::COMM_WORLD.Get_rank() << std::endl;

		} else { 

			std::cout << "Woot I'm working hard. My rank is: " << MPI::COMM_WORLD.Get_rank() << std::endl;

		}
		MPI::Finalize();
	}
#if 0


	std::cout << "Generate RBFFD Weights\n";
	timers[weight_timer_name]->start();
	RBFFD der(RBFFD::LSFC | RBFFD::XSFC | RBFFD::YSFC | RBFFD::ZSFC, grid, 3, 0);
	der.setEpsilonByParameters(eps_c1, eps_c2);
	int der_err = der.loadAllWeightsFromFile();
	if (der_err) {
		der.computeAllWeightsForAllStencils();

		timers[weight_timer_name]->start();
		if (writeIntermediate) {
			der.writeAllWeightsToFile();
		}
	}

	if (!primed)  {
		cout << "Priming GPU with dummy operations (removes compile from benchmarks)\n";
		run_test<DUMMY, DUMMY>(der, *grid);
		primed = true;
	}

	cout << "Running Tests\n" << std::endl;
	{
#if 0
		run_test<COO_CPU, COO_CPU>(der, *grid);
#endif 
		run_test<COO_CPU, COO_GPU>(der, *grid);
#if 1
		run_test<CSR_CPU, CSR_CPU>(der, *grid);
		run_test<CSR_CPU, CSR_GPU>(der, *grid);
		run_test<COO_CPU, CSR_GPU>(der, *grid);
		run_test<CSR_CPU, COO_GPU>(der, *grid);
		run_test<COO_CPU, ELL_GPU>(der, *grid);
		run_test<CSR_CPU, ELL_GPU>(der, *grid);
#endif
	}
#endif 


	return EXIT_SUCCESS;
}

