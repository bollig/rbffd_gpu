#include "utils/comm/communicator.h"
#include <stdlib.h>

#include "pdes/parabolic/heat_pde.h"

#include "grids/regulargrid.h"

#include "rbffd/rbffd_cl.h"
#include "rbffd/fun_cl.h"
#include "utils/random.h"
#include "utils/norms.h"

#include "exact_solutions/exact_regulargrid.h"

#include "timer_eb.h"
#include "utils/io/rbffd_io.h"

vector<double> u_cpu, xderiv_cpu, yderiv_cpu, zderiv_cpu, lderiv_cpu; 
RBFFD_CL::SuperBuffer<double> u_gpu;
RBFFD_CL::SuperBuffer<double> xderiv_gpu, yderiv_gpu, zderiv_gpu, lderiv_gpu;

Grid* grid;
int dim;
int stencil_size;
int use_gpu;
ProjectSettings* settings;
RBFFD* der_cpu;
RBFFD* der;

using namespace std;

//----------------------------------------------------------------------
void initializeArrays()
{
}
//----------------------------------------------------------------------
void computeOnGPU()
{
}
//----------------------------------------------------------------------
void computeOnCPU()
{
}
//----------------------------------------------------------------------
void createGrid()
{
    int nx = REQUIRED<int>("NB_X");
    int ny = REQUIRED<int>("NB_Y");
	int nz = REQUIRED<int>("NB_Z");
	dim = REQUIRED<int>("DIMENSION");

	// FIX: PROGRAM TO DEAL WITH SINGLE WEIGHT 

    double minX = OPTIONAL<double>("MIN_X", "-1."); 	
    double maxX = OPTIONAL<double>("MAX_X", "1."); 	
    double minY = OPTIONAL<double>("MIN_Y", "-1."); 	
    double maxY = OPTIONAL<double>("MAX_Y", "1."); 	
    double minZ = OPTIONAL<double>("MIN_Z", "-1."); 	
    double maxZ = OPTIONAL<double>("MAX_Z", "1."); 

    stencil_size = REQUIRED<int>("STENCIL_SIZE");
    use_gpu = OPTIONAL<int>("USE_GPU", "1");

	grid = new RegularGrid(nx, ny, nz, minX, maxX, minY, maxY, minZ, maxZ); 

    grid->setSortBoundaryNodes(true); 
    grid->generate();

	std::string node_dist = REQUIRED<std::string>("NODE_DIST");
	printf(">>>>>>> node_dist= %s\n", node_dist.c_str());
	Grid::st_generator_t stencil_type;
	if (node_dist == "compact") {
		stencil_type = Grid::ST_COMPACT;
	} else {
		stencil_type = Grid::ST_RANDOM;
	}
    grid->generateStencils(stencil_size, stencil_type);   // nearest nb_points

}
//----------------------------------------------------------------------
void setupDerivativeWeights()
{
    //der = new FUN_CL(RBFFD::X, grid, dim); 
    der = new RBFFD(RBFFD::X, grid, dim); 
    der->computeAllWeightsForAllStencilsEmpty(); 
}
//----------------------------------------------------------------------
void cleanup()
{
}
//----------------------------------------------------------------------
int main(int argc, char** argv)
{
    Communicator* comm_unit = new Communicator(argc, argv);
    settings = new ProjectSettings("test.conf");

	// Parse file created by python script
	// Uncomment if not using a script
	bool use_script = REQUIRED<bool>("USE_PYTHON_SCRIPT");

	if (use_script) {
    	settings->ParseFile("create.conf");
	}

	std::string node_dist = REQUIRED<std::string>("NODE_DIST");
	printf("node_dist= %s\n", node_dist.c_str());

	createGrid();
	setupDerivativeWeights();

	computeOnGPU();

	std::string file;

	//der->setAsciiWeights(0);
	//file = der->getFilename(RBFFD::X);
	//file = node_dist + "_" + file;
	//printf("file: %s\n", file.c_str());
    //der->writeToFile(RBFFD::X, file);
	
	//int writeToBinaryMMFile(std::vector<int>& rows, std::vector<int>& cols,
	//	std::vector<T>& values, int width, int height, std::string& filename);
	
//		void convertWeightToContiguous(std::vector<double>& weights_d, std::vector<int>& stencils_d, int stencil_size, 
//    		bool is_padded, bool nbnode_nbsten_dertype);

	der->setAsciiWeights(1);
	file = der->getFilename(RBFFD::X);
	file = node_dist + "_" + file;
	printf("file: %s\n", file.c_str());
    der->writeToFile(RBFFD::X, file);

	// read asci file and create binary file in float format. Can always read it in 
	// either double or float. This way saves space, and will not influence the benchmarks. 
	// It only influences the accuracy of any check on the derivative values. 
	RBFFD_IO<double> io;
	std::vector<int> rows, cols;
	std::vector<double> values;
	int width, height;
	io.loadFromAsciMMFile(rows, cols, values, width, height, file);

	printf("=======================================\n");
	printf("file= %s\n", file.c_str());
	printf("width, height= %d, %d\n", width, height);
	for (int i=0; i < 10; i++) {
		printf("r,c,v= %d, %d, %f\n", rows[i], cols[i], values[i]);
	}

	// Store in double, read in float works. 
	RBFFD_IO<float> iof;  // does not work with float
	std::vector<float> valuesf(values.size());  // works with float or double

	for (int i=0; i < values.size(); i++) {
		valuesf[i] = values[i];
	}

	//std::fill(values.begin(), values.end(), valuesf.begin());

	file = file + 'b';
	iof.writeToBinaryMMFile(rows, cols, valuesf, width, height, file);

	// Read from binary file to test the read
	RBFFD_IO<float> iof1;
	std::vector<float> valuesf1(values.size(), -1.);
	//rows.resize(0);
	//cols.resize(0);
	iof1.loadFromBinaryMMFile(rows, cols, valuesf1, width, height, file);
	for (int i=0; i < 10; i++) {
		printf("load binary: iof1, r,c,v= %d, %d, %f\n", rows[i], cols[i], valuesf1[i]);
	}

    exit(EXIT_SUCCESS);
}
//----------------------------------------------------------------------
//
//stencils on GPU are zero. WHY? 
