#include <stdlib.h>

#include "pdes/parabolic/new_heat.h"

#include "grids/regulargrid.h"
#include "grids/stencil_generator.h"

//#include "grids/domain_decomposition/domain.h"
#include "rbffd/derivative_cl.h"
//#include "rbffd/new_derivative_tests.h"

#include "exact_solutions/exact_regulargrid.h"

#include "utils/comm/communicator.h"

using namespace std;

//----------------------------------------------------------------------

GPU* distributeSubDomains(Grid *grid, Communicator* comm_unit, double dt) {
#if 1
    // Assume a grid of (gx, gy) GPUs.
    // MPI can create a Cartesian grid for us (TODO)
    // MPI_Cart_create()
    //
    int gx = comm_unit->getSize();
    int gy = 1; // Update this for Cart MPI
    int gz = 1;
    vector<GPU*> gpus;
    gpus.resize(gx * gy * gz);

    // The GPU class partitions the points as if they are in [-1,1]x[-1,1]x[-1,1]
    // We should generalize this.
    int xmin = grid->xmin;
    int xmax = grid->xmax;
    int ymin = grid->ymin;
    int ymax = grid->ymax;
    int zmin = grid->zmin;
    int zmax = grid->zmax;

    double deltax = (double) (xmax - xmin) / (double) gx;
    double deltay = (double) (ymax - ymin) / (double) gy;
    double deltaz = (double) (zmax - zmin) / (double) gz;

    printf("delta gpu x, y, z= %f, %f, %f\n", deltax, deltay, deltaz);

    // Initialize GPU datastructures
    for (int id = 0; id < gx * gy * gz; id++) {
        // Derived these on paper. They work, but it takes a while to verify

        // 1) Find the slice in which we lie (NOTE: "i" or "x" is varying fastest;
        //      for "k" switch gx to gz, and swap igz and igx equations)
        int igz = id / gx*gy;
        // 2) Find the row within the slice
        int igy = (id - igz * (gx*gy)) / gx;
        // 3) Find the column within the row
        int igx = (id - igz * (gx*gy)) - igy * gx;
       
        printf("igx = %d, igy = %d, igz = %d\n", igx, igy, igz);
        double xm = xmin + igx * deltax;
        double ym = ymin + igy * deltay;
        double zm = zmin + igz * deltaz;
        printf("xm= %f, ym= %f, zm=%f, dx = %f, dy = %f, dz = %f\n", xm, ym, zm, deltax, deltay, deltaz);
        gpus[id] = new GPU(xm, xm + deltax, ym, ym + deltay,  zm, zm + deltaz, dt, id, comm_unit->getSize());
    }

    // Figure out the sets Bi, Oi Qi

    printf("nb gpus: %d\n", (int) gpus.size());
    for (int i = 0; i < gpus.size(); i++) {
        printf("\n ***************** CPU %d ***************** \n", i);
        gpus[i]->fillLocalData(grid->getNodeList(), grid->getStencils(),
                grid->getBoundaryIndices(), grid->getStencilRadii()); // Forms sets (Q,O,R) and l2g/g2l maps
        gpus[i]->fillVarData(grid->getNodeList()); // Sets function values in U
    }

    for (int i = 0; i < gpus.size(); i++) {
        printf(
                "\n ***************** FILLING O_by_rank for CPU%d ***************** \n",
                i);
        for (int j = 0; j < gpus.size(); j++) {
            gpus[i]->fillDependencyList(gpus[j]->R, j); // appends to O_by_rank	any nodes required by gpu[j]
        }

    }

    printf("gpu structures (Q\\O,O,R) are initialized\n");
    printf("initialized on scalar variable to linear function\n");

    // Distribute nodes to each GPU.
    for (int i = 1; i < gpus.size(); i++) {
        printf("Distributing to GPU[%d]\n", i);
        comm_unit->sendObject(gpus[i], i);
    }

    return gpus[0];

    // Compute derivative on a single GPU. Check against analytical result
    // du/dx=1, du/dx=2, du/dx=3
#if 0
    RBF_Gaussian rbf(1.);
    const Vec3 xi(.5, 0., 0.);
    for (int i = 0; i < 10; i++) {
        const Vec3 xvec(i * .1, 0., 0.);
        printf("%d, phi=%f, phi'=%f\n", i, rbf.eval(xvec, xi), rbf.xderiv(xvec, xi));
    }
#endif
#endif
}
//----------------------------------------------------------------------


int main(int argc, char** argv) {

    // grid should only be valid instance for MASTER
    Grid* grid; 
    GPU* subdomain; 

    Communicator* comm_unit = new Communicator(argc, argv);
    
    cout << " Got Rank: " << comm_unit->getRank() << endl;
    cout << " Got Size: " << comm_unit->getSize() << endl;

        ProjectSettings* settings = new ProjectSettings(argc, argv, comm_unit);

        int dim = settings->GetSettingAs<int>("DIMENSION", ProjectSettings::required); 

    if (comm_unit->getRank() == Communicator::MASTER) {


        int nx = settings->GetSettingAs<int>("NB_X", ProjectSettings::required); 
        int ny = 1; 
        int nz = 1; 
        if (dim > 1) {
            ny = settings->GetSettingAs<int>("NB_Y", ProjectSettings::required); 
        }
        if (dim > 2) {
            nz = settings->GetSettingAs<int> ("NB_Z", ProjectSettings::required); 
        } 
        if (dim > 3) {
            cout << "ERROR! Dim > 3 Not supported!" << endl;
            exit(EXIT_FAILURE); 
        }

        double minX = settings->GetSettingAs<double>("MIN_X", ProjectSettings::optional, "-1."); 	
        double maxX = settings->GetSettingAs<double>("MAX_X", ProjectSettings::optional, "1."); 	
        double minY = settings->GetSettingAs<double>("MIN_Y", ProjectSettings::optional, "-1."); 	
        double maxY = settings->GetSettingAs<double>("MAX_Y", ProjectSettings::optional, "1."); 	
        double minZ = settings->GetSettingAs<double>("MIN_Z", ProjectSettings::optional, "-1."); 	
        double maxZ = settings->GetSettingAs<double>("MAX_Z", ProjectSettings::optional, "1."); 

        double stencil_size = settings->GetSettingAs<int>("STENCIL_SIZE", ProjectSettings::required); 
        
        double dt = settings->GetSettingAs<double>("DT", ProjectSettings::optional, "0.001"); 

        if (dim == 1) {
            grid = new RegularGrid(nx, 1, minX, maxX, 0., 0.); 
        } else if (dim == 2) {
            grid = new RegularGrid(nx, ny, minX, maxX, minY, maxY); 
        } else if (dim == 3) {
            grid = new RegularGrid(nx, ny, nz, minX, maxX, minY, maxY, minZ, maxZ); 
        } else {
            cout << "ERROR! Dim > 3 Not Supported!" << endl;
        }

        grid->setSortBoundaryNodes(true); 
        grid->generate();
        grid->generateStencils(new StencilGenerator(stencil_size));   // nearest nb_points
        grid->writeToFile(); 
#if 0
        // Compute stencils given a set of generators
        // TODO: get these extents from the cvt class (add constructor to GPU)
        Domain* subdomain = new Domain(-1.,1.,-1.,1.,-1.,1.,0.,comm_unit->getRank(),comm_unit->getSize());    

        // Clean this up. Have GPU class fill data on constructor. Pass Grid class to constructor.
        // Remove need for extents in constructor.
        // Forms sets (Q,O,R) and l2g/g2l maps
        subdomain->fillLocalData(grid->getNodeList(), grid->getStencils(), 
                grid->getBoundaryIndices(), grid->getStencilRadii());    
        subdomain->fillVarData(grid->getNodeList()); // Sets function values in U

        // Verbosely print the memberships of all nodes within the subdomain
        //subdomain->printCenterMemberships(subdomain->G, "G");
#endif
        subdomain = distributeSubDomains(grid, comm_unit, dt); 

    } else {
        cout << "MPI RANK " << comm_unit->getRank() << ": waiting to receive subdomain"
            << endl;
        subdomain = new GPU(); // EMPTY object that will be filled by MPI

        int status = comm_unit->receiveObject(subdomain, 0); // Receive from CPU (0)

    }

    comm_unit->barrier();

    subdomain->printCenters(subdomain->G_centers, "All Centers Needed by this CPU");

    printf("CHECKING STENCILS: \n");
    for (int irbf = 0; irbf < subdomain->Q_stencils.size(); irbf++) {
        printf("Stencil[%d] = ", irbf);
        if (irbf == subdomain->Q_stencils[irbf][0]) {
            printf("PASS\n");
            subdomain->printStencil(subdomain->Q_stencils[irbf], "S");
        } else {
            printf("FAIL\n");
        }
    }

#if 0
    Grid* sub_grid = subdomain->getGrid(); 
    // TODO: Clean this up.


    Derivative* der = new DerivativeCL(sub_grid->getNodeList(), sub_grid->getStencils(), sub_grid->getBoundaryIndices().size(), dim, comm_unit->getRank()); 
#else 
    Derivative* der = new DerivativeCL(subdomain->G_centers, subdomain->Q_stencils, subdomain->global_boundary_nodes.size(), dim, comm_unit->getRank());
#endif 

    double epsilon = settings->GetSettingAs<double>("EPSILON");
    der->setEpsilon(epsilon);

#if 0
    printf("start computing weights\n");
    for (int irbf=0; irbf < sub_grid->getStencils().size(); irbf++) {
        der->computeWeights(sub_grid->getNodeList(), sub_grid->getStencil(irbf), irbf);
    }
    cout << "end computing weights" << endl;
#else 
    printf("start computing weights\n");
    for (int irbf=0; irbf < subdomain->Q_stencils.size(); irbf++) {
        der->computeWeights(subdomain->G_centers, subdomain->Q_stencils[irbf], irbf);
    }
    cout << "end computing weights" << endl;
#endif 
    vector<double>& u = subdomain->U_G;
    cout << "start computing derivative (on CPU)" << endl;

    vector<double> xderiv_cpu(subdomain->Q_stencils.size());	
    vector<double> xderiv_gpu(subdomain->Q_stencils.size());	
    vector<double> yderiv_cpu(subdomain->Q_stencils.size());	
    vector<double> yderiv_gpu(subdomain->Q_stencils.size());	
    vector<double> zderiv_cpu(subdomain->Q_stencils.size());	
    vector<double> zderiv_gpu(subdomain->Q_stencils.size());	
    vector<double> lderiv_cpu(subdomain->Q_stencils.size());	
    vector<double> lderiv_gpu(subdomain->Q_stencils.size());	

    // Verify that the CPU works
    der->computeDerivCPU(Derivative::X, u, xderiv_cpu);
    der->computeDeriv(Derivative::X, u, xderiv_gpu);

    der->computeDerivCPU(Derivative::Y, u, yderiv_cpu);
    der->computeDeriv(Derivative::Y, u, yderiv_gpu);

    der->computeDerivCPU(Derivative::Z, u, zderiv_cpu);
    der->computeDeriv(Derivative::Z, u, zderiv_gpu);

    der->computeDerivCPU(Derivative::LAPL, u, lderiv_cpu);
    der->computeDeriv(Derivative::LAPL, u, lderiv_gpu);

    for (int i = 0; i < subdomain->Q_stencils.size(); i++) {
        //        std::cout << "cpu_x_deriv[" << i << "] - gpu_x_deriv[" << i << "] = " << xderiv_cpu[i] - xderiv_gpu[i] << std::endl;
        if ( (xderiv_gpu[i] - xderiv_cpu[i] > 1e-5) 
                || (yderiv_gpu[i] - yderiv_cpu[i] > 1e-5) 
                || (zderiv_gpu[i] - zderiv_cpu[i] > 1e-5) 
                || (lderiv_gpu[i] - lderiv_cpu[i] > 1e-5) )
        {
            std::cout << "WARNING! SINGLE PRECISION GPU COULD NOT CALCULATE DERIVATIVE WELL ENOUGH!\n";
            std::cout << "Test failed on " << i << std::endl;
            std::cout << "X: " << xderiv_gpu[i] - xderiv_cpu[i] << std:: endl; 
            std::cout << "Y: " << yderiv_gpu[i] - yderiv_cpu[i] << std:: endl; 
            std::cout << "Z: " << zderiv_gpu[i] - zderiv_cpu[i] << std:: endl; 
            std::cout << "LAPL: " << lderiv_gpu[i] - lderiv_cpu[i] << std:: endl; 
            exit(EXIT_FAILURE); 
        }
    }
    std::cout << "CONGRATS! ALL DERIVATIVES WERE CALCULATED THE SAME IN OPENCL AND ON THE CPU\n";



    // (WITH AN AVERAGE ERROR OF:" << avg_error << std::endl;

    // der->computeDeriv(Derivative::Y, u, yderiv);
    // der->computeDeriv(Derivative::LAPL, u, lapl_deriv);


#if 0
    if (settings->GetSettingAs<int>("RUN_DERIVATIVE_TESTS")) {
        DerivativeTests* der_test = new DerivativeTests();
        der_test->testAllFunctions(*der, *grid);
    }
#endif 



    // SOLVE HEAT EQUATION

    //EB 4
#if 1
    // Exact Solution ( freq, decay )
    //ExactSolution* exact = new ExactRegularGrid(1.0, 1.0);
    ExactSolution* exact = new ExactRegularGrid(acos(-1.) / 2., 1.);

    Heat heat(exact, subdomain, der, comm_unit->getRank());
    heat.initialConditions(&subdomain->U_G);

    // Send updates according to MPISendable object.
    comm_unit->broadcastObjectUpdates(subdomain);
    comm_unit->barrier();

    // This is HARDCODED because we dont have the ability currently to call
    // maxEig = der.computeEig() and therefore we have a different timestep than
    // the original code. I will address this next.
    //heat.setDt(0.011122);
    heat.setDt(subdomain->dt);
    subdomain->printVector(subdomain->global_boundary_nodes,
            "GLOBAL BOUNDARY NODES: ");
    // Even with Cartesian, the max norm stays at one. Strange
    int iter;
    for (iter = 0; iter < 1000; iter++) {
        cout << "*********** COMPUTE DERIVATIVES (Iteration: " << iter
            << ") *************" << endl;
        subdomain->printVector(subdomain->U_G, "INPUT_TO_HEAT_ADVANCE");

        heat.advanceOneStepWithComm(comm_unit);
        subdomain->printVector(subdomain->U_G, "AFTER HEAT");

        double nrm = heat.maxNorm();

        // TODO : Need to add a "comm_unit->sendTerminate()" to
        // break all processes when problem is encountered
        if (nrm > 5.)
            break;
        //if (iter > 0) break;
    }

    printf("after heat\n");
    //	exit(0);
#endif 
    //}

    comm_unit->consolidateObjects(subdomain);

    if (comm_unit->getRank() == 0) {
        // TODO assemble final solution
        subdomain->writeFinal(grid->getNodeList(), (char*) "FINAL_SOLUTION.txt");
        // TODO print solution to file
        cout << "FINAL ITER: " << iter << endl;
    }
printf("REACHED THE END OF MAIN\n");

//    delete(subdomain);
delete(grid);
delete(settings);

cout.flush();

exit(EXIT_SUCCESS);
}
//----------------------------------------------------------------------
