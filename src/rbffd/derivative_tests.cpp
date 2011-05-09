#include <algorithm>
#include "derivative_tests.h"
#include "utils/norms.h"
#include <vector>

#include "common_typedefs.h"

using namespace std;

// Struct to sort indices for the boundary below
struct indxltclass {
    bool operator() (size_t i, size_t j) { return (i<j); }
} srter; 

//----------------------------------------------------------------------
// Run a sequence of tests for increasingly complex functions. 
// Our goal is to approximate derivatives to a function f(x,y,z), 
// such that the derivatives are equal to  0, x, y, x^2, etc.
// IF we can accurately approximate the derivatives up to x^3 within 1e-2
// relative error it should be safe to assume we can adequately approximate
// a laplacian and other linear differential operators for our PDEs.
//----------------------------------------------------------------------
void DerivativeTests::testAllFunctions() {
#if 0
    // Test all: C=0,X,Y,X2,XY,Y2,X3,X2Y,XY2,Y3,CUSTOM
    this->testDeriv(DerivativeTests::C, der, grid, grid.getStencilRadii());
    this->testDeriv(DerivativeTests::X, der, grid, grid.getStencilRadii());
    this->testDeriv(DerivativeTests::Y, der, grid, grid.getStencilRadii());
    this->testDeriv(DerivativeTests::X2, der, grid, grid.getStencilRadii());
    this->testDeriv(DerivativeTests::XY, der, grid, grid.getStencilRadii());
    this->testDeriv(DerivativeTests::Y2, der, grid, grid.getStencilRadii());
    this->testDeriv(DerivativeTests::X3, der, grid, grid.getStencilRadii());
    this->testDeriv(DerivativeTests::X2Y, der, grid, grid.getStencilRadii());
    this->testDeriv(DerivativeTests::XY2, der, grid, grid.getStencilRadii());
    this->testDeriv(DerivativeTests::Y3, der, grid, grid.getStencilRadii());
#endif
    //    exit(EXIT_FAILURE);
}

//----------------------------------------------------------------------
//  Test all weights. If weights do not sum to within 1e-5 (single precision)
//  of the value 0 then exit_failure.
//  NOTE: if Derivative der is an instance of a GPU enabled class then 
//  this will check that our GPU precision is sufficiently close to the 
//  CPU for our computation to proceed.
//----------------------------------------------------------------------
void DerivativeTests::compareGPUandCPUDerivs(size_t nb_stencils_to_test) {
    size_t nb_centers = grid->getNodeListSize(); 
    size_t nb_stencils = grid->getStencilsSize(); 

    // If nb_stencils_to_test is 0 we stick with the original assumption we're
    // going to check all stencils
    if (nb_stencils_to_test) { 
        nb_stencils = nb_stencils_to_test;
    }

    // We want weights to sum to 0, so they are all going to be scaled by 1.
    // \sum w_i f(x,y,z) = 0     where     f(x,y,z) = 1
    vector<double> u(nb_centers, 1.);

    vector<double> xderiv_gpu(nb_stencils);	
    vector<double> yderiv_gpu(nb_stencils);	
    vector<double> zderiv_gpu(nb_stencils);	
    vector<double> lderiv_gpu(nb_stencils);	

    vector<double> xderiv_cpu(nb_stencils);	
    vector<double> yderiv_cpu(nb_stencils);	
    vector<double> zderiv_cpu(nb_stencils);	
    vector<double> lderiv_cpu(nb_stencils);	

    cout << "start applying weights to compute derivatives on CPU" << endl;
    // Verify that the CPU works
    // Force der to use the CPU version of applyWeights
    der->RBFFD::applyWeightsForDeriv(RBFFD::X, u, xderiv_cpu);
    der->RBFFD::applyWeightsForDeriv(RBFFD::Y, u, yderiv_cpu);
    der->RBFFD::applyWeightsForDeriv(RBFFD::Z, u, zderiv_cpu);
    der->RBFFD::applyWeightsForDeriv(RBFFD::LAPL, u, lderiv_cpu);

    cout << "start applying weights to compute derivatives on GPU" << endl;
    // Verify the GPU works
    // If der is a GPU class then this uses the GPU. 
    der->applyWeightsForDeriv(RBFFD::X, u, xderiv_gpu);
    der->applyWeightsForDeriv(RBFFD::Y, u, yderiv_gpu);
    der->applyWeightsForDeriv(RBFFD::Z, u, zderiv_gpu);
    der->applyWeightsForDeriv(RBFFD::LAPL, u, lderiv_gpu);

    cout << "start derivative comparison" << endl;
    for (int i = 0; i < nb_stencils; i++) {
        //        std::cout << "cpu_x_deriv[" << i << "] - gpu_x_deriv[" << i << "] = " << xderiv_cpu[i] - xderiv_gpu[i] << std::endl;
        double ex = compareDeriv(xderiv_gpu[i], xderiv_cpu[i], "X", i); 
        double ey = compareDeriv(yderiv_gpu[i], yderiv_cpu[i], "Y", i); 
        double ez = compareDeriv(zderiv_gpu[i], zderiv_cpu[i], "Z", i); 
        double el = compareDeriv(lderiv_gpu[i], lderiv_cpu[i], "Lapl", i); 

        //        std::cout << i << " of " << nb_stencils << "   (Errors: " << ex << ", " << ey << ", " << ez << ", " << el << ")" << std::endl;
    }
    std::cout << "CONGRATS! ALL DERIVATIVES WERE CALCULATED THE SAME ON THE GPU/CPU AND ON THE CPU\n";
}

double DerivativeTests::compareDeriv(double deriv_gpu, double deriv_cpu, std::string label, int indx) {

    if (isnan(deriv_gpu))
    {
        std::cout << "One of the derivs calculated by the GPU is NaN (detected by isnan)!\n"; 
        exit(EXIT_FAILURE); 
    }

    double abs_error = fabs(deriv_gpu - deriv_cpu); 

    if (abs_error > 1e-5) 
    {
        std::cout << "ERROR! GPU DERIVATIVES ARE NOT WITHIN 1e-5 OF CPU. TRY A DIFFERENT SUPPORT PARAMETER!\n";
        std::cout << "Test failed on" << std::endl;
        std::cout << label << "[" << indx << "] = " << abs_error << "    (GPU: " 
            << deriv_gpu << " CPU: " << deriv_cpu << ")"
            << std::endl; 
        std::cout << "NOTE: all derivs are supposed to be 0" << std::endl;
        //exit(EXIT_FAILURE); 
        exit(EXIT_FAILURE);
    }
    return abs_error;
}









#if 0
//----------------------------------------------------------------------
void DerivativeTests::checkDerivatives(Derivative& der, Grid& grid)
{
    vector<NodeType>& rbf_centers = grid.getNodeList();
    int nb_centers = grid.getNodeListSize();
    int nb_stencils = grid.getStencilsSize();

    // change the classes the variables are located in
    vector<double> xderiv(nb_stencils);
    vector<double> yderiv(nb_stencils);
    vector<double> lapl_deriv(nb_stencils);


    // Warning! assume DIM = 2
    this->computeAllWeights(der, rbf_centers, grid.getStencils());

    printf("deriv size: %d\n", (int) rbf_centers.size());
    printf("xderiv size: %d\n", (int) xderiv.size());

    // function to differentiate
    vector<double> u(nb_centers);
    vector<double> du_ex(nb_stencils, 2.); // exact Laplacian

    vector<StencilType>& stencil = grid.getStencils();

#if 0
    for (int i=0; i < 1600; i++) {
        StencilType& v = stencil[i];
        printf("stencil[%d]\n", i);
        for (int s=0; s < v.size(); s++) {
            printf("%d ", v[s]);
        }
        printf("\n");
    }
    exit(0);
#endif

    double s;

    // INPUT buffer has nb_center elements
    for (int i=0; i < nb_centers; i++) {
        Vec3& v = rbf_centers[i];
        //s = 3.*v.x() + 2.*v.y(); //   + 3.*v.z();

        s = v.x()*v.y() + 0.5*v.x()*v.x() + 0.5*v.y()*v.y(); //   + 3.*v.z();

        //s = v.x()+v.y() +v.x()+v.y();
        //du_ex.push_back(0.);

        //s = v.x()*v.x()*v.x();
        //du_ex.push_back(6.*v.x());

        u[i] = s;
    }


    printf("main, u[0]= %f\n", u[0]);

    for (int n=0; n < 1; n++) {
        //der.computeDeriv(Derivative::X, u, xderiv);
        //der.computeDeriv(Derivative::Y, u, yderiv);
        der.computeDeriv(Derivative::LAPL, u, lapl_deriv);
    }

#if 0
    for (int i=0; i < xderiv.size(); i++) {
        Vec3& v = rbf_centers[i];
        StencilType& st = stencil[i];
        printf("(%d), (%f,%f), xder[%d]= %f, yderiv= %f\n", st.size(), v.x(), v.y(), i, xderiv[i], yderiv[i]);
    }
#endif

    // interior points

    for (int i=0; i < lapl_deriv.size(); i++) {
        StencilType& st = stencil[i];
        NodeType& v = rbf_centers[st[0]]; 
        printf("(%d), (%f,%f), lapl_der[%d]= %f, du_ex[%d]= %f\n", (int) st.size(), v.x(), v.y(), i, lapl_deriv[i], i, du_ex[i]);
    }

    int nb_bnd = grid.getBoundaryIndicesSize(); 
    // boundary points
    for (int ib=0; ib < nb_bnd; ib++) {
        int i = grid.getBoundaryIndex(ib);
        StencilType& st = stencil[i];
        NodeType& v = rbf_centers[st[0]]; 
        //printf("bnd (%d), (%f,%f), lapl_der[%d]= %f, du_ex[%d]= %f\n", st.size(), v.x(), v.y(), i, lapl_deriv[i], i, du_ex[i]);
    }

    double inter_error=0.;
    for (int i=nb_bnd; i < nb_stencils; i++) {
        // du_ex is the exact laplacian
        inter_error += (du_ex[i]-lapl_deriv[i])*(du_ex[i]-lapl_deriv[i]);
        printf("inter error[%d] = %14.7g\n", i, du_ex[i]-lapl_deriv[i]);
    }
    inter_error /= (nb_stencils-nb_bnd);

    double bnd_error=0.;
    for (int ib=0; ib < nb_bnd; ib++) {
        int i = grid.getBoundaryIndex(ib);
        bnd_error += (du_ex[i]-lapl_deriv[i])*(du_ex[i]-lapl_deriv[i]);
        printf("bnd error[%d] = %14.7g\n", i, du_ex[i]-lapl_deriv[i]);
    }
    bnd_error /= nb_bnd;

    printf("avg lapl l2_bnd_error= %14.7e\n", sqrt(bnd_error));
    printf("avg lapl l2_interior_error= %14.7e\n", sqrt(inter_error));
}
//----------------------------------------------------------------------
void DerivativeTests::checkXDerivatives(Derivative& der, Grid& grid)
{
    vector<NodeType>& rbf_centers = grid.getNodeList();
    int nb_centers = grid.getNodeListSize();
    int nb_stencils = grid.getStencilsSize();

    // change the classes the variables are located in
    vector<double> xderiv(nb_stencils);
    vector<double> yderiv(nb_stencils);
    vector<double> lapl_deriv(nb_stencils);

    printf("deriv size: %d\n", (int) rbf_centers.size());
    printf("xderiv size: %d\n", (int) xderiv.size());

    // function to differentiate
    vector<double> u(nb_centers);
    vector<double> du_ex(nb_stencils); // exact derivative
    vector<double> dux_ex(nb_stencils); // exact derivative
    vector<double> duy_ex(nb_stencils); // exact derivative

    vector<StencilType>& stencil = grid.getStencils();

    this->computeAllWeights(der, rbf_centers, grid.getStencils());

#if 0
    for (int i=0; i < 1600; i++) {
        StencilType& v = stencil[i];
        printf("stencil[%d]\n", i);
        for (int s=0; s < v.size(); s++) {
            printf("%d ", v[s]);
        }
        printf("\n");
    }
    exit(0);
#endif

    double s;

    for (int i=0; i < nb_centers; i++) {
        Vec3& v = rbf_centers[i];
        //du_ex.push_back(2.);
        dux_ex[i] = v.y();
        duy_ex[i] = v.x();

        //s = v.y();
        du_ex[i] = 1.;
    } 
    for (int i = 0; i < nb_centers; i++) {
        Vec3& v = rbf_centers[i];
        //s = 3.*v.x() + 2.*v.y(); //   + 3.*v.z();
        //s = v.x()*v.y() + 0.5*v.x()*v.x() + 0.5*v.y()*v.y(); //   + 3.*v.z();
        s = v.x()*v.y();
        u[i] = s;
    }

    printf("main, u[0]= %f\n", u[0]);
    printf("main, u= %ld\n", (long int) &u[0]);

    for (int n=0; n < 1; n++) {
        // perhaps I'll need different (rad,eps) for each. To be determined. 
        der.computeDeriv(Derivative::X, u, xderiv);
        der.computeDeriv(Derivative::Y, u, yderiv);
        der.computeDeriv(Derivative::LAPL, u, lapl_deriv);
    }
    der.computeEig(); // needs lapl_weights, analyzes stability of Laplace operator

#if 1
    for (int i=0; i < xderiv.size(); i++) {
        Vec3& v = rbf_centers[i];
        StencilType& st = stencil[i];
        printf("(%d), (%f,%f), xder[%d]= %f, xder_ex= %f\n", (int) st.size(), v.x(), v.y(), i, xderiv[i], dux_ex[i]);
        printf("(%d), (%f,%f), yder[%d]= %f, yder_ex= %f\n", (int) st.size(), v.x(), v.y(), i, yderiv[i], duy_ex[i]);
    }
#endif

    //exit(0);

    // interior points

#if 0
    for (int i=0; i < xderiv.size(); i++) {
        Vec3& v = rbf_centers[i];
        StencilType& st = stencil[i];
        printf("(%d), (%f,%f), lapl_der[%d]= %f, du_ex[%d]= %f\n", (int) st.size(), v.x(), v.y(), i, lapl_deriv[i], i, du_ex[i]);
        printf("(%d), (%f,%f), xder[%d]= %f, xder_ex= %f\n", (int) st.size(), v.x(), v.y(), i, xderiv[i], dux_ex[i]);
        printf("(%d), (%f,%f), yder[%d]= %f, yder_ex= %f\n", (int) st.size(), v.x(), v.y(), i, yderiv[i], duy_ex[i]);
    }
#endif

    // boundary points
    vector<size_t>& boundary = grid.getBoundaryIndices();
    for (int ib=0; ib < boundary.size(); ib++) {
        int i = boundary[ib];
        StencilType& st = stencil[i];
        NodeType& v = rbf_centers[st[0]]; 
        printf("bnd(%d) sz(%d), (%f,%f), xder[%d]= %f, xder_ex= %f\n", ib, (int) st.size(), v.x(), v.y(), i, xderiv[i], dux_ex[i]);
        printf("bnd(%d) sz(%d), (%f,%f), yder[%d]= %f, yder_ex= %f\n", ib, (int) st.size(), v.x(), v.y(), i, yderiv[i], duy_ex[i]);
        //printf("bnd (%d), (%f,%f), lapl_der[%d]= %f, du_ex[%d]= %f\n", st.size(), v.x(), v.y(), i, lapl_deriv[i], i, du_ex[i]);
    }
    exit(0);

    double inter_error=0.;
    for (int i=boundary.size(); i < nb_stencils; i++) {
        inter_error += (du_ex[i]-lapl_deriv[i])*(du_ex[i]-lapl_deriv[i]);
        printf("inter error[%d] = %14.7g\n", i, du_ex[i]-lapl_deriv[i]);
    }
    inter_error /= (nb_stencils-boundary.size());

    double bnd_error=0.;
    for (int ib=0; ib < boundary.size(); ib++) {
        int i = boundary[ib];
        bnd_error += (du_ex[i]-lapl_deriv[i])*(du_ex[i]-lapl_deriv[i]);
        printf("bnd error[%d] = %14.7g\n", i, du_ex[i]-lapl_deriv[i]);
    }
    bnd_error /= boundary.size();

    printf("[checkDerivs] avg l2_bnd_error= %14.7e\n", sqrt(bnd_error));
    printf("[checkDerivs] avg l2_interior_error= %14.7e\n", sqrt(inter_error));

    exit(0);
}
//----------------------------------------------------------------------
//----------------------------------------------------------------------
void DerivativeTests::testEigen(Derivative& der, Grid& grid)
{

    int nb_stencils = grid.getStencilsSize();
    int nb_bnd = grid.getBoundaryIndicesSize();
    int tot_nb_pts = grid.getNodeListSize();

    // read input file
    // compute stencils (do this only 

    //double pert = 0.05;
    vector<double> u(tot_nb_pts);
    vector<double> lapl_deriv(nb_stencils);

    vector<double>& avg_stencil_radius = grid.getStencilRadii(); // get average stencil radius for each point

    vector<StencilType>& stencil = grid.getStencils();

    // global variable
    vector<NodeType>& rbf_centers = grid.getNodeList();

    // Derivative der(rbf_centers, stencil, grid.getNbBnd());
    der.setAvgStencilRadius(avg_stencil_radius);

    // Set things up for variable epsilon


    vector<double> epsv(tot_nb_pts);

    for (int i=0; i < tot_nb_pts; i++) {
        //epsv[i] = 1. / avg_stencil_radius[i];
        epsv[i] = 1.; // fixed epsilon
        //printf("avg rad(%d) = %f\n", i, avg_stencil_radius[i]);
    }
    double mm = minimum(avg_stencil_radius);
    printf("min avg_stencil_radius= %f\n", mm);

    der.setVariableEpsilon(epsv);

    // Laplacian weights with zero grid perturbation
    for (int irbf=0; irbf < nb_stencils; irbf++) {
        der.computeWeightsSVD(rbf_centers, stencil[irbf], irbf, "lapl");
    }

    double max_eig = der.computeEig(); // needs lapl_weights
    printf("zero perturbation: max eig: %f\n", max_eig);

    vector<Vec3> rbf_centers_orig;
    rbf_centers_orig.assign(rbf_centers.begin(), rbf_centers.end());

    double percent = 0.05; // in [0,1]
    printf("percent distortion of original grid= %f\n", percent);

    // set a random seed
    srandom(time(0));

    for (int i=0; i < 100; i++) {
        printf("---- iteration %d ------\n", i);
        //update rbf centers by random perturbations at a fixed percentage of average radius computed
        //based on the unperturbed mesh
        rbf_centers.assign(rbf_centers_orig.begin(), rbf_centers_orig.end());

        for (int j=0; j < tot_nb_pts; j++) {
            Vec3& v = rbf_centers[j];
            double vx = avg_stencil_radius[j]*percent*randf(-1.,1.);
            double vy = avg_stencil_radius[j]*percent*randf(-1.,1.);
            v.setValue(v.x()+vx, v.y()+vy);
        }

        //rbf_centers[10].print("rbf_centers[10]");
        //continue;

        //recompute Laplace weights
        for (int irbf=0; irbf < nb_stencils; irbf++) {
            der.computeWeights(rbf_centers, stencil[irbf], irbf);
        }
        double max_eig = der.computeEig(); // needs lapl_weights
        printf("zero perturbation: max eig: %f\n", max_eig);
    }
}
//----------------------------------------------------------------------
void DerivativeTests::testFunction(DerivativeTests::TESTFUN which, Grid& grid, vector<double>& u, vector<double>& dux_ex, vector<double>& duy_ex,
        vector<double>& dulapl_ex)
{
    u.resize(0);
    dux_ex.resize(0);
    duy_ex.resize(0);
    dulapl_ex.resize(0);

    vector<NodeType>& rbf_centers = grid.getNodeList();
    int nb_centers = grid.getNodeList().size(); 
    int nb_stencils = grid.getStencilsSize();

    switch(which) {
        case C:
            for (int i=0; i < nb_stencils; i++) {
                dux_ex.push_back(0.);
                duy_ex.push_back(0.);
                dulapl_ex.push_back(0.);
            }

            for (int i =0; i < nb_centers; i++) {
                Vec3& v = rbf_centers[i];
                u.push_back(1.);
            }
            break;
        case X:
            printf("nb_stencils= %d\n", nb_stencils);
            for (int i=0; i < nb_stencils; i++) {
                Vec3& v = rbf_centers[i];
                dux_ex.push_back(1.);
                duy_ex.push_back(0.);
                dulapl_ex.push_back(0.);
            }
            for (int i =0; i < nb_centers; i++) {
                Vec3& v = rbf_centers[i];
                u.push_back(v.x());
            }
            printf("u.size= %d\n", (int) u.size());
            break;
        case Y:
            for (int i=0; i < nb_stencils; i++) {
                Vec3& v = rbf_centers[i];
                dux_ex.push_back(0.);
                duy_ex.push_back(1.);
                dulapl_ex.push_back(0.);
            }
            for (int i =0; i < nb_centers; i++) {
                Vec3& v = rbf_centers[i];
                u.push_back(v.y());
            }
            break;
        case X2:
            for (int i=0; i < nb_stencils; i++) {
                Vec3& v = rbf_centers[i];
                dux_ex.push_back(2.*v.x());
                duy_ex.push_back(0.);
                dulapl_ex.push_back(2.);
            }
            for (int i =0; i < nb_centers; i++) {
                Vec3& v = rbf_centers[i];
                u.push_back(v.x()*v.x());
            }
            break;
        case XY:
            for (int i=0; i < nb_stencils; i++) {
                Vec3& v = rbf_centers[i];
                dux_ex.push_back(v.y());
                duy_ex.push_back(v.x());
                dulapl_ex.push_back(0.);
            }
            for (int i =0; i < nb_centers; i++) {
                Vec3& v = rbf_centers[i];
                u.push_back(v.x()*v.y());
            }
            break;
        case Y2:
            for (int i=0; i < nb_stencils; i++) {
                Vec3& v = rbf_centers[i];
                dux_ex.push_back(0.);
                duy_ex.push_back(2.*v.y());
                dulapl_ex.push_back(2.);
            }
            for (int i =0; i < nb_centers; i++) {
                Vec3& v = rbf_centers[i];
                u.push_back(v.y()*v.y());
            }
            break;
        case X3:
            for (int i=0; i < nb_stencils; i++) {
                Vec3& v = rbf_centers[i];
                dux_ex.push_back(3.*v.x()*v.x());
                duy_ex.push_back(0.);
                dulapl_ex.push_back(6.*v.x());
            }
            for (int i =0; i < nb_centers; i++) {
                Vec3& v = rbf_centers[i];
                u.push_back(v.x()*v.x()*v.x());
            }
            break;
        case X2Y:
            for (int i=0; i < nb_stencils; i++) {
                Vec3& v = rbf_centers[i];
                dux_ex.push_back(2.*v.x()*v.y());
                duy_ex.push_back(v.x()*v.x());
                dulapl_ex.push_back(2.*v.y());
            }
            for (int i =0; i < nb_centers; i++) {
                Vec3& v = rbf_centers[i];
                u.push_back(v.x()*v.x()*v.y());
            }
            break;
        case XY2:
            for (int i=0; i < nb_stencils; i++) {
                Vec3& v = rbf_centers[i];
                dux_ex.push_back(v.y()*v.y());
                duy_ex.push_back(2.*v.x()*v.y());
                dulapl_ex.push_back(2.*v.x());
            }
            for (int i =0; i < nb_centers; i++) {
                Vec3& v = rbf_centers[i];
                u.push_back(v.x()*v.y()*v.y());
            }
            break;
        case Y3:
            for (int i=0; i < nb_stencils; i++) {
                Vec3& v = rbf_centers[i];
                dux_ex.push_back(0.);
                duy_ex.push_back(3.*v.y()*v.y());
                dulapl_ex.push_back(6.*v.y());
            }
            for (int i =0; i < nb_centers; i++) {
                Vec3& v = rbf_centers[i];
                u.push_back(v.y()*v.y()*v.y());
            }
            break;
    }
}

//----------------------------------------------------------------------
void DerivativeTests::testDeriv(DerivativeTests::TESTFUN choice, Derivative& der, Grid& grid, std::vector<double> avgDist)
{
    printf("================\n");
    printf("testderiv: choice= %d\n", choice);

    vector<NodeType>& rbf_centers = grid.getNodeList();
    int nb_centers = grid.getNodeListSize();
    int nb_stencils = grid.getStencilsSize();

    vector<double> u(nb_centers);
    vector<double> dux_ex(nb_stencils); // exact derivative
    vector<double> duy_ex(nb_stencils); // exact derivative
    vector<double> dulapl_ex(nb_stencils); // exact derivative

    // change the classes the variables are located in
    vector<double> xderiv(nb_stencils);
    vector<double> yderiv(nb_stencils);
    vector<double> lapl_deriv(nb_stencils);

    this->computeAllWeights(der, rbf_centers, grid.getStencils());
    //EB
    testFunction(choice, grid, u, dux_ex, duy_ex, dulapl_ex);


    avgDist = grid.getStencilRadii();
    for (int n=0; n < 1; n++) {
        // perhaps I'll need different (rad,eps) for each. To be determined. 
        der.computeDeriv(Derivative::X, u, xderiv);
        der.computeDeriv(Derivative::Y, u, yderiv);
        der.computeDeriv(Derivative::LAPL, u, lapl_deriv);
    }

    vector<size_t>& boundary = grid.getBoundaryIndices();
    int nb_bnd = grid.getBoundaryIndicesSize();

    //TODO:
    // - sort boundary indices
    // - create partitions: interior, boundary 
    // - compute norms from independent sets
    // WHY?
    //  because we cannot assume that boundary nodes appear first in the node lists. The partitioning
    //  we perform for the domain decomposition pushes some boundary nodes to the bottom of the list. 
    //  This bug shows itself when we subdivide the domain into multiple subdomains. 

    int nb_int = nb_stencils - nb_bnd;

    std::vector<double> dux_ex_bnd(nb_bnd); 
    std::vector<double> xderiv_bnd(nb_bnd); 

    std::vector<double> dux_ex_int(nb_int); 
    std::vector<double> xderiv_int(nb_int); 

    std::vector<double> duy_ex_bnd(nb_bnd); 
    std::vector<double> yderiv_bnd(nb_bnd); 

    std::vector<double> duy_ex_int(nb_int); 
    std::vector<double> yderiv_int(nb_int); 

    std::vector<double> dulapl_ex_bnd(nb_bnd); 
    std::vector<double> lapl_deriv_bnd(nb_bnd); 

    std::vector<double> dulapl_ex_int(nb_int); 
    std::vector<double> lapl_deriv_int(nb_int); 

    std::vector<double> avgDist_bnd(nb_bnd); 
    std::vector<double> avgDist_int(nb_int); 


    // Sort the boundary indices for easier partitioning
    std::vector<size_t> bindices = grid.getBoundaryIndices(); 
    std::sort(bindices.begin(), bindices.end(), srter); 
    {
        int i = 0;  // Index on boundary
        int k = 0;  // index on interior
        for (int j = 0; j < nb_stencils; j++) {
            // Skim off the boundary
            if (j == bindices[i]) {
                dux_ex_bnd[i] = dux_ex[j]; 
                xderiv_bnd[i] = xderiv[j]; 
                duy_ex_bnd[i] = duy_ex[j]; 
                yderiv_bnd[i] = yderiv[j]; 
                dulapl_ex_bnd[i] = dulapl_ex[j]; 
                lapl_deriv_bnd[i] = lapl_deriv[j]; 
                avgDist_bnd[i] = avgDist[j]; 
                //                std::cout << "BOUNDARY: " << i << " / " << j << ", " << avgDist_bnd[i] << std::endl;
                i++; 
            } else {
                dux_ex_int[k] = dux_ex[j]; 
                xderiv_int[k] = xderiv[j]; 
                duy_ex_int[k] = duy_ex[j]; 
                yderiv_int[k] = yderiv[j]; 
                dulapl_ex_int[k] = dulapl_ex[j]; 
                lapl_deriv_int[k] = lapl_deriv[j]; 
                avgDist_int[k] = avgDist[j]; 
                //              std::cout << "INTERIOR: " << k << " / " << j << ", " << avgDist_int[k] << std::endl;
                k++; 
            }
        } 
    }

    enum DERIV {X=0, Y, LAPL};
    enum NORM {L1=0, L2, LINF};
    enum BNDRY {INT=0, BNDRY};
    double norm[3][3][2]; // norm[DERIV][NORM][BNDRY]
    double normder[3][3][2]; // norm[DERIV][NORM][BNDRY]

    // These norms are || Lu_exact - Lu_approx ||_? 
    // where Lu_* are differential operator L applied to solution u_* 
    // and the || . ||_? is the 1, 2, or inf-norms
    norm[X][L1][BNDRY]   = l1normWeighted(dux_ex_bnd,   xderiv_bnd, avgDist_bnd, 0, nb_bnd);
    norm[X][L2][BNDRY]   = l2normWeighted(dux_ex_bnd,   xderiv_bnd, avgDist_bnd, 0, nb_bnd);
    norm[X][LINF][BNDRY] = linfnorm(dux_ex_bnd, xderiv_bnd, 0, nb_bnd);

    norm[Y][L1][BNDRY]   = l1normWeighted(duy_ex_bnd,   yderiv_bnd, avgDist_bnd, 0, nb_bnd);
    norm[Y][L2][BNDRY]   = l2normWeighted(duy_ex_bnd,   yderiv_bnd, avgDist_bnd, 0, nb_bnd);
    norm[Y][LINF][BNDRY] = linfnorm(duy_ex_bnd, yderiv_bnd, 0, nb_bnd);

    norm[LAPL][L1][BNDRY]   = l1normWeighted(dulapl_ex_bnd,   lapl_deriv_bnd, avgDist_bnd, 0, nb_bnd);
    norm[LAPL][L2][BNDRY]   = l2normWeighted(dulapl_ex_bnd,   lapl_deriv_bnd, avgDist_bnd, 0, nb_bnd);
    norm[LAPL][LINF][BNDRY] = linfnorm(dulapl_ex_bnd, lapl_deriv_bnd, 0, nb_bnd);

    norm[X][L1][INT]   = l1normWeighted(dux_ex_int,   xderiv_int, avgDist_int, 0, nb_int);
    norm[X][L2][INT]   = l2normWeighted(dux_ex_int,   xderiv_int, avgDist_int, 0, nb_int);
    norm[X][LINF][INT] = linfnorm(dux_ex_int, xderiv_int, 0, nb_int);

    norm[Y][L1][INT]   = l1normWeighted(duy_ex_int,   yderiv_int, avgDist_int, 0, nb_int);
    norm[Y][L2][INT]   = l2normWeighted(duy_ex_int,   yderiv_int, avgDist_int, 0, nb_int);
    norm[Y][LINF][INT] = linfnorm(duy_ex_int, yderiv_int, 0, nb_int);

    norm[LAPL][L1][INT]   = l1normWeighted(dulapl_ex_int,   lapl_deriv_int, avgDist_int, 0, nb_int);
    norm[LAPL][L2][INT]   = l2normWeighted(dulapl_ex_int,   lapl_deriv_int, avgDist_int, 0, nb_int);
    norm[LAPL][LINF][INT] = linfnorm(dulapl_ex_int, lapl_deriv_int, 0, nb_int);

    // --- Normalization factors (to get denom for relative error: ||u-u_approx|| / || u || )

    normder[X][L1][BNDRY]   = l1normWeighted(dux_ex_bnd, avgDist_bnd,  0, nb_bnd);
    normder[X][L2][BNDRY]   = l2normWeighted(dux_ex_bnd, avgDist_bnd,  0, nb_bnd);
    normder[X][LINF][BNDRY] = linfnorm(dux_ex_bnd, 0, nb_bnd);

    normder[Y][L1][BNDRY]   = l1normWeighted(duy_ex_bnd, avgDist_bnd,  0, nb_bnd);
    normder[Y][L2][BNDRY]   = l2normWeighted(duy_ex_bnd, avgDist_bnd,  0, nb_bnd);
    normder[Y][LINF][BNDRY] = linfnorm(duy_ex_bnd, 0, nb_bnd);

    normder[LAPL][L1][BNDRY]   = l1normWeighted(dulapl_ex_bnd, avgDist_bnd,  0, nb_bnd);
    normder[LAPL][L2][BNDRY]   = l2normWeighted(dulapl_ex_bnd, avgDist_bnd,  0, nb_bnd);
    normder[LAPL][LINF][BNDRY] = linfnorm(dulapl_ex_bnd, 0, nb_bnd);

    normder[X][L1][INT]   = l1normWeighted(dux_ex_int, avgDist_int,  0, nb_int);
    normder[X][L2][INT]   = l2normWeighted(dux_ex_int, avgDist_int,  0, nb_int);
    normder[X][LINF][INT] = linfnorm(dux_ex_int, 0, nb_int);

    normder[Y][L1][INT]   = l1normWeighted(duy_ex_int, avgDist_int, 0, nb_int);
    normder[Y][L2][INT]   = l2normWeighted(duy_ex_int, avgDist_int, 0, nb_int);
    normder[Y][LINF][INT] = linfnorm(duy_ex_int, 0, nb_int);

    normder[LAPL][L1][INT]   = l1normWeighted(dulapl_ex_int, avgDist_int, 0, nb_int);
    normder[LAPL][L2][INT]   = l2normWeighted(dulapl_ex_int, avgDist_int, 0, nb_int);
    normder[LAPL][LINF][INT] = linfnorm(dulapl_ex_int, 0, nb_int);


    printf("----- RESULTS: %d bnd, %d centers, %d stencils\ntestDeriv( %d ) [** Approximating F(X,Y) = %s **] ---\n", nb_bnd, nb_centers, nb_stencils, choice, TESTFUNSTR[(int)choice].c_str());
    //printf("{C=0,X,Y,X2,XY,Y2,X3,X2Y,XY2,Y3,CUSTOM=10}\n");
    printf("norm[x/y/lapl][L1,L2,LINF][interior/bndry]\n");
    for (int k=0; k < 2; k++) {
        for (int i=0; i < 3; i++) {
            // If our L2 absolute norm is > 1e-9 we show relative.
            if (fabs(normder[i][1][k]) < 1.e-9) {
                printf("(abs err): norm[%d][][%d]= %10.3e, %10.3e, %10.3e\n", i, k, norm[i][0][k], norm[i][1][k], norm[i][2][k]);
            } else {
                printf("(rel err): norm[%d][][%d]= %10.3e, %10.3e, %10.3e, normder= %10.3e\n", i, k, 
                        norm[i][0][k]/normder[i][0][k], 
                        norm[i][1][k]/normder[i][1][k], 
                        norm[i][2][k]/normder[i][2][k], normder[i][1][k]); 
                //printf("   normder[%d][][%d]= %10.3e, %10.3e, %10.3e\n", i, k, 
                //normder[i][0][k], 
                //normder[i][1][k], 
                //normder[i][2][k]); 
            }
        }}

    double inter_error=0.;
    for (int i= 0; i < nb_int; i++) {
        inter_error += (dulapl_ex_int[i]-lapl_deriv_int[i])*(dulapl_ex_int[i]-lapl_deriv_int[i]);
        //    std::cout << i << "dulapl_ex_int " << dulapl_ex_int[i] << ", " << lapl_deriv_int[i] << std::endl;
        //printf("inter error[%d] = %14.7g\n", i, du_ex[i]-lapl_deriv[i]);
    }
    inter_error /= nb_int;

    double bnd_error=0.;
    for (int i=0; i < nb_bnd; i++) {
        bnd_error += (dulapl_ex_bnd[i]-lapl_deriv_bnd[i])*(dulapl_ex_bnd[i]-lapl_deriv_bnd[i]);
        //     printf("bnd error[%d] = %14.7g\n", i, dulapl_ex_bnd[i]-lapl_deriv_bnd[i]);
    }
    bnd_error /= nb_bnd;

#if 1
    double l2_bnd = sqrt(bnd_error);
    printf("avg l2_bnd_error= %14.7e\n", l2_bnd);
    double l2_int = sqrt(inter_error);
    printf("avg l2_interior_error= %14.7e\n", l2_int);
    // NaN is the only number not equal to itself
    if ((l2_int > 1.e-1) || (l2_int != l2_int)) {
        printf ("ERROR! Interior l2 error is too high to continue\n");
        exit(EXIT_FAILURE);
    }
    if ((l2_bnd > 1.e0) || (l2_int != l2_int)) {

        printf ("WARNING! Boundary l2 error is high but we'll trust it to continue\n");
        return ;

        printf ("ERROR! Boundary l2 error is too high to continue\n");
        exit(EXIT_FAILURE);
    }
#endif
}
//----------------------------------------------------------------------
void DerivativeTests::computeAllWeights(Derivative& der, std::vector<Vec3>& rbf_centers, std::vector<StencilType>& stencils) {
    //#define USE_CONTOURSVD 1
    //#undef USE_CONTOURSVD

#if USE_CONTOURSVD
    exit(EXIT_FAILURE);
    // Laplacian weights with zero grid perturbation
    for (int irbf=0; irbf < stencils.size(); irbf++) {
        der.computeWeightsSVD(rbf_centers, stencils[irbf], irbf, "x");
        der.computeWeightsSVD(rbf_centers, stencils[irbf], irbf, "y");
        der.computeWeightsSVD(rbf_centers, stencils[irbf], irbf, "lapl");
    }

#else
    std::cout << "COMPUTING " << stencils.size() << " SETS OF STENCIL WEIGHTS" << std::endl;
    for (int i = 0; i < stencils.size(); i++) {
        der.computeWeights(rbf_centers, stencils[i], i);
    }
#endif
}
#endif
