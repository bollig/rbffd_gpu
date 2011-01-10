#include <stdlib.h>
#include <math.h>
#include "derivative_cl.h"

using namespace std;

//----------------------------------------------------------------------
DerivativeCL::DerivativeCL(vector<Vec3>& rbf_centers_, vector<vector<int> >& stencil_, int nb_bnd_, int dimensions)
: Derivative(rbf_centers_, stencil_, nb_bnd_, dimensions)	
{
	cout << "Inside DerivativeCL constructor" << endl;

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
}
//----------------------------------------------------------------------
