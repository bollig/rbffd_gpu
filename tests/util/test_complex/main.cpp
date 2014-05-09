#include <math.h>
#include <stdlib.h>
#include <iostream>
#include <complex>

typedef std::complex<double> CMPLX;
// GLOBAL VARIABLES

//----------------------------------------------------------------------
int main (int argc, char** argv)
{
	CMPLX ceps = CMPLX(1.1, 0.1); 
	CMPLX ceps2 = ceps*ceps; 
	CMPLX ceps2alt = CMPLX(ceps.real()*ceps.real(), 0.); 

	CMPLX scale = ceps*2.;
	CMPLX scale2 = 2.*ceps;

	std::cout << "CEPS: " << ceps << std::endl;
	std::cout << "CEPS2: " << ceps2 << std::endl;
	std::cout << "CEPS2ALT: " << ceps2alt << std::endl;
	std::cout << "scale: " << scale << std::endl;
	std::cout << "scale ceps by 2: " << scale2 << std::endl; 
	
	if (argc > 1) {
		return EXIT_FAILURE; 	// FAIL TEST
	} 

	return EXIT_SUCCESS; 		// PASS TEST
}
//----------------------------------------------------------------------
