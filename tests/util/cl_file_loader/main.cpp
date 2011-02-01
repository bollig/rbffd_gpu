#include <iostream>
#include "utils/opencl/cl_file_loader.h"

using namespace std;

// GLOBAL VARIABLES

//----------------------------------------------------------------------
int main (int argc, char** argv)
{
    CLFileLoader f; 

    f.printExtensions();

	if (argc > 1) {
        std::cout << "ARGC = " << argc << std::endl;
		return EXIT_FAILURE; 	// FAIL TEST
	} 

	return EXIT_SUCCESS; 		// PASS TEST
}
//----------------------------------------------------------------------
