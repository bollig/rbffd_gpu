#include <stdio.h>
#include <stdlib.h>

// GLOBAL VARIABLES

//----------------------------------------------------------------------
int main (int argc, char** argv)
{
	if (argc > 1) {
		printf("ARGC = %d\n", argc); 
		return EXIT_FAILURE; 	// FAIL TEST
	} 

	return EXIT_SUCCESS; 		// PASS TEST
}
//----------------------------------------------------------------------
