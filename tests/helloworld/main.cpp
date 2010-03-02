#include <iostream>

using namespace std; 

int main (int argc, char** argv) {
	cout << "Hello Cruel World!" << endl;

	if (argc > 1) {
		return EXIT_FAILURE;
	} 

	return EXIT_SUCCESS; 
}
