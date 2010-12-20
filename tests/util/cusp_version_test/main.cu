// THIS IS THE BASIC TEST FOR FUNCTIONALITY IN THRUST AND CUSP
//
// PROVIDED BY THE CUSP v0.1 README

#include <thrust/version.h>
#include <cusp/version.h>
#include <iostream>

int main(void)
{
    int thrust_major = THRUST_MAJOR_VERSION;
    int thrust_minor = THRUST_MINOR_VERSION;

    int cusp_major = CUSP_MAJOR_VERSION;
    int cusp_minor = CUSP_MINOR_VERSION;

    std::cout << "Thrust v" << thrust_major << "." << thrust_minor << std::endl;
    std::cout << "Cusp   v" << cusp_major   << "." << cusp_minor   << std::endl;

    return 0;
}

