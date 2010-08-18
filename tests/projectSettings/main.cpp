#include <math.h>
#include <stdlib.h>
#include <string>
#include <iostream>
#include "projectsettings.h"

using namespace std;
//----------------------------------------------------------------------
int main (int argc, char** argv)
{
    ProjectSettings config("config.example");

    if (config.GetSettingAs<int>("DIM") != 2)
        return EXIT_FAILURE;
    cout << "DIM passed" << endl;
    if (config.GetSettingAs<double>("EPSILON") != 1.23456789012)
        return EXIT_FAILURE;
    cout << "EPSILON passed" << endl;
    if (config.GetSettingAs<float>("RADIUS") != 1.23456f) {
        cout << "RADIUS failed: (" << config.GetSettingAs<float>("RADIUS") - 1.23456f << ")" << endl;
        return EXIT_FAILURE;
    }
    cout << "RADIUS passed" << endl;
    if (config.GetSettingAs<string>("CVT_FILENAME").compare("cvt_circle.txt"))
        return EXIT_FAILURE;
    cout << "CVT_FILENAME passed" << endl;

    cout << "SUCCESS!" << endl;

    return EXIT_SUCCESS; 		// PASS TEST
}
//----------------------------------------------------------------------
