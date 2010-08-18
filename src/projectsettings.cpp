#include "projectsettings.h"
#include <string>
#include <map>
#include <stdlib.h>
#include <fstream>
#include <iostream>

// Read the file and add settings to the settings map
ProjectSettings::ProjectSettings(std::string filename)
{
    this->ParseFile(filename);
}

// Read the file and add/update settings in the settings map
// WARNING: will override the settings if their key already exists in map
void ProjectSettings::ParseFile(std::string filename)
{
    std::ifstream fin;
}
