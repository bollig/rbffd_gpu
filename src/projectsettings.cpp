#include "projectsettings.h"
#include <string>
#include <map>
#include <stdlib.h>
#include <fstream>
#include <iostream>

// Read the file and add settings to the settings map
ProjectSettings::ProjectSettings(const std::string filename)
{
    this->ParseFile(filename);
}

// Read the file and add/update settings in the settings map
// WARNING: will override the settings if their key already exists in map
void ProjectSettings::ParseFile(const std::string filename)
{
    std::ifstream fin;
    std::string comment = "#";
    std::string separator = "=";
    std::string section = "[";     // Just in case we want section headers (e.g. "[Section 1]")

    std::string line = comment;


    fin.open(filename.c_str());
    if (fin.is_open()) {
        while (!fin.eof()) {
            std::getline( fin, line );

            // Replace line with the substring preceeding comments (e.g., "this is beforer # this is after")
            line = line.substr(0, line.find(comment));

            int sepIndx = line.find(separator);

            if (sepIndx < line.size()) { // Implies separator exists
                std::string key = line.substr(0, sepIndx);
                trim(key);
                std::string value = line.substr(sepIndx + 1, line.size());
                trim(value);

                //std::cout << "Found Setting: (" << key << ") : (" << value << ")" << std::endl;
                settings[key] = value;
            }
        }
    } else {
        std::cout << "\n[ERROR] Config file: \"" << filename << "\" was not found. Be sure"
                << " to specify the full/relative path to access the file from"
                << " the directory you are currently executing in.\n" << std::endl;
    }
}

void ProjectSettings::trim( std::string& str )
{
        //static const char whitespace[] = " \n\t\v\r\f";
        std::string whitespace = " \n\t\v\r\f";
        // Trim front:
        str.erase( 0, str.find_first_not_of(whitespace) );
        // Trim back:
        str.erase( str.find_last_not_of(whitespace) + 1 );
}

