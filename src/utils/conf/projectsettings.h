#ifndef __PROJECTSETTINGS_H__
#define __PROJECTSETTINGS_H__
#include <stdlib.h>
#include <string>
#include <map>
//#include <stdlib.h>
//#include <iostream>
#include <stdio.h>
#include <sstream>
//#include <fstream>
#include <sstream>
#include "utils/comm/communicator.h"


/**
  * PARSE A CONFIGURATION FILE FORMATTED AS:
  *
  * <KEY> = <VALUE>
  *
  * THEN RETURN VALUES WHEN GIVEN A KEY
  */
class ProjectSettings
{

public:
    // If a settings is specified as required and it does not exist in teh config file it will cause
    // the code to terminate. If the setting is optional then when it doesnt exist the typecast value of
    // "0" will be returned by GetSettingAs
    enum settings_priority_t {required = 0, optional};

protected:
    // The map of KEY = VALUE settings
    std::map<std::string, std::string> settings;
    std::string cli_filename;

public:
    // Read the file and add settings to the settings map
    ProjectSettings(int argc, char** argv);
    ProjectSettings(int argc, char** argv, Communicator* comm_unit);
    ProjectSettings(const std::string filename);

    // Read the file and add/update settings in the settings map
    // WARNING: will override the settings if their key already exists in map
    void ParseFile(const std::string filename);


    template <typename RT>
    void SetSetting(std::string key, RT value) {
	    // TODO: change to stringstream for any type of input that is cast as string
		char strg[80];
		sprintf(strg, "%f", (float*) &value);
		settings[key] = std::string(strg);
		//settings[key] = ss_typecast<std::string> value;
	}



    // Return the value associate with KEY as the specified template parameter type
    // e.g.,
    //  int i = ProjectSettings.GetSettingAs<int>("key");
    //  double d = ProjectSettings.GetSettingAs<double>("key2");
    //  string s = ProjectSettings.GetSettingAs<string>("key3");
    template <typename RT>
    RT GetSettingAs(std::string key, settings_priority_t priority = ProjectSettings::required, std::string defaultval = "0") {
	std::cout << "[Project Settings] \t"; 
        if (settings.find(key) == settings.end()) {
            if (priority == this->required) {
                std::cout << "ERROR! Request for unknown REQUIRED configuration setting: " << key << std::endl;
                exit(EXIT_FAILURE);
            } else {
                RT ret = ss_typecast<RT>(defaultval);
		std::cout << key << " = " << ret;
		std::cout << "\t<-- default value." << std::endl;
                return ret;
            }
        }
	std::cout << key << " = " << settings[key] << std::endl;
        return ss_typecast<RT>(settings[key]);
    }

    // Check if KEY = VALUE pair was found in config file
    bool Exists(std::string key) { if(settings.find(key) == settings.end()) { return false; } else { return true; } }

    int parseCommandLineArgs(int argc, char** argv, int my_rank);

protected:
    // This routine is adapted from post on GameDev:
    // http://www.gamedev.net/community/forums/topic.asp?topic_id=190991
    // Should be safer to use this than atoi. Performs worse, but our
    // hotspot is not this part of the code.
    template<typename RT, typename _CharT, typename _Traits , typename _Alloc >
    RT ss_typecast( const std::basic_string< _CharT, _Traits, _Alloc >& the_string )
    {
        std::basic_istringstream< _CharT, _Traits, _Alloc > temp_ss(the_string);
        RT num;
        temp_ss >> num;
        return num;
    }

    void trim( std::string& str );
};


#endif // PROJECTSETTINGS_H
