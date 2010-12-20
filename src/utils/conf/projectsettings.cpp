#include "projectsettings.h"
#include <string>
#include <map>
#include <stdlib.h>
#include <fstream>
#include <iostream>
//----------------------------------------------------------------------
//#include <sys/stat.h>
#include <stdlib.h>
#include <getopt.h>		// for getopt




void closeLogFile(void) {
    fprintf(stderr, "Closing STDOUT file\n");
    fclose(stdout);
}

void debugExit(void) {
    fprintf(stderr, "EXIT CALLED\n");
}


ProjectSettings::ProjectSettings(int argc, char** argv) :
        cli_filename("test.conf")
{
    this->parseCommandLineArgs(argc, argv, 0);
    this->ParseFile(cli_filename);
}

ProjectSettings::ProjectSettings(int argc, char** argv, Communicator* comm_unit):
        cli_filename("test.conf")
{
    this->parseCommandLineArgs(argc, argv, comm_unit->getRank());
    this->ParseFile(cli_filename);
}


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
        exit(EXIT_FAILURE);
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

int ProjectSettings::parseCommandLineArgs(int argc, char** argv, int my_rank) {
    // Borrowed from Getopt-Long Example in GNU LibC manual
    int verbose_flag = 0; /* Flag set by '--verbose' and '--brief'. */
    int c;
    int hostname_flag = 0; // Non-zero if hostname is set;

    while (1) {
        // Struct order: { Long_name, Arg_required?, FLAG_TO_SET, Short_name}
        // NOTE: if an option sets a flag, it is set to the Short_name value
        static struct option long_options[] = {
            // NOTE: following are defined in getopt.h
            // 		no_argument			--> no arg to the option is expected
            //		required_argument 	--> arg is mandatory
            // 		optional_argument	--> arg is not necessary
            /* These options set a flag. */
            { "verbose", no_argument, &verbose_flag, 1},
            { "brief",
              no_argument, &verbose_flag, 0},
/* These options don't set a flag.
             We distinguish them by their indices. */
{ "hostname", required_argument, &hostname_flag, 'h'},
{
                "output-file", required_argument, 0, 'o'
                    },
{ "file",
  required_argument, 0, 'o'},
{ "dir",
  required_argument, 0, 'd'},
{ "config-file", required_argument, 0, 'c'},
{ "help", no_argument, 0,
  '?'},
{ 0, 0, 0, 0}
        };

        /* getopt_long stores the option index here. */
        int option_index = 0;

        // the : here indicates a required argument
        c = getopt_long(argc, argv, "?o:c:", long_options, &option_index);

        /* Detect the end of the options. */
        if (c == -1)
            break;

        switch (c) {
        case 0: // Any long option that sets a flag
            printf("option %s", long_options[option_index].name);
            if (optarg)
                printf(" with arg %s", optarg);
            printf("\n");
            break;

        case 'o':
            char logname[256];
            sprintf(logname, "%s.%d", optarg, my_rank);
            printf("Redirecting STDOUT to file: `%s'\n", logname);
            freopen(logname, "w", stdout);
            atexit(closeLogFile);
            break;
        case 'c':
            cli_filename = std::string(optarg);
            break;

        case 'h':
            printf("[DISABLED] option -h with value `%s'\n", optarg);
            break;

        case 'd':
            // 1) copy name into logdir (global) variable = argv/RANK
            // 2) mkdir logdir if not already made
            // 3) set logdir in all classes that write files (Heat, Derivative)
            break;

        case '?':
            printf("\nUsage: %s [options=arguments]\n", argv[0]);
            printf("Options:\n");
            printf(
                    "\t-o (--output-file, --file) \tSpecify the filename for process to redirect STDOUT to.\n");
            printf("\t-? (--help) \t\t\tPrint this message\n\n");
            exit(EXIT_FAILURE);
            break;

        default:
            printf("IN DEFAULT ARG OPTION (WHY?)\n");
            abort(); // abort loop when nothing is left
            break;
        }
    } // END WHILE

    /* Instead of reporting ‘--verbose’
     and ‘--brief’ as they are encountered,
     we report the final status resulting from them. */
    if (verbose_flag)
        printf("verbose flag is set\n");

    /* Print any remaining command line arguments (not options). */
    if (optind < argc) {
        printf("non-option ARGV-elements: ");
        while (optind < argc)
            printf("%s ", argv[optind++]);
        putchar('\n');
    }

    // Tell us whenever program terminates with an exit is called
    atexit(debugExit);

    return 0;
}

