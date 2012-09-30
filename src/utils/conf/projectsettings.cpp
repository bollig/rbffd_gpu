#include "projectsettings.h"
#include <string>
#include <map>
#include <stdlib.h>
#include <fstream>
#include <iostream>
//----------------------------------------------------------------------
// for mkdir
#include <sys/types.h>
#include <sys/stat.h>
//----------------------------------------------------------------------
#include <stdlib.h>
#include <unistd.h>
#include <limits.h>
#include <getopt.h>		// for getopt

// Definition of the static tm in project settings
EB::TimerList ProjectSettings::tm;

void closeLogFile(void) {
    fprintf(stderr, "Closing STDOUT file\n");
    fclose(stdout);
}

void debugExit(void) {

    ProjectSettings::tm["total"]->stop();
    //ProjectSettings::tm.printAll();
    ProjectSettings::tm.writeToFile("atExit_timers.log");

    std::cout.flush();
    fflush(stdout);

    fprintf(stderr, "EXIT CALLED\n");

    fflush(stderr);

}

void ProjectSettings::default_config() {
    char ldirbuf[FILENAME_MAX];
    if (getcwd(ldirbuf, sizeof(ldirbuf)) != NULL) {
        printf("Current Working Directory: %s\n", ldirbuf);
        launch_dir = ldirbuf;
    } else {
        perror("Couldnt get CWD");
    }
    cli_filename = launch_dir;
    cli_filename.append("/test.conf");
}

void ProjectSettings::setupTimers() {
    tm["total"] = new EB::Timer("[AT EXIT] Total runtime until EXIT was called");
    tm["total"]->start();
}

ProjectSettings::ProjectSettings(int argc, char** argv) :
    cwd(".")
{
    setupTimers();
    this->default_config();

    this->parseCommandLineArgs(argc, argv, 0);
    this->ParseFile(cli_filename);
}

ProjectSettings::ProjectSettings(int argc, char** argv, int mpi_rank):
    cwd(".")
{
    setupTimers();
    this->default_config();

    this->parseCommandLineArgs(argc, argv, mpi_rank);
    this->ParseFile(cli_filename);
}


// Read the file and add settings to the settings map
ProjectSettings::ProjectSettings(const std::string filename)
{
    setupTimers();
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

    std::cout << "[ProjectSettings]   Reading config file: " << filename << std::endl;
    fin.open(filename.c_str());
    if (fin.is_open()) {
        while (!fin.eof()) {
            std::getline( fin, line );

            // Replace line with the substring preceeding comments (e.g., "this is beforer # this is after")
            line = line.substr(0, line.find(comment));

            size_t sepIndx = line.find(separator);

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

    char myhostname[FILENAME_MAX];
    int err = gethostname(myhostname, FILENAME_MAX-1);
    printf("[Rank %d is on host: %s (%d)]\n",  my_rank, myhostname, err);

    // Borrowed from Getopt-Long Example in GNU LibC manual
    int verbose_flag = 0; /* Flag set by '--verbose' and '--brief'. */
    int c;
    int hostname_flag = 0; // Non-zero if hostname is set;
    int output_log_flag = 0;

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
            { "brief", no_argument, &verbose_flag, 0},
            /* These options don't set a flag.
               We distinguish them by their indices. */
            { "hostname", required_argument, &hostname_flag, 'h'},
            { "output-file", required_argument, 0, 'o' },
            { "file", required_argument, 0, 'o'},
            { "dir", required_argument, 0, 'd'},
            { "output-dir", required_argument, 0, 'd'},
            { "config-file", required_argument, 0, 'c'},
            { "help", no_argument, 0, '?'},
            { 0, 0, 0, 0}
        };

        /* getopt_long stores the option index here. */
        int option_index = 0;

        // the : here indicates a required argument
        c = getopt_long(argc, argv, "?o:c:d:", long_options, &option_index);

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
                // we need to change stdout to the specified file,
                // but it should be with respect to our output directory
                // specified via to -d flag.
                output_log_flag = 1;
                char logname[256];
                sprintf(logname, "%s.%d", optarg, my_rank);
                log_file = logname;
                break;
            case 'c':
                // Handle some relative and absolute paths (in case our
                // shell (read: submit script) cant help us on this one)
                if (optarg[0] == '/') {
                    cli_filename = optarg;
                } else if (optarg[0] == '~') {
                    cli_filename = getenv("HOME");
                    cli_filename.append("/");
                    cli_filename.append(&optarg[1]);
                } else if (optarg[0] == '.') {
                    char rpat[PATH_MAX];
                    //char* p = realpath(optarg,rpat);
                    //printf("<--%s-->file\n", rpat);
                    cli_filename = rpat;
                } else {
                    cli_filename = launch_dir;
                    cli_filename.append("/");
                    cli_filename.append(optarg);
                }
                break;

            case 'h':
                printf("[DISABLED] option -h with value `%s'\n", optarg);
                break;

            case 'd':
                // 1) copy name into logdir (global) variable = argv/RANK
                // 2) mkdir logdir if not already made
                // 3) set logdir in all classes that write files (Heat, Derivative)
                char dirbuf[FILENAME_MAX];
#if 0
                if (getcwd(dirbuf, sizeof(dirbuf)) != NULL) {
                    printf("Current dir: %s\n", dirbuf);
                    cwd = dirbuf;
                } else {
                    perror("Couldnt get CWD");
                }
#endif
                mkdir(optarg, 0744);
                //printf("Made dir: %s\n", optarg);
                if (chdir(optarg) != 0) {
                    perror("Couldnt change directory!");
                    exit(EXIT_FAILURE);
                }
                //printf("New Working Directory: %s\n", optarg);
                if (getcwd(dirbuf, sizeof(dirbuf)) != NULL) {
                    printf("New Working Directory: %s\n", dirbuf);
                    cwd = dirbuf;
                } else {
                    perror("Couldnt get new CWD");
                }
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

    if (output_log_flag) {
        std::string rfile = cwd;
        rfile.append("/");
        rfile.append(log_file);
        printf("Redirecting STDOUT to file: `%s'\n", rfile.c_str());
        //FILE* temp = freopen(rfile.c_str(), "w", stdout);
        freopen(rfile.c_str(), "w", stdout);
        atexit(closeLogFile);
    }

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

