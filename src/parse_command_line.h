#ifndef __PARSE_COMMAND_LINE_H__
#define __PARSE_COMMAND_LINE_H__

//----------------------------------------------------------------------
//#include <sys/stat.h>
#include <getopt.h>		// for getopt

void closeLogFile(void) {
    fprintf(stderr, "Closing STDOUT file\n");
    fclose(stdout);
}

void debugExit(void) {
    fprintf(stderr, "EXIT CALLED\n");
}

int parseCommandLineArgs(int argc, char** argv, int my_rank) {
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
            { "help", no_argument, 0,
                '?'},
            { 0, 0, 0, 0}
        };

        /* getopt_long stores the option index here. */
        int option_index = 0;

        // the : here indicates a required argument
        c = getopt_long(argc, argv, "?o:", long_options, &option_index);

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

#endif 
