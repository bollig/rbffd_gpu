#include <stdlib.h>
#include <map>

#include "grids/grid_reader.h"

#include <boost/program_options.hpp>

#include "timer_eb.h"

using namespace std;
using namespace EB;

namespace po = boost::program_options;

//----------------------------------------------------------------------
//NOTE: EVERYTHING BELOW IN MAIN WAS COPIED FROM heat_regulargrid_2d/main.cpp
//----------------------------------------------------------------------

int main(int argc, char** argv) {
    TimerList tm;

    tm["total"] = new Timer("[Main] Total runtime for this proc");
    tm["grid"] = new Timer("[Main] Grid generation");
    tm["gridReader"] = new Timer("[Main] Grid Reader Load File From Disk");
    tm["loadGrid"] = new Timer("[Main] Load Grid (and Stencils) from Disk");
    tm["writeGrid"] = new Timer("[Main] Write Grid (and Stencils) to Disk");
    tm["stencils"] = new Timer("[Main] Stencil generation");
    tm["settings"] = new Timer("[Main] Load settings");

    Grid* grid = NULL;

    tm["total"]->start();

    //-----------------
    tm["settings"]->start();

    // Declare the supported options.
    po::options_description desc("Allowed options");
    desc.add_options()
	    ("help", "produce help message")
	    ("weight_method,w", po::value<int>(), "Set weight method (0:LSH, 1:KDTree, 2:BruteForce)")
	    ;

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);    

    if (vm.count("help")) {
	    cout << desc << "\n";
	    return 1;
    }

    if (vm.count("weight_method")) {
	    cout << "Weight method is set to: "
		    << vm["weight_method"].as<int>() << ".\n";
    } else {
	    cout << "Weight method was not set.\n";
    }

#if 0
    ProjectSettings* settings = new ProjectSettings(argc, argv, comm_unit->getRank());

    int dim = settings->GetSettingAs<int>("DIMENSION", ProjectSettings::required);
    int debug = settings->GetSettingAs<int>("DEBUG", ProjectSettings::optional, "0");

    int weight_method = settings->GetSettingAs<int>("WEIGHT_METHOD", ProjectSettings::optional, "0");
    int stencil_size = settings->GetSettingAs<int>("STENCIL_SIZE", ProjectSettings::required);

    int ns_nx = settings->GetSettingAs<int>("NS_NB_X", ProjectSettings::optional, "10");
    int ns_ny = settings->GetSettingAs<int>("NS_NB_Y", ProjectSettings::optional, "10");
    int ns_nz = settings->GetSettingAs<int>("NS_NB_Z", ProjectSettings::optional, "10");

    string grid_filename = settings->GetSettingAs<string>("GRID_FILENAME", ProjectSettings::required);
    int grid_size = settings->GetSettingAs<int>("GRID_SIZE", ProjectSettings::required);
    int grid_num_cols = settings->GetSettingAs<int>("GRID_FILE_NUM_COLS", ProjectSettings::optional, "3");

    tm["settings"]->stop();

    tm["gridReader"]->start();
    grid = new GridReader(grid_filename, grid_num_cols, grid_size);
    grid->setMaxStencilSize(stencil_size);
    tm["gridReader"]->stop();

    tm["loadGrid"]->start();
    Grid::GridLoadErrType err = grid->loadFromFile();
    tm["loadGrid"]->stop();
    if (err == Grid::NO_GRID_FILES)
    {
	    printf("************** Generating new Grid **************\n");
	    grid->setSortBoundaryNodes(true);
	    tm["grid"]->start();
	    grid->generate();
	    tm["grid"]->stop();
	    tm["writeGrid"]->start();
	    grid->writeToFile();
	    tm["writeGrid"]->stop();
    }
    if ((err == Grid::NO_GRID_FILES) || (err == Grid::NO_STENCIL_FILES)) {
	    std::cout << "Generating stencils files\n";
	    tm["stencils"]->start();
	    switch (weight_method) {
		    case 2: 
			    grid->generateStencils(Grid::ST_KDTREE);
			    break; 
		    case 1: 
			    grid->generateStencils(Grid::ST_BRUTE_FORCE);
			    break; 
		    case 0: 
		    default: 
			    grid->setNSHashDims(ns_nx, ns_ny, ns_nz);
			    grid->generateStencils(Grid::ST_HASH);
			    break; 
	    }
	    tm["stencils"]->stop();
	    tm["writeStencils"]->start();
	    grid->writeToFile();
	    tm["writeStencils"]->stop();
    }

    delete(grid);
    std::cout << "Deleted grid\n";

    tm["total"]->stop();
    tm.printAll();


    std::cout << "----------------  END OF MAIN ------------------\n";
    tm.writeAllToFile("time_log.stencils");
    tm.clear();

    printf("\n\nREACHED THE END OF MAIN\n\n");
#endif 
    return 0;
}
//----------------------------------------------------------------------
