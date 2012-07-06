#include "boost/tuple/tuple.hpp" 
#include "boost/tuple/tuple_comparison.hpp" 
#include "boost/tuple/tuple_io.hpp" 
#include <map> 
#include <stdlib.h>

#include "grids/grid_reader.h"

#include "grids/domain.h"
#include "rbffd/derivative_tests.h"
#include "rbffd/rbffd.h"
#include "rbffd/rbffd_cl.h"

#include "timer_eb.h"
#include "utils/comm/communicator.h"
#include "utils/io/pde_writer.h"

#if USE_VTK
#include "utils/io/vtu_pde_writer.h"
#endif 

#include "stokes_steady_vcl.h"

using namespace std;
using namespace EB;
namespace b=boost::tuples; 

//----------------------------------------------------------------------
//NOTE: EVERYTHING BELOW IN MAIN WAS COPIED FROM heat_regulargrid_2d/main.cpp
//----------------------------------------------------------------------

int main(int argc, char** argv) {

    std::vector<std::string> grids; 
#if 1
    //grids.push_back("~/GRIDS/md/md031.01024"); 
   // grids.push_back("~/GRIDS/md/md050.02601"); 
   // grids.push_back("~/GRIDS/md/md063.04096"); 
    grids.push_back("~/GRIDS/md/md079.06400"); 
    //grids.push_back("~/GRIDS/geoff/scvtmesh_1m_nodes.ascii"); 
#else 
    //grids.push_back("~/GRIDS/md/md031.01024"); 
    grids.push_back("~/GRIDS/md/md050.02601"); 
    grids.push_back("~/GRIDS/md/md063.04096"); 
    grids.push_back("~/GRIDS/md/md079.06400"); 
    grids.push_back("~/GRIDS/md/md100.10201"); 
    grids.push_back("~/GRIDS/md/md127.16384"); 
    grids.push_back("~/GRIDS/md/md141.20164");
    grids.push_back("~/GRIDS/md/md159.25600"); 
    grids.push_back("~/GRIDS/md/md165.27556"); 
    grids.push_back("~/GRIDS/geoff/scvtmesh_100k_nodes.ascii"); 
    grids.push_back("~/GRIDS/geoff/scvtmesh_500k_nodes.ascii"); 
    grids.push_back("~/GRIDS/geoff/scvtmesh_1m_nodes.ascii"); 
#endif 

    std::vector<b::tuple<int, double,double> > stencil_params; 
    stencil_params.push_back(b::tuple<int, double,double>(40, 0.038, 0.222)); // Much better 
#if 0
    stencil_params.push_back(b::tuple<int, double,double>(40, 0.055, 0.239)); // EVEN better (for N6400) 
    stencil_params.push_back(b::tuple<int, double,double>(40, 0.038, 0.222)); // Much better 
    stencil_params.push_back(b::tuple<int, double,double>(40, 0.077, 0.220)); // BAD
    stencil_params.push_back(b::tuple<int, double,double>(40, 0.020, 0.295)); // Too IC to work. 
    stencil_params.push_back(b::tuple<int, double,double>(40, 0.027, 0.274)); // Nothing to write home about 
#endif 
#if 0
    stencil_params.push_back(b::tuple<int, double,double>(31, 0.035, 0.1));   // Works well
    stencil_params.push_back(b::tuple<int, double,double>(20, 0.01, 0.01));  
    stencil_params.push_back(b::tuple<int, double,double>(60, 0.037, 0.262));  
    stencil_params.push_back(b::tuple<int, double,double>(80, 0.045, 0.311));  
    stencil_params.push_back(b::tuple<int, double,double>(100, 0.050, 0.308));  
#endif 

    Communicator* comm_unit = new Communicator(argc, argv);

    for (size_t st = 0; st < stencil_params.size(); st++) {
        for (size_t gr = 0; gr < grids.size(); gr++) {

            TimerList tm;

            tm["total"] = new Timer("[Main] Total runtime for this proc");
            tm["grid"] = new Timer("[Main] Grid generation");
            tm["stencils"] = new Timer("[Main] Stencil generation");
            tm["settings"] = new Timer("[Main] Load settings"); 
            tm["decompose"] = new Timer("[Main] Decompose domain"); 
            tm["consolidate"] = new Timer("[Main] Consolidate subdomain solutions"); 
            tm["updates"] = new Timer("[Main] Broadcast solution updates"); 
            tm["send"] = new Timer("[Main] Send subdomains to other processors (master only)"); 
            tm["receive"] = new Timer("[Main] Receive subdomain from master (clients only)"); 
            tm["weights"] = new Timer("[Main] Compute all stencils weights"); 
            tm["oneWeight"] = new Timer("[Main] Compute single stencil weights"); 

            // grid should only be valid instance for MASTER
            Grid* grid=NULL; 
            Domain* subdomain; 

            int dim = 1; 

            tm["total"]->start(); 


            cout << " Got Rank: " << comm_unit->getRank() << endl;
            cout << " Got Size: " << comm_unit->getSize() << endl;

            tm["settings"]->start(); 

            ProjectSettings* settings = new ProjectSettings(argc, argv, comm_unit->getRank());

            int writeIntermediate = settings->GetSettingAs<int>("WRITE_INTERMEDIATE", ProjectSettings::required); 

            std::string& grid_name = grids[gr]; 
            if (comm_unit->isMaster()) {

                int ns_nx = settings->GetSettingAs<int>("NS_NB_X", ProjectSettings::optional, "10"); 
                int ns_ny = settings->GetSettingAs<int>("NS_NB_Y", ProjectSettings::optional, "10");
                int ns_nz = settings->GetSettingAs<int>("NS_NB_Z", ProjectSettings::optional, "10");

                //            int stencil_size = settings->GetSettingAs<int>("STENCIL_SIZE", ProjectSettings::required); 

                tm["settings"]->stop(); 

                grid = new GridReader(grid_name, 4); 
                // Trickery. We load the quadrature weights from file
                ((GridReader*)grid)->setLoadExtra(1);
                grid->setMaxStencilSize(b::get<0>(stencil_params[st])); 

                Grid::GridLoadErrType err = grid->loadFromFile(); 
                if (err == Grid::NO_GRID_FILES) 
                {
                    printf("\n************** Generating new Grid **************\n"); 
                    grid->setSortBoundaryNodes(true); 
                    tm["grid"]->start(); 
                    grid->generate();
                    tm["grid"]->stop(); 
                    if (writeIntermediate) {
                        grid->writeToFile(); 
                    }
                } 
                if ((err == Grid::NO_GRID_FILES) || (err == Grid::NO_STENCIL_FILES)) {
                    std::cout << "Generating stencils files\n";
                    tm["stencils"]->start(); 
#if 1
                    grid->setNSHashDims(ns_nx, ns_ny, ns_nz);
                    grid->generateStencils(Grid::ST_HASH);   
#else 
                    //            grid->generateStencils(Grid::ST_BRUTE_FORCE);  
                    grid->generateStencils(Grid::ST_KDTREE);   
#endif 
                    tm["stencils"]->stop();
                    if (writeIntermediate) {
                        grid->writeToFile(); 
                    }
                }

                int x_subdivisions = comm_unit->getSize();		// reduce this to impact y dimension as well 
                int y_subdivisions = (comm_unit->getSize() - x_subdivisions) + 1; 

                // TODO: load subdomain from disk

                // Construct a new domain given a grid. 
                Domain* original_domain = new Domain(dim, grid, comm_unit->getSize()); 
                // pre allocate pointers to all of the subdivisions
                std::vector<Domain*> subdomain_list(x_subdivisions*y_subdivisions);
                // allocate and fill in details on subdivisions

                std::cout << "Generating subdomains\n";
                tm["decompose"]->start();
                //original_domain->printVerboseDependencyGraph();
                original_domain->generateDecomposition(subdomain_list, x_subdivisions, y_subdivisions); 
                tm["decompose"]->stop();

                tm["send"]->start(); 
                subdomain = subdomain_list[0]; 
                for (int i = 1; i < comm_unit->getSize(); i++) {
                    std::cout << "Sending subdomain[" << i << "]\n";
                    comm_unit->sendObject(subdomain_list[i], i); 

                    // Now that its sent, we can free the memory for domains on other processors.
                    delete(subdomain_list[i]);
                }
                tm["send"]->stop(); 

                printf("----------------------\nEND MASTER ONLY\n----------------------\n\n\n");

            } else {
                tm["settings"]->stop(); 
                cout << "MPI RANK " << comm_unit->getRank() << ": waiting to receive subdomain"
                    << endl;

                tm["receive"]->start(); 
                subdomain = new Domain(); // EMPTY object that will be filled by MPI
                comm_unit->receiveObject(subdomain, 0); // Receive from CPU (0)
                subdomain->setCommSize(comm_unit->getSize());
                tm["receive"]->stop(); 
                //subdomain->writeToFile();
            }

            comm_unit->barrier();

            RBFFD* der = new RBFFD(RBFFD::LSFC | RBFFD::XSFC | RBFFD::YSFC | RBFFD::ZSFC, subdomain, dim, comm_unit->getRank()); 

//            double eps_c1 = settings->GetSettingAs<double>("EPSILON_C1", ProjectSettings::required); 
//            double eps_c2 = settings->GetSettingAs<double>("EPSILON_C2", ProjectSettings::required); 

            der->setEpsilonByParameters(b::get<1>(stencil_params[st]),b::get<2>(stencil_params[st]));
            int der_err = der->loadAllWeightsFromFile(); 
            if (der_err) {
                tm["weights"]->start(); 
                der->computeAllWeightsForAllStencils(); 

                tm["weights"]->stop(); 
#if 0
                if (writeIntermediate) {
                    der->writeAllWeightsToFile(); 
                }
#endif 
            }
            StokesSteady_PDE_VCL* pde = new StokesSteady_PDE_VCL(subdomain, der, comm_unit); 

            pde->assemble();
            if (writeIntermediate) {
                pde->write_System(); 
            }
            pde->solve();
//            if (writeIntermediate) {
                pde->write_Solution(); 
//            }

            delete(pde);

            delete(subdomain);
            delete(settings);

            tm["total"]->stop();
            tm.printAllNonStatic(); 
            tm.clear();
        }
    } 
    delete(comm_unit); 
    printf("\n\nREACHED THE END OF MAIN\n\n");

    return 0;
}
    //----------------------------------------------------------------------
