#ifndef _GRID_READER_GRID_H_
#define _GRID_READER_GRID_H_

#include "grid_interface.h"

class GridReader : public Grid {
    protected: 
        bool file_loaded;
        std::string filename; 
        unsigned int n_nodes; 

    public:

        GridReader(std::string filename_to_load, unsigned int n_nodes_to_read);
        virtual ~GridReader();

        // Overrides Grid::generate()
        virtual void generate();

        virtual int readNodeList(int expected_nb_extra_dbls_per_line); 

        // Overrides Grid::	
        virtual std::string getFileDetailString(); 

        virtual std::string className() {return "loadedgrid";}
};


#endif //_GRID_READER_GRID_H_
