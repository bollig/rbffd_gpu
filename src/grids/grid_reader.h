#ifndef _GRID_READER_GRID_H_
#define _GRID_READER_GRID_H_

#include "grid_interface.h"
#include "limits.h"

class GridReader : public Grid {
    protected: 
        int numExtraCols;
        unsigned int n_nodes; 
        std::string filename; 
        bool file_loaded;

        int loadExtra; 

        std::vector< std::vector<double> > extra_cols; 

    public:

        GridReader(std::string filename_to_load, int numCols, unsigned int n_nodes_to_read = UINT_MAX);
        virtual ~GridReader();

        // Overrides Grid::generate()
        virtual void generate();

        virtual int readNodeList(int expected_nb_extra_dbls_per_line); 

        void setLoadExtra(int numExtra) {
            loadExtra = numExtra;
        }

        std::vector<double>& getExtraCol(unsigned int i=0) { 
            return extra_cols[i]; 
        }

        // Overrides Grid::	
        virtual std::string getFileDetailString(); 

        virtual std::string className() {return "loadedgrid";}
};


#endif //_GRID_READER_GRID_H_
