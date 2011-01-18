/*
 * File:   stencil.h
 * Author: bollig
 *
 * Created on May 11, 2010, 12:48 PM
 */

#ifndef _STENCIL_H
#define	_STENCIL_H

#include <vector>
#include <iostream> 

#include "utils/comm/communicator.h"
#include "grids/grid_interface.h"

class Stencil {
public:
    Stencil(Grid* grid, int st_max_size, double st_max_radius);
    ~Stencil();

    void setRadius(double st_max_radius);
    void setSize(double st_max_size);

    void generate();

    std::vector<std::vector<int> >& getStencils();
    std::vector<int>& getStencil(int indx) { return stencil_map[indx]; }
    std::vector<double>& getAvgDist();
    double getAvgDist(int indx) { return avg_stencil_radii[indx]; }

    friend std::ostream& operator<< (std::ostream& os, const Stencil& p) {
	for (int i = 0 ; i < p.stencil_map.size(); i++) {
		for (int j = 0; j < p.stencil_map[i].size(); j++) {
			os << " [" << p.stencil_map[i][j] << "] "; 
		}
		os << std::endl;
	}
	return os; 
    }


protected:
    void computeStencils();

protected: 
    // A vector of stencils which  map of indices into the grid node list
    // with one set per stencil
    std::vector<std::vector<int> > stencil_map;
    std::vector<double> avg_stencil_radii;
    
    int st_max_size;
    double st_max_radius;
    Grid* grid;


};

#endif
