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
#include "common_typedefs.h"

// Basic StencilGenerator 

class StencilGenerator {
public:
    StencilGenerator(int st_max_size, double st_max_radius = 0.);
    ~StencilGenerator();

    void setRadius(double st_max_radius);
    void setSize(double st_max_size);

	// KEY ROUTINE: generates the stencils according to st_max_size and st_max_radius requirements
	// Remember: if st_max_size is < 0 then we assume 
    virtual void computeStencils(std::vector<NodeType>& node_list, std::vector<size_t>& boundary_list, std::vector<StencilType>& stencil_map, std::vector<double>& avg_stencil_radii);
    	    

protected:
    void computeStencils();

protected: 
    int st_max_size;
    double st_max_radius; 		// TODO: add support for a maximum radius 
};

#endif
