#ifndef _STENCIL_GENERATOR_H
#define	_STENCIL_GENERATOR_H

#include <vector>
#include <iostream> 

#include "common_typedefs.h"

// Basic StencilGenerator 

class StencilGenerator {
public:
    StencilGenerator(double st_max_radius = 0.);
//    StencilGenerator(int st_max_size, double st_max_radius = 0.);
    ~StencilGenerator();

    void setRadius(double st_max_radius);

	// KEY ROUTINE: generates the stencils according to st_max_size and st_max_radius requirements
	// Remember: if st_max_size is < 0 then we assume 
    virtual void computeStencils(std::vector<NodeType>& node_list, std::vector<size_t>& boundary_list, std::vector<StencilType>& stencil_map, size_t max_stencil_size, std::vector<double>& avg_stencil_radii);
    	    

protected:
    void computeStencils();

protected: 
    int st_max_size;
    double st_max_radius; 		// TODO: add support for a maximum radius 
};

#endif
