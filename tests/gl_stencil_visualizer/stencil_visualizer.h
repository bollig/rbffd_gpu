#ifndef __STENCIL_VISUALIZER_H__
#define __STENCIL_VISUALIZER_H__
#include <vector> 
#include "Vec3.h"

class StencilVisualizer
{ 
	public: 
		// Each StencilVisualizer is given a set of coordinates 
		// for nodes in the stencil and the weights associated with
		// each node. 
		StencilVisualizer(std::vector<Vec3> nodes, std::vector<double> weights);
		~StencilVisualizer(); 

		// Routine that needs to be called when the GL callback 
		// display() is called. This routine should ONLY draw
		// the stencil sheet for a single stencil and nothing more. 
		// Setup for the world is managed by an outside routine. 
		void Draw(); 

		// These routines will allow us to select/deselect a stencil
		// When selected we can draw/color the stencil differently. 
		void Select(); 
		void Deselect(); 

	protected: 
		// calculate the set of grid points that will be used
		// to evaluate the stencil interpolant polynomial
		virtual void GetInterpolationPoints(); 
		
		// draw the generated grid points and the interpolant
		// passing through them. 
                virtual void DrawInterpolation();


	private:	
		// The Cartesian overlay is specific to this class
		// We could have a GetDiskInterpolationPoints in an
		// extending class. Both would be called by the 
		// virtual GetInterpolationPoints() routine. 
		void GetCartesianInterpolationPoints(); 
};

#endif 	// __STENCIL_VISUALIZER_H__
