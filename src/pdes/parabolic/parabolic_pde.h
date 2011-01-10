#ifndef __PARABOLIC_PDE_H__
#define __PARABOLIC_PDE_H__

// Since we can have scalar or vector solutions we might as well
// think ahead.
typedef double SolutionType; 

class ParabolicPDE
{
	protected: 
		std::vector<Node>* node_list; 
		std::vector<int>* boundary_indices;

		Derivative* deriv; 
		Grid* grid; 

		std::vector<SolutionType> solution[2];	// Lets keep a little history so we can check convergence rates
		int current_solution_switch; 		// 0 or 1 allows us to switch between solution buffers

		double current_time; 
		double dt; 

		//NOTE: we dont have an exact solution in here because the 
		// experiments might not have an exact solution. We might be 
		// checking convergence to see if our solution is correct.

	public: 
		ParabolicPDE(Grid* _grid, Derivative* _deriv) 
			: dt(0.), current_time(0.), current_solution_switch(0),
			grid(_grid), deriv(_deriv), 
			node_list(_grid->getNodeList()), boundary_indicies(_grid->getBoundaryIndices())
		{
			solution[0]->resize(node_list->size()); 
			solution[1]->resize(node_list->size()); 
		}

		virtual void setDt (double _dt) { dt = _dt; }  

		// Masking off negative bit with &1 works faster than fabs
		// will need to replace logic if we allow for more than one solution
		// as part of history
		int nextSolutionSwitch() { return (current_solution_switch+1)&1; }
		int prevSolutionSwitch() { return (current_solution_switch-1)&1; }


		std::vector<SolutionType> getCurrentSolution() { return solution[current_solution_switch]; }
		std::vector<SolutionType> getPreviousSolution() { return solution[this->prevSolutionSwitch()]; }

		// If initial_solution is NOT NULL then we apply the update to set an initial solution 
		// (perhaps you will need for parallel solves...)
		virtual void initialConditions(std::vector<SolutionType>* initial_solution = NULL); 

		// NOTE: we dont provide routines to check error because that depends on a known exact solution
		// but we can check norms
		// Compute L1, L2 and LInf norms for current solution and write to file. 
		virtual void checkSolutionNorms();


		// Solve and advance by one timestep. Here we could allow for different timestep 
		// logic like Euler or RK4 etc. 
		virtual void advanceOneTimestep() = 0;
};

#endif //__PARABOLIC_PDE_H__
