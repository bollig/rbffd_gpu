#ifndef __IMPLICIT_PDE_H__
#define __IMPLICIT_PDE_H__

#include "pdes/pde.h"
#include "grids/domain.h"
#include "utils/comm/communicator.h"

#include <iostream> 
#include <fstream> 
#include <iomanip> 

class ImplicitPDE : public PDE
{
    public: 
        ImplicitPDE(Domain* grid, RBFFD* der, Communicator* comm, unsigned int solution_dim) 
            // The 1 here indicates the solution will have one component
            : PDE(grid, der, comm, solution_dim)
        {   
        }

        virtual void solve(std::vector<SolutionType>& y, std::vector<SolutionType>* f_out, unsigned int n_stencils, unsigned int n_nodes) {}; 
        virtual std::string className() {
            return "ImplicitPDE"; 
        }

        virtual void assemble() =0; 

        virtual void solve() =0; 

        template <typename VecT>
            void write_to_file(VecT vec, std::string filename)
            {
                std::ofstream fout;
                fout.open(filename.c_str());
                for (size_t i = 0; i < vec.size(); i++) {
                    fout << std::setprecision(10) << vec[i] << std::endl;
                }
                fout.close();
                std::cout << "Wrote " << filename << std::endl;
            }
};

#endif 
