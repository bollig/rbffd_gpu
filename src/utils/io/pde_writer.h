#ifndef __PDE_WRITER_H__
#define __PDE_WRITER_H__

#include "grids/domain.h"
//TODO: change to a base_pde class so we can reuse the 
//      writer for elliptic and parabolic
#include "pdes/parabolic/heat.h"
#include "utils/comm/communicator.h"

class PDEWriter
{

    protected:
        int local_write_freq; 
        int global_write_freq; 
        Domain* subdomain; 
        Heat* heat; 
        Communicator* comm_unit;

    public: 
        PDEWriter(Domain* subdomain_, Heat* heat_, Communicator* comm_unit_, int local_write_freq_, int global_write_freq_)
            : subdomain(subdomain_), heat(heat_), comm_unit(comm_unit_), 
            local_write_freq(local_write_freq_), 
            global_write_freq(global_write_freq_) { ; } 

        virtual ~PDEWriter() {
            this->writeFinal(); 
        }

        void setLocalWriteFrequency(int freq) { local_write_freq = freq; } 
        void setGlobalWriteFrequency(int freq) { global_write_freq = freq; } 

        /** 
         * Call this every iteration with the current iter count. 
         * On local_write_freq or global_write_freq (or both), it 
         * will dump info to disk. 
         */
        void update(int iter) { 
            if (iter == 0) { 
                this->writeInitial(); 
            }

            if ((iter % local_write_freq) == 0) { 
                this->writeLocal(iter); 
            }

            if ((iter % global_write_freq) == 0) { 
                this->writeGlobal(iter); 
            } 
        }

        /** 
         * Tasks to perform only on the initial iteration.
         * For example, maybe only write the grid nodes ONCE
         * at the first iteration. Or maybe you want to override
         * this to only write METADATA once at initialization. 
         */
        virtual void writeInitial() { 
            subdomain->writeToFile(); 
        }

        /** 
         * Write the local solution, error, etc for a subdomain
         */
        virtual void writeLocal(int iter) { 
            subdomain->writeLocalSolutionToFile(iter); 
            //    heat->writeSolutionToFile(iter); 
            //    heat->writeErrorToFile(iter);  
        }

        /** Consolidate and write the global solution and grid to file 
         *  NOTE: this is currently a subset of the data available when
         *      writing local information (e.g., no error)
         */
        virtual void writeGlobal(int iter) {
            comm_unit->consolidateObjects(subdomain);
            comm_unit->barrier();
            subdomain->writeGlobalSolutionToFile(iter);
        }

        /** 
         * Redirect allows us to override if we want to do something
         * special on the final iteration. For example, maybe we only
         * want to write the full dataset at the end (including boundary
         * vectors, stencil indices, etc.)
         */
        virtual void writeFinal() {
            this->writeGlobal(-1); 
        }
};

#endif 
