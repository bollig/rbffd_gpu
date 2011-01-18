#ifndef _BOLLIG_CVT_H_
#define _BOLLIG_CVT_H_

#include <vector>
#include <map> 
#include "Vec3.h"
#include "density.h"
#include "timer_eb.h"
#include "KDTree.h"
#include "grids/grid_interface.h"

class CVT : public Grid
{
protected:
	Density* rho;

	std::map<std::string, Timer*> timers;
	KDTree* kdtree;

    	unsigned int dim_num;
  
       // Signed int iteration: 
       // iter < 0 => final
       // iter = 0 => initial
       // iter > 0 => iter # 
	unsigned int cvt_iter;
	unsigned int it_max;
        unsigned int write_freq; 	
	unsigned int nb_samples; 
        unsigned int sample_batch_size;

	// Current seed for cvt_sample
	int rand_seed; 


	// The number of nodes (from the beginning of the list) which are locked in place. 
	// These nodes are typically boundary nodes whos position should be updated by another algorithm
	unsigned int nb_locked_nodes; 

	// Can we assume the generators were initialized already or should we call cvt_init?
	bool generatorsInitialized; 

	// Different type of sampling for our domain
	enum sample_type {RANDOM, GRID, USER_SAMPLE, USER_INIT};

public:
        // Construct a CVT given the total number of nodes in the domain and 
	// the number of dimensions for the CVT. Dimension allows us to generate
	// a CVT in lower dimensions than the nodes (e.g., 3D node cloud with a
	// cvt on each xy plane) 
	CVT (unsigned int nb_nodes, unsigned int dimension, unsigned int nb_locked=0, unsigned int num_samples=2000, unsigned int max_num_iters=10, unsigned int write_frequency=5, unsigned int sample_batch_size=800); 
	CVT (std::vector<NodeType>& nodes, unsigned int dimension, unsigned int nb_locked=0, unsigned int num_samples=2000, unsigned int max_num_iters=10, unsigned int write_frequency=5, unsigned int sample_batch_size=800); 

	~CVT(); 


/*******************
 * OVERRIDES GRID::
 *******************/
	// Overrides Grid::generate()
	virtual void generate(); 
	
	// Overrides Grid::getFileDetailString()
	virtual std::string getFileDetailString(); 

	virtual std::string className() {return "cvt";}

	virtual void writeToFile() { 
		if (cvt_iter < it_max) { 
			Grid::writeToFile(this->getFilename(cvt_iter)); 
		} else { 
			Grid::writeToFile(this->getFilename(-1));
		} 
	}


/***********************
 * OVERRIDABLE ROUTINES:
 ***********************/
	// Customized initial sampling of domain could be redirected to the user_sample
	// so both node initialization and cvt sampling are the same
	//
	// For CVT:: this samples randomly in unit circle
	virtual void user_sample(std::vector<NodeType>& user_node_list, int indx_start, int n_now, bool init_rand); 

	// For CVT:: this samples randomly in unit circle
	virtual void user_init(std::vector<NodeType>& user_node_list, int indx_start, int n_now, bool init_rand); 


/***********************
 * HIDDEN ROUTINES
 ***********************/
protected: 
	void cvt_sample(std::vector<NodeType>& sample_node_list, int indx_start, int n_now, sample_type sample_kind, bool init_rand=false);
	void cvt_iterate(unsigned int sample_batch_size, unsigned int num_samples, sample_type sample);
	void find_closest(std::vector<NodeType>& sample_node_list, std::vector<NodeType>& generator_list, std::vector<unsigned int>& closest_indx_list); 

	unsigned long random_initialize(int seed); 
	void tuple_next_fast(int m, int n, int rank, int x[]);
	void initTimers();
	int get_seed(void);
	
#if 0	


    ~CVT() {
#if USE_KDTREE
        delete(kdtree);
#endif
        delete [] generators;
        tm.dumpTimings();
    }

    KDTree* getKDTree() {
        if (kdtree == NULL) {
    // Construct a kdtree for range_query
            kdtree = new KDTree(generators, nb_pts, dim_num);
        }
        return kdtree;
    }

    double cvt_energy(int dim_num, int n, int batch, int sample, bool initialize,
            int sample_num, int *seed, double r[]);
    void cvt_write(int dim_num, int n, int batch, int seed_init, int seed,
            const char *init_string, int it_max, int it_fixed, int it_num,
            double it_diff, double energy, const char *sample_string, int sample_num, double r[],
            const char *file_out_name, bool comment);
    void cvt_write_binary(int dim_num, int n, int batch, int seed_init, int seed,
            const char *init_string, int it_max, int it_fixed, int it_num,
            double it_diff, double energy, const char *sample_string, int sample_num, double r[],
            const char *file_out_name, bool comment);
    int data_read(const char *file_in_name, int dim_num, int n, double r[]);
    void timestamp(void);
    char *timestring(void);

    // Gordon Erlebacher and Evan Bollig:

    // Load the output file for specified iteration
    //  -1 = "final"
    //  -2 = "initial"
    //  0->inf = that specific iteration number.
    virtual int cvt_load(int iter = -1);

    // Write the output file for a specified iteration (should call to
    // cvt_write(...) 
    virtual void cvt_checkpoint(int iter);

    // Override this to change the prefix of the filename
    virtual void cvt_get_file_prefix(char* filename_buffer);

    // Override this to change suffix (iteration detail) format for the cvt output files
    virtual void cvt_get_filename(int iter, char *filename_buffer);

    void setNbBnd(int nb_bnd_) {
        this->nb_bnd = nb_bnd_;
    }

    int getNbBnd() {
        return nb_bnd;
    }

    void setNbPts(int nb_pts_) {
        this->nb_pts = nb_pts_;
    }

    int getNbPts() {
        return nb_pts;
    }

    void setDensity(Density* rho_) {
        this->rho = rho_;
    }

    void setBoundaryPts(std::vector<Vec3>& bndry_pts_) {
        this->bndry_pts = bndry_pts_;
    }

#endif 
};

#endif
