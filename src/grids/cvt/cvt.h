#ifndef _BURKARDT_CVT_H_
#define _BURKARDT_CVT_H_

#include <vector>
#include "Vec3.h"
#include "density.h"
#include "timer_eb.h"
#include "KDTree.h"
#include "grids/grid_interface.h"

class CVT : public Grid 
{
    protected:
        //  std::vector<double> rd;
        int nb_bnd; // number of seeds on the boundary
        int nb_pts; // total number of seeds
        Density* rho;
        std::vector<Vec3> bndry_pts;

        // 0 = Debug output off; 1 = Verbose output and write intermediate files
        int DEBUG;
        EB::TimerList timers;
        KDTree* kdtree;

        double PI;

        double *generators;

        int dim_num;

        // These correspond to John Burkardts CVT implementation parameters
        int seed, batch, it_fixed;
        // NOTE: we dont have support for changing "init","sample" and their
        // corresponding strings. 
        int init, sample;
        size_t sample_num, it_max; 
        size_t writeFreq;

    public:
        /*******************
         * OVERRIDES GRID::
         *******************/
        // Overrides Grid::generate()
        virtual void generate();

        virtual std::string className() {return "cvt";}

        virtual std::string getFileDetailString(); 

        /*******************
         *  CVT:: Only:
         *******************/
        // Construct a CVT given the total number of nodes in the domain and 
        // the number of dimensions for the CVT. Dimension allows us to generate
        // a CVT in lower dimensions than the nodes (e.g., 3D node cloud with a
        // cvt on each xy plane) 
        CVT (size_t nb_nodes, size_t dimension, size_t nb_locked=0, Density* density_function=NULL, size_t num_samples=10000, size_t max_num_iters=1000, size_t write_frequency=20, size_t sample_batch_size=1000); 
        CVT (std::vector<NodeType>& nodes, size_t dimension, size_t nb_locked=0, Density* density_function=NULL, size_t num_samples=10000, size_t max_num_iters=1000, size_t write_frequency=20, size_t sample_batch_size=1000); 

        virtual ~CVT() {
#if USE_KDTREE
            delete(kdtree);
#endif
            delete [] generators;
            timers["total"]->printAll();
        }

        void syncCVTandGrid() {
            for (size_t i = 0; i < nb_bnd; i++) {
                Vec3 nd;
                for (int j = 0; j < dim_num; j++) {
                    nd[j] = generators[i*dim_num + j];
                }
                this->setNode(i, nd);
                this->setBoundaryIndex(i, i);
                // TODO: boundary normals
                //            this->getBoundaryNormal(i) = computeBoundaryNormal(bndry_pts[i]);
            }
            for (size_t i = nb_bnd; i < nb_nodes; i++) {
                Vec3 nd;
                for (int j = 0; j < dim_num; j++) {
                    nd[j] = generators[i*dim_num + j];
                }
                this->setNode(i, nd);
            }
        }

        KDTree* getKDTree() {
            if (kdtree == NULL) {
                // Construct a kdtree for range_query
                kdtree = new KDTree(generators, nb_pts, dim_num);
            }
            return kdtree;
        }

        char ch_cap(char c);
        bool ch_eqi(char c1, char c2);
        int ch_to_digit(char c);
        double cvt_energy(int dim_num, int n, int batch, int sample, bool initialize,
                int sample_num, int *seed, double r[]);

        char digit_to_ch(int i);
        void find_closest(int dim_num, int n, int sample_num, double s[], double r[],
                int nearest[]);
        int get_seed(void);
        bool halham_leap_check(int dim_num, int leap[]);
        bool halham_n_check(int n);
        bool halham_dim_num_check(int dim_num);
        bool halham_seed_check(int dim_num, int seed[]);
        bool halham_step_check(int step);
        bool halton_base_check(int dim_num, int base[]);
        int i4_log_10(int i);
        int i4_max(int i1, int i2);
        int i4_min(int i1, int i2);
        void i4_to_halton_sequence(int dim_num, int n, int step, int seed[], int leap[],
                int base[], double r[]);
        char *i4_to_s(int i);
        int prime(int n);
        double r8_epsilon(void);
        double r8_huge(void);
        void r8mat_transpose_print(int m, int n, double a[], const char *title);
        void r8mat_transpose_print_some(int m, int n, double a[], int ilo, int jlo,
                int ihi, int jhi, const char *title);
        void r8mat_uniform_01(int m, int n, int *seed, double r[]);
        unsigned long random_initialize(int seed);
        void s_blank_delete(const char *s);
        void s_cap(const char *s);
        bool s_eqi(const char *s1, const char *s2);
        int s_len_trim(const char* s);
        double s_to_r8(const char *s, int *lchar, bool *error);
        bool s_to_r8vec(const char *s, int n, double rvec[]);
        void timestamp(void);
        char *timestring(void);
        void tuple_next_fast(int m, int n, int rank, int x[]);


        // Gordon Erlebacher and Evan Bollig:

        // Override this routine for a custom sampling routine
        // over your desired region. THIS IS FOR INITIALIZATION ONLY
        virtual void user_init(int dim_num, int n, int *seed, double r[]);

        // Override this routine for custom sampling for user defined sampling
        // (perhaps sampling with rejection outside your domain?)
        virtual void user_sample(int dim_num, int n, int *seed, double r[]);

        void rejection2d(int nb_samples, double area, double weighted_area, Density& density, vector<Vec3>& samples);
        virtual Vec3 singleRejection2d(double area, double weighted_area, Density& density);

        // Override this routine to change initialization and execution details
        // (e.g., if you want to project as part of the initialization)

        // INPUT ONLY : sample_num, it_max
        // OUTPUT ONLY: r, it_num, it_diff, energy
        virtual void cvt(int *it_num, double *it_diff, double *energy);

        // generate cvt using provided memory array r;
        void cvt(int *it_num, double *it_diff, double *energy, double r[]) {
            delete [] generators;
            generators = &r[0];
            this->cvt(it_num, it_diff, energy);
        }
        virtual void cvt(int dim_num_, int n, int batch_, int init_, int sample_, int sample_num_, int it_max_, int it_fixed_, int *seed_, double r[], int *it_num, double *it_diff, double *energy) {
            dim_num = dim_num_;
            nb_pts = n;
            seed = *seed_;
            batch = batch_;
            it_fixed = it_fixed_;
            init = init_;
            sample = sample_;
            sample_num = sample_num_;
            it_max = it_max_;

            this->cvt(it_num, it_diff, energy, r);
        }

        // Override this to customize the sampling within the domain (e.g., if you
        // want to sample uniformly in the SPHERE and not the CUBE.
        // NOTE: this is called within cvt_iterate multiple times and once form cvt(). 
        virtual void cvt_sample(int dim_num, int n, int n_now, int sample, bool initialize, int *seed, double r[]);

        // Override this to customize the initialization 
        virtual void cvt_init(int dim_num, int n, int n_now, int sample, bool initialize, int *seed, double r[]);

        // Override this to customize what happens during each iteration
        // of the CVT generation (e.g., if you want to perform projections
        virtual void cvt_iterate(int dim_num, int n, int batch, int sample, bool initialize, int sample_num, int *seed, double r[], double *it_diff, double *energy);

        double random(double a, double b);

        void setDensity(Density* rho_) {
            this->rho = rho_;
        }

        double* getGenerators() {
            return this->generators;
        }

        void setupTimers();

};

#endif
