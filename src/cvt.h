#ifndef _BURKARDT_CVT_H_
#define _BURKARDT_CVT_H_

#include <vector>
#include "Vec3.h"
#include "density.h"

class CVT {
protected:
    std::vector<double> rd;
    int nb_bnd; // number of seeds on the boundary
    int nb_pts; // total number of seeds
    Density* rho;
    std::vector<Vec3> bndry_pts;

    // 0 = Debug output off; 1 = Verbose output and write intermediate files
    int DEBUG;

public:
    CVT(int DEBUG_ = 0);

    char ch_cap(char c);
    bool ch_eqi(char c1, char c2);
    int ch_to_digit(char c);
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
    void data_read(const char *file_in_name, int dim_num, int n, double r[]);
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
    // over your desired region.
    virtual void user(int dim_num, int n, int *seed, double r[]);

    // Override this routine to change initialization and execution details
    // (e.g., if you want to project as part of the initialization)
    virtual void cvt(int dim_num, int n, int batch, int init, int sample, int sample_num, int it_max, int it_fixed, int *seed, double r[], int *it_num, double *it_diff, double *energy);

    // Override this to customize the sampling within the domain (e.g., if you
    // want to sample uniformly in the SPHERE and not the CUBE.
    // NOTE: this is called within cvt_iterate multiple times and once form cvt(). 
    virtual void cvt_sample(int dim_num, int n, int n_now, int sample, bool initialize, int *seed, double r[]);

    // Override this to customize what happens during each iteration
    // of the CVT generation (e.g., if you want to perform projections
    virtual void cvt_iterate(int dim_num, int n, int batch, int sample, bool initialize, int sample_num, int *seed, double r[], double *it_diff, double *energy);

    double random(double a, double b);
    
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
};

#endif
