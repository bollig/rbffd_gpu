// Evan Bollig
// Sample read write of matrix market format 
// borrowed from examples posted here:
//   http://math.nist.gov/MatrixMarket/mmio-c.html

#include <string>
#include <stdio.h>
#include <stdlib.h>
#include "mmio.h"

//------------------------------------------------------------------------
int test_mm_write(std::string filename)
{
    const int nz = 4;
    const int M = 10;
    const int N= 10;

    MM_typecode matcode;                        
    int I[nz] = { 0, 4, 2, 8 };
    int J[nz] = { 3, 8, 7, 5 };
    double val[nz] = {1.1, 2.2, 3.2, 4.4};
    int i;

    int err = 0; 
    FILE *f; 
    f = fopen(filename.c_str(), "w"); 
    err += mm_initialize_typecode(&matcode);
    err += mm_set_matrix(&matcode);
    err += mm_set_coordinate(&matcode);
    err += mm_set_real(&matcode);

    err += mm_write_banner(f, matcode); 
    err += mm_write_mtx_crd_size(f, M, N, nz);

    /* NOTE: matrix market files use 1-based indices, i.e. first element
       of a vector has index 1, not 0.  */
    fprintf(stdout, "Writing file contents: \n"); 
    for (i=0; i<nz; i++)
    {
        fprintf(f, "%d %d %lg\n", I[i]+1, J[i]+1, val[i]);
        fprintf(stdout, "%d %d %lg\n", I[i]+1, J[i]+1, val[i]);
    }

    fclose(f);

    return err;
}
//------------------------------------------------------------------------
int test_mm_read(std::string filename)
{
    int ret_code;
    MM_typecode matcode;
    FILE *f;
    int M, N, nz;   
    int i, *I, *J;
    double *val;    
    int err=0; 

    if ((f = fopen(filename.c_str(), "r")) == NULL) 
        return -1;

    if (mm_read_banner(f, &matcode) != 0)
    {
        printf("Could not process Matrix Market banner.\n");
        return -2;
    }


    /*  This is how one can screen matrix types if their application */
    /*  only supports a subset of the Matrix Market data types.      */

    if (mm_is_complex(matcode) && mm_is_matrix(matcode) && 
            mm_is_sparse(matcode) )
    {
        printf("Sorry, this application does not support ");
        printf("Market Market type: [%s]\n", mm_typecode_to_str(matcode));
        return -3;
    }

    /* find out size of sparse matrix .... */

    if ((ret_code = mm_read_mtx_crd_size(f, &M, &N, &nz)) !=0)
        return -4;


    /* reseve memory for matrices */

    I = (int *) malloc(nz * sizeof(int));
    J = (int *) malloc(nz * sizeof(int));
    val = (double *) malloc(nz * sizeof(double));


    /* NOTE: when reading in doubles, ANSI C requires the use of the "l"  */
    /*   specifier as in "%lg", "%lf", "%le", otherwise errors will occur */
    /*  (ANSI C X3.159-1989, Sec. 4.9.6.2, p. 136 lines 13-15)            */

    for (i=0; i<nz; i++)
    {
        fscanf(f, "%d %d %lg\n", &I[i], &J[i], &val[i]);
        I[i]--;  /* adjust from 1-based to 0-based */
        J[i]--;
    }

    if (f !=stdin) fclose(f);

    /************************/
    /* now write out matrix */
    /************************/

    fprintf(stdout, "Read full file contents: \n================================\n"); 
    err += mm_write_banner(stdout, matcode);
    err += mm_write_mtx_crd_size(stdout, M, N, nz);
    for (i=0; i<nz; i++)
    {
        fprintf(stdout, "%d %d %lg\n", I[i]+1, J[i]+1, val[i]);
    }

    return err;
}
//------------------------------------------------------------------------

int main(int argc, char** argv) {
    std::string fname = "test_mm_io.mtx";
    if (argc > 1) {
        fname = argv[1]; 
    }
    int err=0; 

    err += test_mm_write(fname); 
    err += test_mm_read(fname); 

    return err; 
}
