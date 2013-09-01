#ifndef _VCL_BANDWIDTH_REDUCITON_H_
#define _VCL_BANDWIDTH_REDUCITON_H_
/* =========================================================================
   Copyright (c) 2010-2013, Institute for Microelectronics,
                            Institute for Analysis and Scientific Computing,
                            TU Wien.
   Portions of this software are copyright by UChicago Argonne, LLC.

                            -----------------
                  ViennaCL - The Vienna Computing Library
                            -----------------

   Project Head:    Karl Rupp                   rupp@iue.tuwien.ac.at
               
   (A list of authors and contributors can be found in the PDF manual)

   License:         MIT (X11), see file LICENSE in the base directory
============================================================================= */

/*
*   Tutorial: Matrix bandwidth reduction algorithms
*/


#include <stdio.h>
#include <iostream>
#include <fstream>
#include <string>
#include <algorithm>
#include <map>
#include <vector>
#include <deque>
#include <cmath>

#include "viennacl/misc/bandwidth_reduction.hpp"


namespace vcl {

//----------------------------------------------------------------------

class RCM_VCL
{
    public:

//
// Part 1: Helper functions
//

// Reorders a matrix according to a previously generated node
// number permutation vector r
    //----------------------------------------------------------------------
    //----------------------------------------------------------------------
    std::vector< std::map<int, double> > reorder_matrix(std::vector< std::map<int, double> > const & matrix, std::vector<int> const & r)
    {
        std::vector< std::map<int, double> > matrix2(r.size());
        std::vector<std::size_t> r2(r.size());
        
        for (std::size_t i = 0; i < r.size(); i++)
            r2[r[i]] = i;

        for (std::size_t i = 0; i < r.size(); i++)
            for (std::map<int, double>::const_iterator it = matrix[r[i]].begin();  it != matrix[r[i]].end(); it++)
                matrix2[i][r2[it->first]] = it->second;
        
        return matrix2;
    }
    //----------------------------------------------------------------------

// Calculates the bandwidth of a matrix
    int calc_bw(std::vector< std::map<int, double> > const & matrix, float& bw_mean, float& bw_std)
    {
        int bw = 0;
        std::vector<int> bwr(matrix.size());
        
        for (std::size_t i = 0; i < matrix.size(); i++) {
            int col_min = matrix.size() + 1;
            int col_max = -1;
            for (std::map<int, double>::const_iterator it = matrix[i].begin();  it != matrix[i].end(); it++) {
                col_min = (it->first < col_min) ? it->first : col_min;
                col_max = (it->first > col_max) ? it->first : col_max;
           }
           bwr[i] = col_max - col_min + 1;
           bw = std::max(bw, bwr[i]);
        }

        // mean bandwidth
        bw_mean = 0;
        for (int i=0; i < matrix.size(); i++) {
            bw_mean += bwr[i];
        }
        bw_mean /= matrix.size();

        // bandwidth standard deviation
        bw_std = 0.0;
        for (int i=0; i < matrix.size(); i++) {
            bw_std += (bwr[i] - bw_mean) * (bwr[i] - bw_mean);
        }
        bw_std /= matrix.size();
        bw_std = sqrt(bw_std);

        //printf("bw= %d, mean(bw)= %f, std(bw) = %f\n", bw, bw_mean, bw_std);
        
        return bw;
    }
    //----------------------------------------------------------------------
    // Calculate the bandwidth of a reordered matrix
    int calc_reordered_bw(std::vector< std::map<int, double> > const & matrix,  std::vector<int> const & r) 
        //, float& bw_mean, float& bw_std)
    {
        std::vector<int> r2(r.size());
        int bw = 0;

        // evaluate the number of nodes outside a specified bandwidth
        //int spec_bw = 4000;  // specified bandwidth
        int count_outside; // number of nodes outside a bandwidth of 4000
        
        for (std::size_t i = 0; i < r.size(); i++)
            r2[r[i]] = i;

        for (std::size_t i = 0; i < r.size(); i++) {
            int col_min = matrix.size() + 1;
            int col_max = -1;
            int row_bw = 0.0;
            for (std::map<int, double>::const_iterator it = matrix[r[i]].begin();  it != matrix[r[i]].end(); it++) {
                int rfirst = r2[it->first];
                col_min = (rfirst < col_min) ? rfirst : col_min;
                col_max = (rfirst > col_max) ? rfirst : col_max;
                //int width = static_cast<int>(r2[it->first]);
                //if (width > spec_bw) count_outside++;
                //row_bw = std::max(bw, std::abs(static_cast<int>(i - r2[it->first])));
                //bw = std::max(bw, std::abs(static_cast<int>(i - r2[it->first])));
            }
           bw = std::max(bw, col_max-col_min+1);
           // printf("row %d, bw: %d\n", i, row_bw);
       }

        //printf("nb points ouside bandwidth of %d : %d\n", spec_bw, count_outside);
        //printf("reordered bw= %d\n", bw);
        
        return bw;
    }
    //----------------------------------------------------------------------
    // Generates a random permutation by Knuth shuffle algorithm
    // reference: http://en.wikipedia.org/wiki/Knuth_shuffle 
    //  (URL taken on July 2nd, 2011)
    std::vector<int> generate_random_reordering(int n)
    {
        std::vector<int> r(n);
        int tmp;
        int j;
        
        for (int i = 0; i < n; i++)
            r[i] = i;
        
        for (int i = 0; i < n - 1; i++)
        {
            j = i + static_cast<std::size_t>((static_cast<double>(rand()) / static_cast<double>(RAND_MAX)) * (n - 1 - i));
            if (j != i)
            {
                tmp = r[i];
                r[i] = r[j];
                r[j] = tmp;
            }
        }
        
        return r;
    }
    //----------------------------------------------------------------------
    // function for the generation of a three-dimensional mesh incidence matrix
    //  l:  x dimension
    //  m:  y dimension
    //  n:  z dimension
    //  tri: true for tetrahedral mesh, false for cubic mesh
    //  return value: matrix of size l * m * n
    std::vector< std::map<int, double> > gen_3d_mesh_matrix(int l, int m, int n, bool tri)
    {
        std::vector< std::map<int, double> > matrix;
        int s;
        int ind;
        int ind1;
        int ind2;
        
        s = l * m * n;
        matrix.resize(s);
        for (int i = 0; i < l; i++)
        {
            for (int j = 0; j < m; j++)
            {
                for (int k = 0; k < n; k++)
                {
                    ind = i + l * j + l * m * k;
                    
                    matrix[ind][ind] = 1.0;
                    
                    if (i > 0)
                    {
                        ind2 = ind - 1;
                        matrix[ind][ind2] = 1.0;
                        matrix[ind2][ind] = 1.0;
                    }
                    if (j > 0)
                    {
                        ind2 = ind - l;
                        matrix[ind][ind2] = 1.0;
                        matrix[ind2][ind] = 1.0;
                    }
                    if (k > 0)
                    {
                        ind2 = ind - l * m;
                        matrix[ind][ind2] = 1.0;
                        matrix[ind2][ind] = 1.0;
                    }
                    
                    if (tri)
                    {
                        if (i < l - 1 && j < m - 1)
                        {
                            if ((i + j + k) % 2 == 0)
                            {
                                ind1 = ind;
                                ind2 = ind + 1 + l;
                            }
                            else
                            {
                                ind1 = ind + 1;
                                ind2 = ind + l;
                            }
                            matrix[ind1][ind2] = 1.0;
                            matrix[ind2][ind1] = 1.0;
                        }
                        if (i < l - 1 && k < n - 1)
                        {
                            if ((i + j + k) % 2 == 0)
                            {
                                ind1 = ind;
                                ind2 = ind + 1 + l * m;
                            }
                            else
                            {
                                ind1 = ind + 1;
                                ind2 = ind + l * m;
                            }
                            matrix[ind1][ind2] = 1.0;
                            matrix[ind2][ind1] = 1.0;
                        }
                        if (j < m - 1 && k < n - 1)
                        {
                            if ((i + j + k) % 2 == 0)
                            {
                                ind1 = ind;
                                ind2 = ind + l + l * m;
                            }
                            else
                            {
                                ind1 = ind + l;
                                ind2 = ind + l * m;
                            }
                            matrix[ind1][ind2] = 1.0;
                            matrix[ind2][ind1] = 1.0;
                        }
                    }
                }
            }
        }
    
        return matrix;
    }
}; // class

//----------------------------------------------------------------------
class ConvertMatrix
{
public:
    typedef std::vector< std::map<int, double> > MAP;
    typedef std::map<int, double>::const_iterator MapIter;
    RCM_VCL rvcl;
    int nb_rows;
    std::vector<int>& col_ind;
    std::vector<int>& row_ptr;
    std::vector<int>& new_col_ind;
    std::vector<int>& new_row_ptr;
    MAP matrix; 
    MAP matrix2; 
    MAP matrix3; 
    std::vector<int> order;
    std::vector<int> inv_order;

public:

    ConvertMatrix(int nb_rows_, std::vector<int>& col_ind_, std::vector<int>& row_ptr_,
                                std::vector<int>& new_col_ind_, std::vector<int>& new_row_ptr_,
                                int sym_adj) :
        col_ind(col_ind_), row_ptr(row_ptr_),
        new_col_ind(new_col_ind_), new_row_ptr(new_row_ptr_)
    {
        this->nb_rows = nb_rows_;
        matrix.resize(nb_rows);
        int sz = col_ind.size();

    if (sym_adj == 1) {
        printf("SYMMETRIZE the adjacency graph\n");
    } else {
        printf("Do NOT SYMMETRIZE adjacency graph\n");
    }

        for (int i=0; i < nb_rows; i++) {
            int b = row_ptr[i];
            int e = row_ptr[i+1];
            for (int j=b; j < e; j++) {
                int col = col_ind[j];
                matrix[i][col] = 1.0;
                if (sym_adj == 1) {
                    matrix[col][i] = 1.0;  // symmetrize for ViennaCL to work (need input control)
                }
            }
        }

        // Add a unit vector to the diagonal
        if (sym_adj == 1) {
            for (int i=0; i < nb_rows; i++) {
                matrix[i][i] = 1.0;  // symmetrize for ViennaCL to work (need input control)
            }
        }

#if 0
        for (int i=0; i < nb_rows; i++) {
            int sz = matrix[i].size();
            for (MapIter it = matrix[i].begin();  it != matrix[i].end(); it++) {
                printf("mat(%d,%d) : %f\n", i, it->first, it->second);
            }
        }
#endif
    }
    //---------------------------------------------------------
    void reorderMatrix()
    {
        matrix2.resize(order.size());

        // matrix without the extra symmetry components
        matrix3.resize(order.size());

        std::vector<std::size_t> inv_order(order.size());

        for (std::size_t i = 0; i < order.size(); i++) {
            inv_order[order[i]] = i;
        }
        
        //--------------------
        //
        //MATRIX 2 and matrix 3 MUST BE identical (at least n the SYMMETRIX case in terms of bandwidth structure). 
        //They are not. WHY NOT? THEY ARE CLOSE THOUGH. 
        for (std::size_t i = 0; i < order.size(); i++) {
            for (MapIter it = matrix[order[i]].begin();  it != matrix[order[i]].end(); it++) {
                matrix2[i][inv_order[it->first]] = it->second;
            }
        }
        //printf("\n\n");

        //--------------------
        // Regenerate the reordered matrix from row_ptr
        // Assume initial matrix was Ellpack
        for (int i=0; i < nb_rows; i++) {
            int new_row = inv_order[i];
            // number elements on transformed row is the same as on original row
            int b = row_ptr[i];
            int e = row_ptr[i+1];
            //printf("b,e= %d, %\n", b, e);
            matrix3[new_row].clear();
            for (int j=b; j < e; j++) {
                int new_col = inv_order[ col_ind[j] ];
                matrix3[new_row][new_col] = 1.;
            }
        }

        //printf("matrix2 and matrix3 should be identical!!!! ARE THEY?\n");

        //--------------------
#if 0
        printf("\n PRINT row_ptr\n");
        for (int i=0; i < nb_rows; i++) {
            if (i > 50) break;
            int b = row_ptr[i];
            int e = row_ptr[i+1];
            printf("\nrow %d: ", i);
            for (int j=b; j < e; j++) {
                printf("%d, ", col_ind[j]);
            }
        }
        printf("\n\n");
#endif

#if 0
        for (int i=0; i < nb_rows; i++) {
            if (i > 20) break;
            for (MapIter it = matrix2[i].begin();  it != matrix2[i].end(); it++) {
                printf("mat2(%d,%d) : %f\n", i, it->first, it->second);
            }
        }

        printf("\n\n\n");
        for (int i=0; i < nb_rows; i++) {
            if (i > 20) break;
            for (MapIter it = matrix3[i].begin();  it != matrix3[i].end(); it++) {
                printf("mat3(%d,%d) : %f\n", i, it->first, it->second);
            }
        }
#endif


        //printf("\n\n");
    }
    //----------------------------------------------------------------------
    void computeReorderedEllMatrix(int nb_nonzeros_per_row)
    {
        // Soft Ellpack matrix before exiting
        printf("computer reordered\n");
        printf("order size: %d\n", order.size());
        int nnz = nb_nonzeros_per_row;
        inv_order.resize(order.size());

        for (std::size_t i = 0; i < order.size(); i++) {
            inv_order[order[i]] = i;
        }

        //printf("nb_rows = %d\n", nb_rows);
        new_col_ind.resize(col_ind.size());

        // first reorder row_ptr
        #if 0
        for (int i=0; i < row_ptr.size()-1; i++) {
            //printf("row_ptr[%d] = %d\n", i, row_ptr[i]);
            int new_row = inv_order[i];
        }
        #endif

#if 1
        // nnz not defined
 
            for (int i=0; i < nb_rows; i++) {
                for (int j=0; j < nnz; j++) {
                    int row_index = inv_order[i];
                    int col_index = inv_order[col_ind[j+i*nnz]];
                    new_col_ind[j+row_index*nnz] = col_index;
                }
                // Placing the sort at this location does not work for reasons unknown
                // It is because new_col_ind is not filled one of ITS rows at a time
            }
            for (int i=0; i < nb_rows; i++) {
                std::sort(&new_col_ind[i*nnz], &new_col_ind[(i+1)*nnz]);
            }
#endif
    }
    //---------------------------------------------------------
    int calcOrigBandwidth(float& mean, float& std) 
    {
        return rvcl.calc_bw(matrix, mean, std);
    }
    //----------------------------------------------------------------------
    int calcFinalBandwidth(float& mean, float& std) 
    {
        int bw2 = rvcl.calc_bw(matrix2, mean, std);
        int bw3 = rvcl.calc_bw(matrix3, mean, std);
        printf("b3(matrix2)= %d\n", bw2);
        printf("b3(matrix2)= %d\n", bw3);
        return bw3;
    }
    //---------------------------------------------------------
    int bandwidthEllpack(int* col_id, int nb_rows, int stencil_size)
    // compute bandwidth, average bandwidth, rms bandwidth
    // given row r, and stencil element s, col_id[s+stencil_size*r] is the index into the 
    // vector vec of length vec_size
    {
        // assume col_id are sorted !!!
        //printf("nb_rows= %d\n", nb_rows);
        //printf("stencil_size= %d\n", stencil_size);
        std::vector<int> bw(nb_rows);
        int max_bandwidth = 0;

        //printf("col_id, nb rows: %d\n", nb_rows);
        for (int r=0; r < nb_rows; r++) {
            // sort row that is unsorted in Evan code
             //printf("before sort, size: %d\n", stencil_size);
            //std::sort(col_id+r*stencil_size, col_id+(r+1)*stencil_size);
            bw[r] = col_id[(r+1)*stencil_size-1] - col_id[r*stencil_size] + 1;
            //printf("bw[%d]= %d\n", r, bw[r]);
            max_bandwidth = bw[r] > max_bandwidth ? bw[r] : max_bandwidth;

            #if 0
            if (r > 262000) {
                printf("\n==== bandwidth, row %d: \n", r);
                for (int j=0; j < stencil_size; j++) {
                    printf("%d,", col_id[j+stencil_size*r]);
                }
            } else { exit(0); }
            #endif
        }
        //printf("\n");
        //printf("max bandwidth: %d\n", max_bandwidth);
        return(max_bandwidth);
    }
    //----------------------------------------------------------------------
    int calc_bw(std::vector< std::map<int, double> > const & matrix, float& mean, float& std)
    {
        return rvcl.calc_bw(matrix, mean, std);
    }
    //---------------------------------------------------------
    //int calcReorderedBandwidth()
    //{
        //rvcl.calc_bw();
        //return rvcl.calc_...(matrix);
    //}
    //---------------------------------------------------------
    void reduceBandwidthRCM() 
    {
        float bw_mean, bw_std;
        int bw;
        //r = rvcl.generate_random_reordering(n);
        //std::vector<int> r = rvcl.generate_random_reordering(n);
        
        // Reorder using Cuthill-McKee algorithm
        
        std::cout << "-- Cuthill-McKee algorithm --" << std::endl;
        //std::cout << " * Original bandwidth: " << rvcl.calc_bw(matrix) << std::endl;
        bw = rvcl.calc_bw(matrix, bw_mean, bw_std);
        printf("original bandwidth/mean/std: %d, %f, %f\n", bw, bw_mean, bw_std);
        printf("matrix size: %d\n", matrix.size());
        order = viennacl::reorder(matrix, viennacl::cuthill_mckee_tag());
        //std::cout << " * Reordered bandwidth(matrix): " << rvcl.calc_reordered_bw(matrix, order) << std::endl;
        reorderMatrix(); // matrix -> matrix2
        bw = rvcl.calc_bw(matrix2, bw_mean, bw_std);
        printf("Reordered(matrix2) bandwidth/mean/std: %d, %f, %f\n", bw, bw_mean, bw_std);
        bw = rvcl.calc_bw(matrix3, bw_mean, bw_std);
        printf("Reordered(matrix3) bandwidth/mean/std: %d, %f, %f\n", bw, bw_mean, bw_std);
        //printf("reorder matrix\n");

        //std::cout << " * Bandwidth(matrix2): " << rvcl.calc_bw(matrix2) << std::endl;

        //
        // Reorder using advanced Cuthill-McKee algorithm
        //
        //std::cout << "-- Advanced Cuthill-McKee algorithm --" << std::endl;
        //double a = 0.0;
        //std::size_t gmax = 1;
        //r = viennacl::reorder(matrix2, viennacl::advanced_cuthill_mckee_tag(a, gmax));
        //std::cout << " * Reordered bandwidth: " << rvcl.calc_reordered_bw(matrix2, r) << std::endl;

    }
    //---------------------------------------------------------
    void registerDensity(std::vector<int>& col_id, int nb_rows, int stencil_size, const std::string& msg)
    {
        // Assumes Ellpack format (constant number of nonzeros per row
        // (Generalize later)
        // Assumes that each row is sorted
        // one cache line of floats ==> 16 floats (64 bytes)
        //printf("ENTER REGISTER DENSITY\n");
        int nb_per_reg = 64 / sizeof(float);
        //printf("register density, nb_rows= %d\n", nb_rows);
        int reg_per_row = nb_rows / nb_per_reg + 1;
        //printf("line 2\n");
        std::vector<int> reg(reg_per_row);
        std::vector<float> reg_density(nb_rows,0.);
        std::vector<int> nb_nonzero_reg(nb_rows,0);
        //printf("line 3\n");
        //printf("nb_rows= %d\n", nb_rows);
        //printf("stencil_size= %d\n", stencil_size);

        // min/max cache lines on row i
#define minreg(i)   col_id[(0)+(i)*stencil_size] / nb_per_reg; 
#define maxreg(i)   col_id[(stencil_size-1)+(i)*stencil_size] / nb_per_reg + 1; 
//#define REG(i,c)     col_id[(c)+(i)*stencil_size] / nb_per_reg; 
#define REG(c)     col_id[(c)] / nb_per_reg; 
        
        //printf("before row loop, %d\n", nb_rows);
        //printf("col_id.size= %d\n", col_id.size());
        for (int i=0; i < nb_rows; i++) {
            //int mnreg = minreg(i);
            //printf("stencil_size= %d\n", stencil_size);
            //printf("nb_per_reg= %d\n", nb_per_reg);
            //printf("col_id address: %ld\n", (long) &col_id[0]);
            //printf("row %d, col_id= %d\n", i, col_id[i*stencil_size]);
            int mnreg = minreg(i);
            int mxreg = maxreg(i)+1;
            //printf("mn/mx reg= %d, %d\n", mnreg, mxreg);
            int b = i*stencil_size;
            int e = (i+1)*stencil_size;
            std::fill(reg.begin()+mnreg, reg.begin()+mxreg, 0);
            for (int c=b; c < e; c++) {
                int r = REG(c);
                reg[r] += 1;
            }
            for (int r=mnreg; r < mxreg; r++) {
                if (reg[r] > 0) nb_nonzero_reg[i]++;
            }
#if 0
            if (i < 10) {
                printf("---- row %d ----\ncols: ", i);
                printf("mnreg, mxreg= %d, %d\n", mnreg, mxreg);
                for (int r=0; r < stencil_size; r++) {
                    printf("%d, ", col_id[r+stencil_size*i]);
                }
                printf("\n(equal to previous, 2nd way: ");
                for (int r=b; r < e; r++) {
                    printf("%d, ", col_id[r]);
                }
                printf("\n(reg#,reg[reg#]): ");
                for (int r=mnreg; r < mxreg; r++) {
                    printf("%d-%d, ", r,reg[r]);
                }
                printf("\n");
            }
#endif
            // the denominator should really be the number of cache lines with content
            // as opposed to the number of cache lines that span the row (which is larger)
            reg_density[i] = ((float) stencil_size) / (nb_nonzero_reg[i]*nb_per_reg);
#if 0
            if (reg_density[i] < 0.06) {
                printf("mxreg-mnreg= %d, nonzero_reg_per_row: %d, dens: %f\n", mxreg-mnreg, nb_nonzero_reg[i], reg_density[i]);
            }
#endif
        }

        #undef minreg
        #undef maxreg
        #undef REG

        // Average register density
        float mean_dens = 0.;
        float std_dens = 0.;
        for (int i=0; i < nb_rows; i++) {
            mean_dens += reg_density[i];
        }
        mean_dens /= nb_rows;

        for (int i=0; i < nb_rows; i++) {
            std_dens += (reg_density[i]-mean_dens)*(reg_density[i]-mean_dens);
        }
        std_dens = sqrt(std_dens/nb_rows);
        printf("%s: mean reg density: %f, std reg density: %f\n", msg.c_str(), mean_dens, std_dens);
    }
    //---------------------------------------------------------
    //---------------------------------------------------------
    //---------------------------------------------------------
    //---------------------------------------------------------
}; // class ConvertMatrix


}; // namespace


#endif
