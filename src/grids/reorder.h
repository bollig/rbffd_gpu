#ifndef __REORDER_H__
#define __REORDER_H__


#include <stdio.h>
#include <vector> // for ConvertMatrix
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/cuthill_mckee_ordering.hpp>
#include <boost/graph/properties.hpp>
#include <boost/graph/bandwidth.hpp>
#include <boost/numeric/ublas/vector_sparse.hpp>
#include <boost/numeric/ublas/matrix_sparse.hpp>

#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/vector_sparse.hpp>
#include <boost/numeric/ublas/matrix_sparse.hpp>

namespace bollig {

using namespace boost::numeric::ublas;


typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::undirectedS,
	boost::property<boost::vertex_color_t, boost::default_color_type,
	boost::property<boost::vertex_degree_t, size_t> > > AdjacencyGraph;

typedef boost::graph_traits<AdjacencyGraph>::vertex_descriptor VertexType;

typedef boost::graph_traits<AdjacencyGraph>::vertices_size_type size_type;


//----------------------------------------------------------------------
//----------------------------------------------------------------------
//----------------------------------------------------------------------
//----------------------------------------------------------------------
template <class MatType, class VecType>
class Reorder
{
    public:
        int nb_rows;
        std::vector<int> inv_order;
        std::vector<int> order;

    private:
        Reorder();

	public: 
        Reorder(int nb_rows)
        {
            this->nb_rows = nb_rows;
            order.resize(nb_rows);
            inv_order.resize(nb_rows);
        }

		void build_graph(MatType& mat, AdjacencyGraph& G) {
			for (typename MatType::const_iterator1 row_it = mat.begin1();
					row_it != mat.end1();
					++row_it) {
				for (typename MatType::const_iterator2 col_it = row_it.begin();
						col_it != row_it.end();
						++col_it) {
					boost::add_edge( col_it.index1(), col_it.index2(), G);
				}
			}
		}

		void get_cuthill_mckee_order(AdjacencyGraph& G) 
        {
			// The initial ordering
			boost::property_map<AdjacencyGraph,boost::vertex_index_t>::type
				index_map = get(boost::vertex_index, G);

            printf("after index map\n");

			// Use the boost::graph cuthill mckee algorithm
			std::vector<VertexType> inv_perm(boost::num_vertices(G));
			std::vector<size_type> perm(boost::num_vertices(G));
            //get(boost::vertex_color, G); // fast
            //printf("after get\n");
            //make_degree_map(G); // fast
            //printf("after make_degree_map\n");
            printf("before rcm\n");
            //printf("size of inv_perm= %d\n", inv_perm.size());
			cuthill_mckee_ordering(G, inv_perm.rbegin(), get(boost::vertex_color,G), make_degree_map(G));
            printf("after cuthill mcgee\n");

			// This order array will convert from reordered to the original ordering
			for ( size_t i = 0; i < inv_perm.size(); i++ )
				order[i] = index_map[inv_perm[i]];
		}

        //-----------------------------------------------------------
		void get_reordered_system(MatType& in_mat, VecType& in_vec, MatType& out_mat, VecType& out_vec) {
			// We construct an inverse map so we iterate ONLY through nonzero elements
			// in a sparse matrix.
            inv_order.resize(order.size());
       
            for (size_t i = 0; i < order.size(); i++) {
                inv_order[order[i]] = i;
            }

            printf("reorder: order.size: %d\n", order.size());
            printf("nb_rows= %d\n", nb_rows);
            //std::vector<int> new_col_id(col_id);
            std::vector<int> new_row_ptr(order.size()+1);

			for (typename MatType::const_iterator1 row_it = in_mat.begin1();
					row_it != in_mat.end1(); ++row_it)
            {
				for (typename MatType::const_iterator2 col_it = row_it.begin();
						col_it != row_it.end(); ++col_it)
               {
					size_t row_ind = inv_order[col_it.index1()];
					size_t col_ind = inv_order[col_it.index2()];

					out_mat(row_ind, col_ind) = *col_it;
				}
			}
			for (typename VecType::const_iterator row_it = in_vec.begin();
					row_it != in_vec.end(); ++row_it) {
				size_t row_ind = inv_order[row_it.index()];
				out_vec[row_ind] = *row_it;
			}
		}
        //----------------------------------------------------------------
        void getEllpackOutMatrix(std::vector<int>& col_ind, std::vector<int>& new_col_ind)
        {
            // TO DO:  We should have a routine to computer inv_order, and only when order changes
            // new_row_ptr is the same as row_ptr
            inv_order.resize(order.size());
       
            for (size_t i = 0; i < order.size(); i++) {
                inv_order[order[i]] = i;
            }

            // assumes nb of elements per row is constant (ellpack as opposed to csr format)
            int nnz = col_ind.size() / nb_rows;
            printf("getEllpackOutMatrix, nnz= %d\n", nnz);
            new_col_ind.resize(nnz*nb_rows);

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
                std::sort(&new_col_ind[0]+i*nnz, &new_col_ind[0]+(i+1)*nnz);
            }

            printf("inverse order array: \n");
            for (int i=0; i < inv_order.size(); i++) {
                if (i > 20) break;
                printf("%d, ", inv_order[i]);
            }
            printf("\n\n");

            for (int i=0; i < nb_rows; i++) {
                if (i > 20) break;
                printf("\nrow %d: ", i);
                for (int j=0; j < nnz; j++) {
                    printf("%d,", new_col_ind[j+i*nnz]);
                }
            }
            printf("\n");
        }
        //----------------------------------------------------

#if 0
		void get_original_order(VecType& in_vec, VecType& out_vec) {
			for (VecType::const_iterator row_it = in_vec.begin();
					row_it != in_vec.end();
					++row_it) {
				size_t row_ind = (*order)(row_it.index());
				out_vec(row_ind) = *row_it;
			}
		}
#endif

};
//----------------------------------------------------------------------
class ConvertMatrix
{
    typedef boost::numeric::ublas::compressed_matrix<double> MatType; 
    typedef boost::numeric::ublas::vector<double> VecType;
    typedef MatType CSR_Mat; 
    typedef VecType UBLAS_Vec; 

public:

    int nb_rows;
    Reorder<CSR_Mat, UBLAS_Vec>* r;
    CSR_Mat* AA;
    CSR_Mat* AA_reordered;
    UBLAS_Vec* VV;
    UBLAS_Vec* VV_reordered;
    std::vector<int>& col_ind;
    std::vector<int>& row_ptr;
    std::vector<int>& new_col_ind;
    std::vector<int>& new_row_ptr;

    ConvertMatrix(int nb_rows_, std::vector<int>& col_ind_, std::vector<int>& row_ptr_,
                                std::vector<int>& new_col_ind_, std::vector<int>& new_row_ptr_) :
        col_ind(col_ind_), row_ptr(row_ptr_),
        new_col_ind(new_col_ind_), new_row_ptr(new_row_ptr_)
    {
        this->nb_rows = nb_rows_;
        r = new Reorder<CSR_Mat, UBLAS_Vec>(nb_rows);
        AA = new compressed_matrix<double>(nb_rows, nb_rows, col_ind.size());
        AA_reordered = new compressed_matrix<double>(nb_rows, nb_rows, col_ind.size());
        VV = new UBLAS_Vec(nb_rows);
        VV_reordered = new UBLAS_Vec(nb_rows);
    }

    void convertToCSR()
    {
        CSR_Mat& A = *AA;

        // Convert to Boost CSR from non-boost CSR
        for (int k=0, i=0; i < nb_rows; i++) {
            int b = row_ptr[i];
            int e = row_ptr[i+1];
            for (int j=b; j < e; j++) {
                //A(i,col_ind[j]) = values[k];
                A(i,col_ind[j]) = 1;
                k++;
            }
        }
    }
    //------------------------------
    void reduceBandwidthRCM() 
    {
        CSR_Mat& A = *AA;
        CSR_Mat& A_reordered = *AA_reordered;
        UBLAS_Vec& V = *VV;
        UBLAS_Vec& V_reordered = *VV_reordered;

        AdjacencyGraph agraph; 

        //std::cout << A_reordered << std::endl;

        r->build_graph(A, agraph); 
        printf(".. after build_graph\n");
        r->get_cuthill_mckee_order(agraph); // takes forever with 20000 rows, 2 nnz per row
        printf(".. after get_cuthill_mckee-order\n");

        //r->get_reordered_system(A, V, A_reordered, V_reordered);
        //printf(".. after get_reordered_system\n");

        // get_reordered_system() must be called first
        r->getEllpackOutMatrix(col_ind, new_col_ind);
    }
    //----------------------------------------
    void printOutputMatrix()
    {
        // IS NOT WORKING
        std::cout << "\n\nOUTPUT: (not working properly?)\n";
        CSR_Mat& A_reordered = *AA_reordered;

        for (int i = 0; i < nb_rows; i ++ ){ 
            for (int j = 0; j < nb_rows; j++) { 
                std::cout << A_reordered(i,j) << " ";
            }
            std::cout << "\n";
        }
    }
    //---------------------------------------------

#if 0
    for (int i=0; i < nb_rows; i++) {
        int bb = i*nb_nnz_per_row;
        int be = (i+1)*nb_nnz_per_row;
        std::sort(&new_col_id[bb], &new_col_id[be]);
        std::sort(&col_id[bb], &col_id[be]);
    }

    bw = getBandwidth(new_col_id, nb_rows, nb_nnz_per_row);
#endif
//----------------------------------------------------------------------
}; // class
}; // namespace bollig
//----------------------------------------------------------------------
#endif
