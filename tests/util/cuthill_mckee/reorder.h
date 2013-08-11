#ifndef __REORDER_H__
#define __REORDER_H__

#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/cuthill_mckee_ordering.hpp>
#include <boost/graph/properties.hpp>
#include <boost/graph/bandwidth.hpp>
#include <boost/numeric/ublas/vector_sparse.hpp>
#include <boost/numeric/ublas/matrix_sparse.hpp>

#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/vector_sparse.hpp>
#include <boost/numeric/ublas/matrix_sparse.hpp>


typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::undirectedS,
	boost::property<boost::vertex_color_t, boost::default_color_type,
	boost::property<boost::vertex_degree_t, size_t> > > AdjacencyGraph;

typedef boost::graph_traits<AdjacencyGraph>::vertex_descriptor VertexType;

typedef boost::graph_traits<AdjacencyGraph>::vertices_size_type size_type;

template <class MatType, class VecType>
class Reorder
{

	public: 
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


		void get_cuthill_mckee_order(AdjacencyGraph& G, std::vector<int>& order) {

			// The initial ordering
			boost::property_map<AdjacencyGraph,boost::vertex_index_t>::type
				index_map = get(boost::vertex_index, G);

			// Use the boost::graph cuthill mckee algorithm
			std::vector<VertexType> inv_perm(boost::num_vertices(G));
			std::vector<size_type> perm(boost::num_vertices(G));
			cuthill_mckee_ordering(G, inv_perm.rbegin(), get(boost::vertex_color,G), make_degree_map(G));

			// This order array will convert from reordered to the original ordering
			for ( size_t i = 0; i < inv_perm.size(); i++ )
				order[i] = index_map[inv_perm[i]];

		}


		void get_reordered_system(MatType& in_mat, VecType& in_vec, std::vector<int>& order, MatType& out_mat, VecType& out_vec) {

			// We construct an inverse map so we iterate ONLY through nonzero elements
			// in a sparse matrix.
			std::vector<size_t> inv_order(order.size());

			for (size_t i = 0; i < order.size(); i++) {
				inv_order[order[i]] = i;
			}

			for (typename MatType::const_iterator1 row_it = in_mat.begin1();
					row_it != in_mat.end1();
					++row_it) {
				for (typename MatType::const_iterator2 col_it = row_it.begin();
						col_it != row_it.end();
						++col_it) {
					size_t row_ind = inv_order[col_it.index1()];
					size_t col_ind = inv_order[col_it.index2()];

					out_mat(row_ind, col_ind) = *col_it;
				}
			}
			for (typename VecType::const_iterator row_it = in_vec.begin();
					row_it != in_vec.end();
					++row_it) {
				size_t row_ind = inv_order[row_it.index()];
				out_vec[row_ind] = *row_it;
			}
		}


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

#endif
