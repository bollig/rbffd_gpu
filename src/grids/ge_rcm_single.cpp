//=======================================================================
// Copyright 1997, 1998, 1999, 2000 University of Notre Dame.
// Authors: Andrew Lumsdaine, Lie-Quan Lee, Jeremy G. Siek
//          Doug Gregor, D. Kevin McGrath
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)
//
// geRCM written by Gordon Erlebacher, Aug. 2013, based on code above.
//=======================================================================

#include <boost/config.hpp>
#include <vector>
#include <iostream>
#include <stdio.h>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/cuthill_mckee_ordering.hpp>
#include <boost/graph/properties.hpp>
#include <boost/graph/bandwidth.hpp>

namespace gercm {
using namespace boost;

// Details description of Boost Graph Library
// http://www.boost.org/doc/libs/1_38_0/libs/graph/doc/using_adjacency_list.html#sec:choosing-graph-type

/*
  Sample Output
  original bandwidth: 8
  Reverse Cuthill-McKee ordering starting at: 6
    8 3 0 9 2 5 1 4 7 6 
    bandwidth: 4
  Reverse Cuthill-McKee ordering starting at: 0
    9 1 4 6 7 2 8 5 3 0 
    bandwidth: 4
  Reverse Cuthill-McKee ordering:
    0 8 5 7 3 6 4 2 1 9 
    bandwidth: 4
 */
//----------------------------------------------------------------------
int getBandwidth(std::vector<int>& col_id, int nb_rows, int nb_nnz_row)
{
    int max_bw = -1;

    if (col_id.size() != (nb_nnz_row * nb_rows) ) {
        printf("col_id.size() : %d\n", col_id.size());
        printf("nb_nnz_per row: %d\n", nb_nnz_row);
        printf("nb_rows= %d\n", nb_rows);
        exit(1);
    }

    for (int i=0; i < nb_rows; i++) {
        int bw = col_id[nb_nnz_row-1+i*nb_nnz_row] - col_id[i*nb_nnz_row] + 1;
        max_bw = (bw > max_bw) ? bw : max_bw;
    }

    printf("bandwidth = %d\n", max_bw);
    return(max_bw);
}
//----------------------------------------------------------------------
void geRCM(std::vector<int>& col_id, int nb_rows, int nb_nnz_per_row, std::vector<int>& perm, std::vector<int>& perm_inv)
{
  using namespace boost;
  using namespace std;
  int bw;

  // TEST
  nb_nnz_per_row = 2;
  col_id.resize(nb_nnz_per_row*nb_rows);
  for (int i=0; i < nb_rows; i++) {
      col_id[0+i*nb_nnz_per_row] = i;
      col_id[1+i*nb_nnz_per_row] = nb_rows-1-i;
      std::sort(&col_id[0]+i*nb_nnz_per_row, &col_id[0]+(i+1)*nb_nnz_per_row);
  }

// Consider a 4 by 4 test matrix
//   0 1 0 1
//   0 0 1 1
//   1 0 1 0
//   1 1 0 0
//

#if 0
  printf("test matrix: \n");
   printf("0 1 0 1\n");
   printf("0 0 1 1\n");
   printf("1 0 1 0\n");
   printf("1 1 0 0\n\n");

#if 1
  nb_rows = 4;
  nb_nnz_per_row = 2;
  col_id.resize(8);
  int co = 0;
  col_id[co++] = 1;
  col_id[co++] = 3;
  col_id[co++] = 2;
  col_id[co++] = 3;
  col_id[co++] = 0;
  col_id[co++] = 2;
  col_id[co++] = 0;
  col_id[co++] = 1;
#endif
#endif

    printf("ENTER geRCM\n");
    bw = getBandwidth(col_id, nb_rows, nb_nnz_per_row);


  if (col_id.size() != (nb_nnz_per_row * nb_rows) ) {
    printf("col_id.size() : %d\n", col_id.size());
    printf("nb_nnz_per row: %d\n", nb_nnz_per_row);
    printf("nb_rows= %d\n", nb_rows);
  }

  typedef adjacency_list<vecS, vecS, undirectedS, 
     property<vertex_color_t, default_color_type,
       property<vertex_degree_t,int> > > Graph;
  typedef graph_traits<Graph>::vertex_descriptor Vertex;
  typedef graph_traits<Graph>::vertices_size_type size_type;

  typedef std::pair<std::size_t, std::size_t> Pair;
  std::vector<Pair> edges(col_id.size());
  Pair p;

  int count=0;
  for (int r=0; r < nb_rows; r++) {
    p.first = r;
    for (int i=0; i < nb_nnz_per_row; i++) {
        p.second = col_id[i+nb_nnz_per_row*r];
//        if (p.first == p.second) continue;
        edges[i] = p;
        count++;
    }
  }

  edges.resize(count);

  printf("nb_rows: %d, nb pairs: %d\n", nb_rows, edges.size());

  Graph G(nb_rows);

  // I kept diagonal elements (which reference themselves)
    for (int i = 0; i < nb_rows; ++i) {
        add_edge(edges[i].first, edges[i].second, G);
    }


#if 1
// get_cuthill_McGee_order (From Bollig)
  graph_traits<Graph>::vertex_iterator ui, ui_end;

  property_map<Graph,vertex_degree_t>::type deg = get(vertex_degree, G);
  for (boost::tie(ui, ui_end) = vertices(G); ui != ui_end; ++ui)
    deg[*ui] = degree(*ui, G);

  property_map<Graph, vertex_index_t>::type index_map = get(vertex_index, G);

  printf("original bandwidth: %d\n", bandwidth(G));

  std::vector<Vertex> inv_perm(num_vertices(G));
  perm.resize(num_vertices(G));
  std::vector<int> order(perm.size());

    //reverse cuthill_mckee_ordering
    cuthill_mckee_ordering(G, inv_perm.rbegin(), get(vertex_color, G),
                           make_degree_map(G));

    for (int i=0; i < inv_perm.size(); i++) {
        order[i] = index_map[inv_perm[i]];
    }

    // reorder linear system
    std::vector<int> inv_order(order.size());
    for (int i=0; i < order.size(); i++) {
        inv_order[order[i]] = i;
    }

    std::vector<int> new_col_id(col_id.size(), 0);
    //std::fill(new_col_id.begin(), new_col_id.end(), 0); // should not be required
    //printf("size of new_col_id = %d\n", new_col_id.size());
    //for (int i=0; i < new_col_id.size(); i++) {
       // printf("1, %d, new_col: %d\n", i, new_col_id[i]);
    //}

    for (int i=0; i < nb_rows; i++) {
        for (int j=0; j < nb_nnz_per_row; j++) {
            int row_index = inv_order[i];
            int col_index = inv_order[col_id[j+i*nb_nnz_per_row]];
            new_col_id[j+row_index*nb_nnz_per_row] = col_index;
        }
    }

    for (int i=0; i < nb_rows; i++) {
        int bb = i*nb_nnz_per_row;
        int be = (i+1)*nb_nnz_per_row;
        std::sort(&new_col_id[bb], &new_col_id[be]);
        std::sort(&col_id[bb], &col_id[be]);
    }

    bw = getBandwidth(new_col_id, nb_rows, nb_nnz_per_row);
#endif
  exit(0);
}
//----------------------------------------------------------------------
};  // namespace gercm
