#include "vcl_bandwidth_reduction.h"
//----------------------------------------------------------------------
//
// Part 2: Tutorial code
//


int vcl_main(int, char **)
{
    vcl::RCM_VCL vicl;

  srand(42);
  std::cout << "-- Generating matrix --" << std::endl;
  std::size_t dof_per_dim = 96;   //number of grid points per coordinate direction
  std::size_t n = dof_per_dim * dof_per_dim * dof_per_dim; //total number of unknowns
  std::vector< std::map<int, double> > matrix = vicl.gen_3d_mesh_matrix(dof_per_dim, dof_per_dim, dof_per_dim, false);  //If last parameter is 'true', a tetrahedral grid instead of a hexahedral grid is used.
  
  //
  // Shuffle the generated matrix
  //
  std::vector<int> r = vicl.generate_random_reordering(n);
  std::vector< std::map<int, double> > matrix2 = vicl.reorder_matrix(matrix, r);
  
  
  //
  // Print some statistics:
  //
  std::cout << " * Unknowns: " << n << std::endl;
  std::cout << " * Initial bandwidth: " << vicl.calc_bw(matrix) << std::endl;
  std::cout << " * Randomly reordered bandwidth: " << vicl.calc_bw(matrix2) << std::endl;

  //
  // Reorder using Cuthill-McKee algorithm
  //
  std::cout << "-- Cuthill-McKee algorithm --" << std::endl;
  r = viennacl::reorder(matrix2, viennacl::cuthill_mckee_tag());
  std::cout << " * Reordered bandwidth: " << vicl.calc_reordered_bw(matrix2, r) << std::endl;
  
  //
  // Reorder using advanced Cuthill-McKee algorithm
  //
  std::cout << "-- Advanced Cuthill-McKee algorithm --" << std::endl;
  double a = 0.0;
  std::size_t gmax = 1;
  r = viennacl::reorder(matrix2, viennacl::advanced_cuthill_mckee_tag(a, gmax));
  std::cout << " * Reordered bandwidth: " << vicl.calc_reordered_bw(matrix2, r) << std::endl;
  
  //
  // Reorder using Gibbs-Poole-Stockmeyer algorithm
  //
  //std::cout << "-- Gibbs-Poole-Stockmeyer algorithm --" << std::endl;
  //r = viennacl::reorder(matrix2, viennacl::gibbs_poole_stockmeyer_tag());
  //std::cout << " * Reordered bandwidth: " << calc_reordered_bw(matrix2, r) << std::endl;
    
  //
  //  That's it.
  //
  std::cout << "!!!! TUTORIAL COMPLETED SUCCESSFULLY !!!!" << std::endl;
    
  return EXIT_SUCCESS;
}

