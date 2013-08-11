#include <iostream>
#include <vector>

#include "reorder.h"

using namespace std;


int main (int argc, char** argv) {

	std::vector<int> row_ptr; 
	std::vector<int> col_ind; 
	
	// 1 0 0 0 1 0 0 0
	row_ptr.push_back(0); 
	// 0 1 1 0 0 1 0 1
	row_ptr.push_back(2); 
	// 0 1 1 0 1 0 0 0
	row_ptr.push_back(6); 
	// 0 0 0 1 0 0 1 0
	row_ptr.push_back(9); 
	// 1 0 1 0 1 0 0 0
	row_ptr.push_back(12); 
	// 0 0 0 1 0 0 1 0
	row_ptr.push_back(14); 
	// 0 1 0 0 0 1 0 1
	row_ptr.push_back(17); 


	col_ind.push_back(0); 
	col_ind.push_back(4); 

	col_ind.push_back(1); 
	col_ind.push_back(2); 
	col_ind.push_back(5); 
	col_ind.push_back(7); 

	col_ind.push_back(1); 
	col_ind.push_back(2); 
	col_ind.push_back(4); 

	col_ind.push_back(3); 
	col_ind.push_back(6); 

	col_ind.push_back(0); 
	col_ind.push_back(2); 
	col_ind.push_back(4); 

	col_ind.push_back(1); 
	col_ind.push_back(5); 
	col_ind.push_back(7); 

	col_ind.push_back(3); 
	col_ind.push_back(6); 

	col_ind.push_back(1); 
	col_ind.push_back(5); 
	col_ind.push_back(7); 


	std::vector<double> values(col_ind.size(),1);
typedef boost::numeric::ublas::compressed_matrix<double> MatType; 
typedef boost::numeric::ublas::vector<double> VecType;

	typedef MatType CSR_Mat; 
	typedef VecType UBLAS_Vec; 


	CSR_Mat A(8, 8, values.size()); 
	CSR_Mat A_reordered(8, 8, values.size()); 
	UBLAS_Vec V(8); 
	UBLAS_Vec V_reordered(8); 

	// Convert to Boost CSR 
	int k = 0;
	for (int i = 0; i < 8; i ++ ){ 
		V(i) = i; 
		for (int j = 0; j < 8; j++) { 
			if (col_ind[k] == j) {
				A(i,j) = values[k]; 
				k++;
			}
		}
	}	

	AdjacencyGraph agraph; 

	std::vector<int> new_order(8); 

	//std::cout << A_reordered << std::endl;

	Reorder< CSR_Mat, UBLAS_Vec >  r; 
	r.build_graph(A, agraph); 
	r.get_cuthill_mckee_order(agraph, new_order); 
	r.get_reordered_system(A, V, new_order, A_reordered, V_reordered);

	std::cout << "INPUT:\n";
	for (int i = 0; i < 8; i ++ ){ 
		for (int j = 0; j < 8; j++) { 
			std::cout << A(i,j) << " ";
		}
		std::cout << "\n";
	}

	
	std::cout << "\n\nOUTPUT:\n";

	for (int i = 0; i < 8; i ++ ){ 
		for (int j = 0; j < 8; j++) { 
			std::cout << A_reordered(i,j) << " ";
		}
		std::cout << "\n";
	}

	return 0; 

}
