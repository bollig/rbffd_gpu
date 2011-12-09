/* =======================================================================
   Copyright (c) 2010, Institute for Microelectronics, TU Vienna.
   http://www.iue.tuwien.ac.at
                             -----------------
                     ViennaCL - The Vienna Computing Library
                             -----------------
                            
   authors:    Karl Rupp                          rupp@iue.tuwien.ac.at
               Florian Rudolf                     flo.rudy+viennacl@gmail.com
               Josef Weinbub                      weinbub@iue.tuwien.ac.at

   license:    MIT (X11), see file LICENSE in the ViennaCL base directory

   file changelog: - May 28, 2010   New from scratch for first release
======================================================================= */


/////////// direct solver kernels ////////////////
// lower:
template <typename TestData>
void matrix_trans_lower_triangular_substitute_inplace(TestData & data)
{
  viennacl::linalg::inplace_solve(trans(data.mat), data.v1, viennacl::linalg::lower_tag());
}

template <typename TestData>
void matrix_lower_triangular_substitute_inplace(TestData & data)
{
  viennacl::linalg::inplace_solve(data.mat, data.v1, viennacl::linalg::lower_tag());
}

template <typename TestData>
void matrix_unit_lower_triangular_substitute_inplace(TestData & data)
{
  viennacl::linalg::inplace_solve(data.mat, data.v1, viennacl::linalg::unit_lower_tag());
}

// upper:
template <typename TestData>
void matrix_upper_triangular_substitute_inplace(TestData & data)
{
  viennacl::linalg::inplace_solve(data.mat, data.v1, viennacl::linalg::upper_tag());
}

template <typename TestData>
void matrix_trans_upper_triangular_substitute_inplace(TestData & data)
{
  viennacl::linalg::inplace_solve(trans(data.mat), data.v1, viennacl::linalg::upper_tag());
}

template <typename TestData>
void matrix_unit_upper_triangular_substitute_inplace(TestData & data)
{
  viennacl::linalg::inplace_solve(data.mat, data.v1, viennacl::linalg::unit_upper_tag());
}


template <typename TestData>
void matrix_lu_factorize(TestData & data)
{
  viennacl::linalg::lu_factorize(data.mat);
}




//////////// other matrix operations: //////////////////
template <typename TestData>
void matrix_rank1_update(TestData & data)
{
  data.mat += viennacl::linalg::outer_prod(data.v1, data.v2);
}

template <typename TestData>
void matrix_scaled_rank1_update(TestData & data)
{
  typedef typename TestData::value_type   NumericT;
  data.mat += NumericT(2.0) * viennacl::linalg::outer_prod(data.v1, data.v2);
}

template <typename TestData>
void matrix_vec_mul(TestData & data)
{
  data.v2 = viennacl::linalg::prod(data.mat, data.v1);
}

template <typename TestData>
void matrix_trans_vec_mul(TestData & data)
{
  data.v2 = viennacl::linalg::prod(trans(data.mat), data.v1);
}
