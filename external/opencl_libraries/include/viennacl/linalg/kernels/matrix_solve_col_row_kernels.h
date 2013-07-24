#ifndef VIENNACL_MATRIX_SOLVE_COL_ROW_KERNELS_HPP_
#define VIENNACL_MATRIX_SOLVE_COL_ROW_KERNELS_HPP_
#include "viennacl/tools/tools.hpp"
#include "viennacl/ocl/kernel.hpp"
#include "viennacl/ocl/platform.hpp"
#include "viennacl/ocl/utils.hpp"
#include "viennacl/linalg/kernels/matrix_solve_col_row_source.h"

//Automatically generated file from aux-directory, do not edit manually!
/** @file matrix_solve_col_row_kernels.h
 *  @brief OpenCL kernel file, generated automatically from scripts in auxiliary/. */
namespace viennacl
{
 namespace linalg
 {
  namespace kernels
  {
   template<class TYPE, unsigned int alignment>
   struct matrix_solve_col_row;


    /////////////// single precision kernels //////////////// 
   template <>
   struct matrix_solve_col_row<float, 1>
   {
    static std::string program_name()
    {
      return "f_matrix_solve_col_row_1";
    }
    static void init(viennacl::ocl::context & ctx)
    {
      viennacl::ocl::DOUBLE_PRECISION_CHECKER<float>::apply(ctx);
      static std::map<cl_context, bool> init_done;
      if (!init_done[ctx.handle().get()])
      {
        std::string source;
        source.reserve(8192);
        source.append(matrix_solve_col_row_align1_lower_solve);
        source.append(matrix_solve_col_row_align1_unit_lower_solve);
        source.append(matrix_solve_col_row_align1_upper_solve);
        source.append(matrix_solve_col_row_align1_unit_upper_solve);
        source.append(matrix_solve_col_row_align1_lower_trans_solve);
        source.append(matrix_solve_col_row_align1_unit_lower_trans_solve);
        source.append(matrix_solve_col_row_align1_upper_trans_solve);
        source.append(matrix_solve_col_row_align1_unit_upper_trans_solve);
        source.append(matrix_solve_col_row_align1_trans_lower_solve);
        source.append(matrix_solve_col_row_align1_trans_unit_lower_solve);
        source.append(matrix_solve_col_row_align1_trans_upper_solve);
        source.append(matrix_solve_col_row_align1_trans_unit_upper_solve);
        source.append(matrix_solve_col_row_align1_trans_lower_trans_solve);
        source.append(matrix_solve_col_row_align1_trans_unit_lower_trans_solve);
        source.append(matrix_solve_col_row_align1_trans_upper_trans_solve);
        source.append(matrix_solve_col_row_align1_trans_unit_upper_trans_solve);
        std::string prog_name = program_name();
        #ifdef VIENNACL_BUILD_INFO
        std::cout << "Creating program " << prog_name << std::endl;
        #endif
        ctx.add_program(source, prog_name);
        init_done[ctx.handle().get()] = true;
       } //if
     } //init
    }; // struct



    /////////////// double precision kernels //////////////// 
   template <>
   struct matrix_solve_col_row<double, 1>
   {
    static std::string program_name()
    {
      return "d_matrix_solve_col_row_1";
    }
    static void init(viennacl::ocl::context & ctx)
    {
      viennacl::ocl::DOUBLE_PRECISION_CHECKER<double>::apply(ctx);
      static std::map<cl_context, bool> init_done;
      if (!init_done[ctx.handle().get()])
      {
        std::string source;
        source.reserve(8192);
        std::string fp64_ext = ctx.current_device().double_support_extension();
        source.append(viennacl::tools::make_double_kernel(matrix_solve_col_row_align1_lower_solve, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(matrix_solve_col_row_align1_unit_lower_solve, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(matrix_solve_col_row_align1_upper_solve, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(matrix_solve_col_row_align1_unit_upper_solve, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(matrix_solve_col_row_align1_lower_trans_solve, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(matrix_solve_col_row_align1_unit_lower_trans_solve, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(matrix_solve_col_row_align1_upper_trans_solve, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(matrix_solve_col_row_align1_unit_upper_trans_solve, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(matrix_solve_col_row_align1_trans_lower_solve, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(matrix_solve_col_row_align1_trans_unit_lower_solve, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(matrix_solve_col_row_align1_trans_upper_solve, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(matrix_solve_col_row_align1_trans_unit_upper_solve, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(matrix_solve_col_row_align1_trans_lower_trans_solve, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(matrix_solve_col_row_align1_trans_unit_lower_trans_solve, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(matrix_solve_col_row_align1_trans_upper_trans_solve, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(matrix_solve_col_row_align1_trans_unit_upper_trans_solve, fp64_ext));
        std::string prog_name = program_name();
        #ifdef VIENNACL_BUILD_INFO
        std::cout << "Creating program " << prog_name << std::endl;
        #endif
        ctx.add_program(source, prog_name);
        init_done[ctx.handle().get()] = true;
       } //if
     } //init
    }; // struct


  }  //namespace kernels
 }  //namespace linalg
}  //namespace viennacl
#endif

