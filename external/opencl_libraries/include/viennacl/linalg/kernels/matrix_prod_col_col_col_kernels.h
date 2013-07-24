#ifndef VIENNACL_MATRIX_PROD_COL_COL_COL_KERNELS_HPP_
#define VIENNACL_MATRIX_PROD_COL_COL_COL_KERNELS_HPP_
#include "viennacl/tools/tools.hpp"
#include "viennacl/ocl/kernel.hpp"
#include "viennacl/ocl/platform.hpp"
#include "viennacl/ocl/utils.hpp"
#include "viennacl/linalg/kernels/matrix_prod_col_col_col_source.h"

//Automatically generated file from aux-directory, do not edit manually!
/** @file matrix_prod_col_col_col_kernels.h
 *  @brief OpenCL kernel file, generated automatically from scripts in auxiliary/. */
namespace viennacl
{
 namespace linalg
 {
  namespace kernels
  {
   template<class TYPE, unsigned int alignment>
   struct matrix_prod_col_col_col;


    /////////////// single precision kernels //////////////// 
   template <>
   struct matrix_prod_col_col_col<float, 1>
   {
    static std::string program_name()
    {
      return "f_matrix_prod_col_col_col_1";
    }
    static void init(viennacl::ocl::context & ctx)
    {
      viennacl::ocl::DOUBLE_PRECISION_CHECKER<float>::apply(ctx);
      static std::map<cl_context, bool> init_done;
      if (!init_done[ctx.handle().get()])
      {
        std::string source;
        source.reserve(8192);
        source.append(matrix_prod_col_col_col_align1_prod_AA);
        source.append(matrix_prod_col_col_col_align1_prod16_AA);
        if (ctx.current_device().local_mem_size() > 20000)
          source.append(matrix_prod_col_col_col_align1_prod_AA_amd);
        source.append(matrix_prod_col_col_col_align1_prod_AT);
        source.append(matrix_prod_col_col_col_align1_prod16_AT);
        if (ctx.current_device().local_mem_size() > 20000)
          source.append(matrix_prod_col_col_col_align1_prod_AT_amd);
        source.append(matrix_prod_col_col_col_align1_prod_TA);
        source.append(matrix_prod_col_col_col_align1_prod16_TA);
        if (ctx.current_device().local_mem_size() > 20000)
          source.append(matrix_prod_col_col_col_align1_prod_TA_amd);
        source.append(matrix_prod_col_col_col_align1_prod_TT);
        source.append(matrix_prod_col_col_col_align1_prod16_TT);
        if (ctx.current_device().local_mem_size() > 20000)
          source.append(matrix_prod_col_col_col_align1_prod_TT_amd);
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
   struct matrix_prod_col_col_col<double, 1>
   {
    static std::string program_name()
    {
      return "d_matrix_prod_col_col_col_1";
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
        source.append(viennacl::tools::make_double_kernel(matrix_prod_col_col_col_align1_prod_AA, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(matrix_prod_col_col_col_align1_prod16_AA, fp64_ext));
        if (ctx.current_device().local_mem_size() > 20000)
          source.append(viennacl::tools::make_double_kernel(matrix_prod_col_col_col_align1_prod_AA_amd, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(matrix_prod_col_col_col_align1_prod_AT, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(matrix_prod_col_col_col_align1_prod16_AT, fp64_ext));
        if (ctx.current_device().local_mem_size() > 20000)
          source.append(viennacl::tools::make_double_kernel(matrix_prod_col_col_col_align1_prod_AT_amd, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(matrix_prod_col_col_col_align1_prod_TA, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(matrix_prod_col_col_col_align1_prod16_TA, fp64_ext));
        if (ctx.current_device().local_mem_size() > 20000)
          source.append(viennacl::tools::make_double_kernel(matrix_prod_col_col_col_align1_prod_TA_amd, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(matrix_prod_col_col_col_align1_prod_TT, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(matrix_prod_col_col_col_align1_prod16_TT, fp64_ext));
        if (ctx.current_device().local_mem_size() > 20000)
          source.append(viennacl::tools::make_double_kernel(matrix_prod_col_col_col_align1_prod_TT_amd, fp64_ext));
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

