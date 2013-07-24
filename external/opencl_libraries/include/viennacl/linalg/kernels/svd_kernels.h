#ifndef VIENNACL_SVD_KERNELS_HPP_
#define VIENNACL_SVD_KERNELS_HPP_
#include "viennacl/tools/tools.hpp"
#include "viennacl/ocl/kernel.hpp"
#include "viennacl/ocl/platform.hpp"
#include "viennacl/ocl/utils.hpp"
#include "viennacl/linalg/kernels/svd_source.h"

//Automatically generated file from aux-directory, do not edit manually!
/** @file svd_kernels.h
 *  @brief OpenCL kernel file, generated automatically from scripts in auxiliary/. */
namespace viennacl
{
 namespace linalg
 {
  namespace kernels
  {
   template<class TYPE, unsigned int alignment>
   struct svd;


    /////////////// single precision kernels //////////////// 
   template <>
   struct svd<float, 1>
   {
    static std::string program_name()
    {
      return "f_svd_1";
    }
    static void init(viennacl::ocl::context & ctx)
    {
      viennacl::ocl::DOUBLE_PRECISION_CHECKER<float>::apply(ctx);
      static std::map<cl_context, bool> init_done;
      if (!init_done[ctx.handle().get()])
      {
        std::string source;
        source.reserve(8192);
        source.append(svd_align1_bidiag_pack);
        source.append(svd_align1_col_reduce_lcl_array);
        source.append(svd_align1_copy_col);
        source.append(svd_align1_copy_row);
        source.append(svd_align1_final_iter_update);
        source.append(svd_align1_givens_next);
        source.append(svd_align1_givens_prev);
        source.append(svd_align1_house_update_A_left);
        source.append(svd_align1_house_update_A_right);
        source.append(svd_align1_house_update_QL);
        source.append(svd_align1_house_update_QR);
        source.append(svd_align1_inverse_signs);
        source.append(svd_align1_transpose_inplace);
        source.append(svd_align1_update_qr_column);
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
   struct svd<double, 1>
   {
    static std::string program_name()
    {
      return "d_svd_1";
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
        source.append(viennacl::tools::make_double_kernel(svd_align1_bidiag_pack, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(svd_align1_col_reduce_lcl_array, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(svd_align1_copy_col, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(svd_align1_copy_row, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(svd_align1_final_iter_update, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(svd_align1_givens_next, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(svd_align1_givens_prev, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(svd_align1_house_update_A_left, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(svd_align1_house_update_A_right, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(svd_align1_house_update_QL, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(svd_align1_house_update_QR, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(svd_align1_inverse_signs, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(svd_align1_transpose_inplace, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(svd_align1_update_qr_column, fp64_ext));
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

