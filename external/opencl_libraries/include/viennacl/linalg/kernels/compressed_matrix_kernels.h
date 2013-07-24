#ifndef VIENNACL_COMPRESSED_MATRIX_KERNELS_HPP_
#define VIENNACL_COMPRESSED_MATRIX_KERNELS_HPP_
#include "viennacl/tools/tools.hpp"
#include "viennacl/ocl/kernel.hpp"
#include "viennacl/ocl/platform.hpp"
#include "viennacl/ocl/utils.hpp"
#include "viennacl/linalg/kernels/compressed_matrix_source.h"

//Automatically generated file from aux-directory, do not edit manually!
/** @file compressed_matrix_kernels.h
 *  @brief OpenCL kernel file, generated automatically from scripts in auxiliary/. */
namespace viennacl
{
 namespace linalg
 {
  namespace kernels
  {
   template<class TYPE, unsigned int alignment>
   struct compressed_matrix;


    /////////////// single precision kernels //////////////// 
   template <>
   struct compressed_matrix<float, 1>
   {
    static std::string program_name()
    {
      return "f_compressed_matrix_1";
    }
    static void init(viennacl::ocl::context & ctx)
    {
      viennacl::ocl::DOUBLE_PRECISION_CHECKER<float>::apply(ctx);
      static std::map<cl_context, bool> init_done;
      if (!init_done[ctx.handle().get()])
      {
        std::string source;
        source.reserve(8192);
        source.append(compressed_matrix_align1_block_trans_lu_backward);
        source.append(compressed_matrix_align1_block_trans_unit_lu_forward);
        source.append(compressed_matrix_align1_jacobi);
        source.append(compressed_matrix_align1_lu_backward);
        source.append(compressed_matrix_align1_lu_forward);
        source.append(compressed_matrix_align1_trans_lu_backward);
        source.append(compressed_matrix_align1_trans_lu_forward);
        source.append(compressed_matrix_align1_trans_unit_lu_backward);
        source.append(compressed_matrix_align1_trans_unit_lu_forward);
        source.append(compressed_matrix_align1_trans_unit_lu_forward_slow);
        source.append(compressed_matrix_align1_row_info_extractor);
        source.append(compressed_matrix_align1_unit_lu_backward);
        source.append(compressed_matrix_align1_unit_lu_forward);
        source.append(compressed_matrix_align1_vec_mul);
        source.append(compressed_matrix_align1_vec_mul_cpu);
        std::string prog_name = program_name();
        #ifdef VIENNACL_BUILD_INFO
        std::cout << "Creating program " << prog_name << std::endl;
        #endif
        ctx.add_program(source, prog_name);
        init_done[ctx.handle().get()] = true;
       } //if
     } //init
    }; // struct

   template <>
   struct compressed_matrix<float, 4>
   {
    static std::string program_name()
    {
      return "f_compressed_matrix_4";
    }
    static void init(viennacl::ocl::context & ctx)
    {
      viennacl::ocl::DOUBLE_PRECISION_CHECKER<float>::apply(ctx);
      static std::map<cl_context, bool> init_done;
      if (!init_done[ctx.handle().get()])
      {
        std::string source;
        source.reserve(8192);
        source.append(compressed_matrix_align1_block_trans_lu_backward);
        source.append(compressed_matrix_align1_block_trans_unit_lu_forward);
        source.append(compressed_matrix_align1_jacobi);
        source.append(compressed_matrix_align1_lu_backward);
        source.append(compressed_matrix_align1_lu_forward);
        source.append(compressed_matrix_align1_trans_lu_backward);
        source.append(compressed_matrix_align1_trans_lu_forward);
        source.append(compressed_matrix_align1_trans_unit_lu_backward);
        source.append(compressed_matrix_align1_trans_unit_lu_forward);
        source.append(compressed_matrix_align1_trans_unit_lu_forward_slow);
        source.append(compressed_matrix_align1_row_info_extractor);
        source.append(compressed_matrix_align1_unit_lu_backward);
        source.append(compressed_matrix_align1_unit_lu_forward);
        source.append(compressed_matrix_align4_vec_mul);
        source.append(compressed_matrix_align1_vec_mul_cpu);
        std::string prog_name = program_name();
        #ifdef VIENNACL_BUILD_INFO
        std::cout << "Creating program " << prog_name << std::endl;
        #endif
        ctx.add_program(source, prog_name);
        init_done[ctx.handle().get()] = true;
       } //if
     } //init
    }; // struct

   template <>
   struct compressed_matrix<float, 8>
   {
    static std::string program_name()
    {
      return "f_compressed_matrix_8";
    }
    static void init(viennacl::ocl::context & ctx)
    {
      viennacl::ocl::DOUBLE_PRECISION_CHECKER<float>::apply(ctx);
      static std::map<cl_context, bool> init_done;
      if (!init_done[ctx.handle().get()])
      {
        std::string source;
        source.reserve(8192);
        source.append(compressed_matrix_align1_block_trans_lu_backward);
        source.append(compressed_matrix_align1_block_trans_unit_lu_forward);
        source.append(compressed_matrix_align1_jacobi);
        source.append(compressed_matrix_align1_lu_backward);
        source.append(compressed_matrix_align1_lu_forward);
        source.append(compressed_matrix_align1_trans_lu_backward);
        source.append(compressed_matrix_align1_trans_lu_forward);
        source.append(compressed_matrix_align1_trans_unit_lu_backward);
        source.append(compressed_matrix_align1_trans_unit_lu_forward);
        source.append(compressed_matrix_align1_trans_unit_lu_forward_slow);
        source.append(compressed_matrix_align1_row_info_extractor);
        source.append(compressed_matrix_align1_unit_lu_backward);
        source.append(compressed_matrix_align1_unit_lu_forward);
        source.append(compressed_matrix_align8_vec_mul);
        source.append(compressed_matrix_align1_vec_mul_cpu);
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
   struct compressed_matrix<double, 1>
   {
    static std::string program_name()
    {
      return "d_compressed_matrix_1";
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
        source.append(viennacl::tools::make_double_kernel(compressed_matrix_align1_block_trans_lu_backward, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(compressed_matrix_align1_block_trans_unit_lu_forward, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(compressed_matrix_align1_jacobi, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(compressed_matrix_align1_lu_backward, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(compressed_matrix_align1_lu_forward, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(compressed_matrix_align1_trans_lu_backward, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(compressed_matrix_align1_trans_lu_forward, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(compressed_matrix_align1_trans_unit_lu_backward, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(compressed_matrix_align1_trans_unit_lu_forward, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(compressed_matrix_align1_trans_unit_lu_forward_slow, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(compressed_matrix_align1_row_info_extractor, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(compressed_matrix_align1_unit_lu_backward, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(compressed_matrix_align1_unit_lu_forward, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(compressed_matrix_align1_vec_mul, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(compressed_matrix_align1_vec_mul_cpu, fp64_ext));
        std::string prog_name = program_name();
        #ifdef VIENNACL_BUILD_INFO
        std::cout << "Creating program " << prog_name << std::endl;
        #endif
        ctx.add_program(source, prog_name);
        init_done[ctx.handle().get()] = true;
       } //if
     } //init
    }; // struct

   template <>
   struct compressed_matrix<double, 4>
   {
    static std::string program_name()
    {
      return "d_compressed_matrix_4";
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
        source.append(viennacl::tools::make_double_kernel(compressed_matrix_align1_block_trans_lu_backward, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(compressed_matrix_align1_block_trans_unit_lu_forward, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(compressed_matrix_align1_jacobi, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(compressed_matrix_align1_lu_backward, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(compressed_matrix_align1_lu_forward, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(compressed_matrix_align1_trans_lu_backward, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(compressed_matrix_align1_trans_lu_forward, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(compressed_matrix_align1_trans_unit_lu_backward, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(compressed_matrix_align1_trans_unit_lu_forward, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(compressed_matrix_align1_trans_unit_lu_forward_slow, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(compressed_matrix_align1_row_info_extractor, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(compressed_matrix_align1_unit_lu_backward, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(compressed_matrix_align1_unit_lu_forward, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(compressed_matrix_align4_vec_mul, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(compressed_matrix_align1_vec_mul_cpu, fp64_ext));
        std::string prog_name = program_name();
        #ifdef VIENNACL_BUILD_INFO
        std::cout << "Creating program " << prog_name << std::endl;
        #endif
        ctx.add_program(source, prog_name);
        init_done[ctx.handle().get()] = true;
       } //if
     } //init
    }; // struct

   template <>
   struct compressed_matrix<double, 8>
   {
    static std::string program_name()
    {
      return "d_compressed_matrix_8";
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
        source.append(viennacl::tools::make_double_kernel(compressed_matrix_align1_block_trans_lu_backward, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(compressed_matrix_align1_block_trans_unit_lu_forward, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(compressed_matrix_align1_jacobi, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(compressed_matrix_align1_lu_backward, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(compressed_matrix_align1_lu_forward, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(compressed_matrix_align1_trans_lu_backward, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(compressed_matrix_align1_trans_lu_forward, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(compressed_matrix_align1_trans_unit_lu_backward, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(compressed_matrix_align1_trans_unit_lu_forward, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(compressed_matrix_align1_trans_unit_lu_forward_slow, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(compressed_matrix_align1_row_info_extractor, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(compressed_matrix_align1_unit_lu_backward, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(compressed_matrix_align1_unit_lu_forward, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(compressed_matrix_align8_vec_mul, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(compressed_matrix_align1_vec_mul_cpu, fp64_ext));
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

