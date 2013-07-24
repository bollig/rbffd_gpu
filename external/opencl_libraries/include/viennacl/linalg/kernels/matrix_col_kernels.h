#ifndef VIENNACL_MATRIX_COL_KERNELS_HPP_
#define VIENNACL_MATRIX_COL_KERNELS_HPP_
#include "viennacl/tools/tools.hpp"
#include "viennacl/ocl/kernel.hpp"
#include "viennacl/ocl/platform.hpp"
#include "viennacl/ocl/utils.hpp"
#include "viennacl/linalg/kernels/matrix_col_source.h"

//Automatically generated file from aux-directory, do not edit manually!
/** @file matrix_col_kernels.h
 *  @brief OpenCL kernel file, generated automatically from scripts in auxiliary/. */
namespace viennacl
{
 namespace linalg
 {
  namespace kernels
  {
   template<class TYPE, unsigned int alignment>
   struct matrix_col;


    /////////////// single precision kernels //////////////// 
   template <>
   struct matrix_col<float, 1>
   {
    static std::string program_name()
    {
      return "f_matrix_col_1";
    }
    static void init(viennacl::ocl::context & ctx)
    {
      viennacl::ocl::DOUBLE_PRECISION_CHECKER<float>::apply(ctx);
      static std::map<cl_context, bool> init_done;
      if (!init_done[ctx.handle().get()])
      {
        std::string source;
        source.reserve(8192);
        source.append(matrix_col_align1_am_cpu);
        source.append(matrix_col_align1_am_gpu);
        source.append(matrix_col_align1_ambm_cpu_cpu);
        source.append(matrix_col_align1_ambm_cpu_gpu);
        source.append(matrix_col_align1_ambm_gpu_cpu);
        source.append(matrix_col_align1_ambm_gpu_gpu);
        source.append(matrix_col_align1_ambm_m_cpu_cpu);
        source.append(matrix_col_align1_ambm_m_cpu_gpu);
        source.append(matrix_col_align1_ambm_m_gpu_cpu);
        source.append(matrix_col_align1_ambm_m_gpu_gpu);
        source.append(matrix_col_align1_assign_cpu);
        source.append(matrix_col_align1_diagonal_assign_cpu);
        source.append(matrix_col_align1_element_op);
        source.append(matrix_col_align1_fft_direct);
        source.append(matrix_col_align1_fft_radix2);
        source.append(matrix_col_align1_fft_radix2_local);
        source.append(matrix_col_align1_fft_reorder);
        source.append(matrix_col_align1_triangular_substitute_inplace);
        source.append(matrix_col_align1_lu_factorize);
        source.append(matrix_col_align1_scaled_rank1_update_cpu);
        source.append(matrix_col_align1_scaled_rank1_update_gpu);
        source.append(matrix_col_align1_trans_vec_mul);
        source.append(matrix_col_align1_vec_mul);
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
   struct matrix_col<float, 16>
   {
    static std::string program_name()
    {
      return "f_matrix_col_16";
    }
    static void init(viennacl::ocl::context & ctx)
    {
      viennacl::ocl::DOUBLE_PRECISION_CHECKER<float>::apply(ctx);
      static std::map<cl_context, bool> init_done;
      if (!init_done[ctx.handle().get()])
      {
        std::string source;
        source.reserve(8192);
        source.append(matrix_col_align1_am_cpu);
        source.append(matrix_col_align1_am_gpu);
        source.append(matrix_col_align1_ambm_cpu_cpu);
        source.append(matrix_col_align1_ambm_cpu_gpu);
        source.append(matrix_col_align1_ambm_gpu_cpu);
        source.append(matrix_col_align1_ambm_gpu_gpu);
        source.append(matrix_col_align1_ambm_m_cpu_cpu);
        source.append(matrix_col_align1_ambm_m_cpu_gpu);
        source.append(matrix_col_align1_ambm_m_gpu_cpu);
        source.append(matrix_col_align1_ambm_m_gpu_gpu);
        source.append(matrix_col_align1_assign_cpu);
        source.append(matrix_col_align1_diagonal_assign_cpu);
        source.append(matrix_col_align1_element_op);
        source.append(matrix_col_align1_fft_direct);
        source.append(matrix_col_align1_fft_radix2);
        source.append(matrix_col_align1_fft_radix2_local);
        source.append(matrix_col_align1_fft_reorder);
        source.append(matrix_col_align1_triangular_substitute_inplace);
        source.append(matrix_col_align1_lu_factorize);
        source.append(matrix_col_align1_scaled_rank1_update_cpu);
        source.append(matrix_col_align1_scaled_rank1_update_gpu);
        source.append(matrix_col_align1_trans_vec_mul);
        source.append(matrix_col_align1_vec_mul);
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
   struct matrix_col<double, 1>
   {
    static std::string program_name()
    {
      return "d_matrix_col_1";
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
        source.append(viennacl::tools::make_double_kernel(matrix_col_align1_am_cpu, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(matrix_col_align1_am_gpu, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(matrix_col_align1_ambm_cpu_cpu, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(matrix_col_align1_ambm_cpu_gpu, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(matrix_col_align1_ambm_gpu_cpu, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(matrix_col_align1_ambm_gpu_gpu, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(matrix_col_align1_ambm_m_cpu_cpu, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(matrix_col_align1_ambm_m_cpu_gpu, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(matrix_col_align1_ambm_m_gpu_cpu, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(matrix_col_align1_ambm_m_gpu_gpu, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(matrix_col_align1_assign_cpu, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(matrix_col_align1_diagonal_assign_cpu, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(matrix_col_align1_element_op, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(matrix_col_align1_fft_direct, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(matrix_col_align1_fft_radix2, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(matrix_col_align1_fft_radix2_local, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(matrix_col_align1_fft_reorder, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(matrix_col_align1_triangular_substitute_inplace, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(matrix_col_align1_lu_factorize, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(matrix_col_align1_scaled_rank1_update_cpu, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(matrix_col_align1_scaled_rank1_update_gpu, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(matrix_col_align1_trans_vec_mul, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(matrix_col_align1_vec_mul, fp64_ext));
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
   struct matrix_col<double, 16>
   {
    static std::string program_name()
    {
      return "d_matrix_col_16";
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
        source.append(viennacl::tools::make_double_kernel(matrix_col_align1_am_cpu, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(matrix_col_align1_am_gpu, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(matrix_col_align1_ambm_cpu_cpu, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(matrix_col_align1_ambm_cpu_gpu, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(matrix_col_align1_ambm_gpu_cpu, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(matrix_col_align1_ambm_gpu_gpu, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(matrix_col_align1_ambm_m_cpu_cpu, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(matrix_col_align1_ambm_m_cpu_gpu, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(matrix_col_align1_ambm_m_gpu_cpu, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(matrix_col_align1_ambm_m_gpu_gpu, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(matrix_col_align1_assign_cpu, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(matrix_col_align1_diagonal_assign_cpu, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(matrix_col_align1_element_op, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(matrix_col_align1_fft_direct, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(matrix_col_align1_fft_radix2, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(matrix_col_align1_fft_radix2_local, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(matrix_col_align1_fft_reorder, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(matrix_col_align1_triangular_substitute_inplace, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(matrix_col_align1_lu_factorize, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(matrix_col_align1_scaled_rank1_update_cpu, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(matrix_col_align1_scaled_rank1_update_gpu, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(matrix_col_align1_trans_vec_mul, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(matrix_col_align1_vec_mul, fp64_ext));
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

