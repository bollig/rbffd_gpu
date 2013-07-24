#ifndef VIENNACL_FFT_KERNELS_HPP_
#define VIENNACL_FFT_KERNELS_HPP_
#include "viennacl/tools/tools.hpp"
#include "viennacl/ocl/kernel.hpp"
#include "viennacl/ocl/platform.hpp"
#include "viennacl/ocl/utils.hpp"
#include "viennacl/linalg/kernels/fft_source.h"

//Automatically generated file from aux-directory, do not edit manually!
/** @file fft_kernels.h
 *  @brief OpenCL kernel file, generated automatically from scripts in auxiliary/. */
namespace viennacl
{
 namespace linalg
 {
  namespace kernels
  {
   template<class TYPE, unsigned int alignment>
   struct fft;


    /////////////// single precision kernels //////////////// 
   template <>
   struct fft<float, 1>
   {
    static std::string program_name()
    {
      return "f_fft_1";
    }
    static void init(viennacl::ocl::context & ctx)
    {
      viennacl::ocl::DOUBLE_PRECISION_CHECKER<float>::apply(ctx);
      static std::map<cl_context, bool> init_done;
      if (!init_done[ctx.handle().get()])
      {
        std::string source;
        source.reserve(8192);
        source.append(fft_align1_bluestein_post);
        source.append(fft_align1_bluestein_pre);
        source.append(fft_align1_complex_to_real);
        source.append(fft_align1_fft_div_vec_scalar);
        source.append(fft_align1_fft_mult_vec);
        source.append(fft_align1_real_to_complex);
        source.append(fft_align1_reverse_inplace);
        source.append(fft_align1_transpose);
        source.append(fft_align1_transpose_inplace);
        source.append(fft_align1_vandermonde_prod);
        source.append(fft_align1_zero2);
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
   struct fft<double, 1>
   {
    static std::string program_name()
    {
      return "d_fft_1";
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
        source.append(viennacl::tools::make_double_kernel(fft_align1_bluestein_post, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(fft_align1_bluestein_pre, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(fft_align1_complex_to_real, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(fft_align1_fft_div_vec_scalar, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(fft_align1_fft_mult_vec, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(fft_align1_real_to_complex, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(fft_align1_reverse_inplace, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(fft_align1_transpose, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(fft_align1_transpose_inplace, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(fft_align1_vandermonde_prod, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(fft_align1_zero2, fp64_ext));
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

