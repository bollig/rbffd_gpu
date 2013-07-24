#ifndef VIENNACL_SCALAR_KERNELS_HPP_
#define VIENNACL_SCALAR_KERNELS_HPP_
#include "viennacl/tools/tools.hpp"
#include "viennacl/ocl/kernel.hpp"
#include "viennacl/ocl/platform.hpp"
#include "viennacl/ocl/utils.hpp"
#include "viennacl/linalg/kernels/scalar_source.h"

//Automatically generated file from aux-directory, do not edit manually!
/** @file scalar_kernels.h
 *  @brief OpenCL kernel file, generated automatically from scripts in auxiliary/. */
namespace viennacl
{
 namespace linalg
 {
  namespace kernels
  {
   template<class TYPE, unsigned int alignment>
   struct scalar;


    /////////////// single precision kernels //////////////// 
   template <>
   struct scalar<float, 1>
   {
    static std::string program_name()
    {
      return "f_scalar_1";
    }
    static void init(viennacl::ocl::context & ctx)
    {
      viennacl::ocl::DOUBLE_PRECISION_CHECKER<float>::apply(ctx);
      static std::map<cl_context, bool> init_done;
      if (!init_done[ctx.handle().get()])
      {
        std::string source;
        source.reserve(8192);
        source.append(scalar_align1_as_cpu);
        source.append(scalar_align1_as_gpu);
        source.append(scalar_align1_asbs_cpu_cpu);
        source.append(scalar_align1_asbs_cpu_gpu);
        source.append(scalar_align1_asbs_gpu_cpu);
        source.append(scalar_align1_asbs_gpu_gpu);
        source.append(scalar_align1_asbs_s_cpu_cpu);
        source.append(scalar_align1_asbs_s_cpu_gpu);
        source.append(scalar_align1_asbs_s_gpu_cpu);
        source.append(scalar_align1_asbs_s_gpu_gpu);
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
   struct scalar<double, 1>
   {
    static std::string program_name()
    {
      return "d_scalar_1";
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
        source.append(viennacl::tools::make_double_kernel(scalar_align1_as_cpu, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(scalar_align1_as_gpu, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(scalar_align1_asbs_cpu_cpu, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(scalar_align1_asbs_cpu_gpu, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(scalar_align1_asbs_gpu_cpu, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(scalar_align1_asbs_gpu_gpu, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(scalar_align1_asbs_s_cpu_cpu, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(scalar_align1_asbs_s_cpu_gpu, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(scalar_align1_asbs_s_gpu_cpu, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(scalar_align1_asbs_s_gpu_gpu, fp64_ext));
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

