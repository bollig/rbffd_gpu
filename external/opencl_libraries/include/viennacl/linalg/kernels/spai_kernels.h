#ifndef VIENNACL_SPAI_KERNELS_HPP_
#define VIENNACL_SPAI_KERNELS_HPP_
#include "viennacl/tools/tools.hpp"
#include "viennacl/ocl/kernel.hpp"
#include "viennacl/ocl/platform.hpp"
#include "viennacl/ocl/utils.hpp"
#include "viennacl/linalg/kernels/spai_source.h"

//Automatically generated file from aux-directory, do not edit manually!
/** @file spai_kernels.h
 *  @brief OpenCL kernel file, generated automatically from scripts in auxiliary/. */
namespace viennacl
{
 namespace linalg
 {
  namespace kernels
  {
   template<class TYPE, unsigned int alignment>
   struct spai;


    /////////////// single precision kernels //////////////// 
   template <>
   struct spai<float, 1>
   {
    static std::string program_name()
    {
      return "f_spai_1";
    }
    static void init(viennacl::ocl::context & ctx)
    {
      viennacl::ocl::DOUBLE_PRECISION_CHECKER<float>::apply(ctx);
      static std::map<cl_context, bool> init_done;
      if (!init_done[ctx.handle().get()])
      {
        std::string source;
        source.reserve(8192);
        source.append(spai_align1_assemble_blocks);
        source.append(spai_align1_block_bv_assembly);
        source.append(spai_align1_block_least_squares);
        source.append(spai_align1_block_q_mult);
        source.append(spai_align1_block_qr);
        source.append(spai_align1_block_qr_assembly);
        source.append(spai_align1_block_qr_assembly_1);
        source.append(spai_align1_block_r_assembly);
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
   struct spai<double, 1>
   {
    static std::string program_name()
    {
      return "d_spai_1";
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
        source.append(viennacl::tools::make_double_kernel(spai_align1_assemble_blocks, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(spai_align1_block_bv_assembly, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(spai_align1_block_least_squares, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(spai_align1_block_q_mult, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(spai_align1_block_qr, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(spai_align1_block_qr_assembly, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(spai_align1_block_qr_assembly_1, fp64_ext));
        source.append(viennacl::tools::make_double_kernel(spai_align1_block_r_assembly, fp64_ext));
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

