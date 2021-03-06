#ifndef VIENNACL_OCL_ENQUEUE_HPP_
#define VIENNACL_OCL_ENQUEUE_HPP_

/* =========================================================================
   Copyright (c) 2010-2013, Institute for Microelectronics,
                            Institute for Analysis and Scientific Computing,
                            TU Wien.
   Portions of this software are copyright by UChicago Argonne, LLC.

                            -----------------
                  ViennaCL - The Vienna Computing Library
                            -----------------

   Project Head:    Karl Rupp                   rupp@iue.tuwien.ac.at

   (A list of authors and contributors can be found in the PDF manual)

   License:         MIT (X11), see file LICENSE in the base directory
============================================================================= */

/** @file viennacl/ocl/enqueue.hpp
    @brief Enqueues kernels into command queues
*/

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#include "viennacl/ocl/kernel.hpp"
#include "viennacl/ocl/command_queue.hpp"

namespace viennacl
{
  namespace generator{
      class custom_operation;
      void enqueue_custom_op(viennacl::generator::custom_operation & op, viennacl::ocl::command_queue const & queue);
  }

  namespace ocl
  {

    /** @brief Enqueues a kernel in the provided queue */
    template <typename KernelType>
    void enqueue(KernelType & k, viennacl::ocl::command_queue const & queue)
    {
      // 1D kernel:
      if (k.local_work_size(1) == 0)
      {
        #if defined(VIENNACL_DEBUG_ALL) || defined(VIENNACL_DEBUG_KERNEL)
        std::cout << "ViennaCL: Starting 1D-kernel '" << k.name() << "'..." << std::endl;
        std::cout << "ViennaCL: Global work size: '"  << k.global_work_size() << "'..." << std::endl;
        std::cout << "ViennaCL: Local work size: '"   << k.local_work_size() << "'..." << std::endl;
        #endif

        std::size_t tmp_global = k.global_work_size();
        std::size_t tmp_local = k.local_work_size();

        cl_int err;
        if (tmp_global == 1 && tmp_local == 1)
          err = clEnqueueTask(queue.handle().get(), k.handle().get(), 0, NULL, NULL);
        else
          err = clEnqueueNDRangeKernel(queue.handle().get(), k.handle().get(), 1, NULL, &tmp_global, &tmp_local, 0, NULL, NULL);

        if (err != CL_SUCCESS)  //if not successful, try to start with smaller work size
        {
          //std::cout << "FAIL: " << std::endl; exit(0);
          while (err != CL_SUCCESS && tmp_local > 1)
          {
            //std::cout << "Flushing queue, then enqueuing again with half the size..." << std::endl;
            //std::cout << "Error code: " << err << std::endl;

            tmp_global /= 2;
            tmp_local /= 2;

            #if defined(VIENNACL_DEBUG_ALL) || defined(VIENNACL_DEBUG_KERNEL)
            std::cout << "ViennaCL: Kernel start failed for '" << k.name() << "'." << std::endl;
            std::cout << "ViennaCL: Global work size: '"  << tmp_global << "'..." << std::endl;
            std::cout << "ViennaCL: Local work size: '"   << tmp_local << "'..." << std::endl;
            #endif

            queue.finish();
            err = clEnqueueNDRangeKernel(queue.handle().get(), k.handle().get(), 1, NULL, &tmp_global, &tmp_local, 0, NULL, NULL);
          }

          if (err != CL_SUCCESS)
          {
            //could not start kernel with any parameters
            std::cerr << "ViennaCL: FATAL ERROR: Kernel start failed for '" << k.name() << "'." << std::endl;
            std::cerr << "ViennaCL: Smaller work sizes could not solve the problem. " << std::endl;
            VIENNACL_ERR_CHECK(err);
          }
          else
          {
            //remember parameters:
            k.local_work_size(0, tmp_local);
            k.global_work_size(0, tmp_global);
            #if defined(VIENNACL_DEBUG_ALL) || defined(VIENNACL_DEBUG_KERNEL)
            std::cout << "ViennaCL: Kernel '" << k.name() << "' now uses global work size " << tmp_global << " and local work size " << tmp_local << "."  << std::endl;
            #endif
          }
        }
      }
      else //2D or 3D kernel
      {
        #if defined(VIENNACL_DEBUG_ALL) || defined(VIENNACL_DEBUG_KERNEL)
        std::cout << "ViennaCL: Starting 2D/3D-kernel '" << k.name() << "'..." << std::endl;
        std::cout << "ViennaCL: Global work size: '"  << k.global_work_size(0) << ", " << k.global_work_size(1) << ", " << k.global_work_size(2) << "'..." << std::endl;
        std::cout << "ViennaCL: Local work size: '"   << k.local_work_size(0) << ", " << k.local_work_size(1) << ", " << k.local_work_size(2) << "'..." << std::endl;
        #endif

        std::size_t tmp_global[3];
        tmp_global[0] = k.global_work_size(0);
        tmp_global[1] = k.global_work_size(1);
        tmp_global[2] = k.global_work_size(2);

        std::size_t tmp_local[3];
        tmp_local[0] = k.local_work_size(0);
        tmp_local[1] = k.local_work_size(1);
        tmp_local[2] = k.local_work_size(2);

        cl_int err = clEnqueueNDRangeKernel(queue.handle().get(), k.handle().get(), (tmp_global[2] == 0) ? 2 : 3, NULL, tmp_global, tmp_local, 0, NULL, NULL);

        if (err != CL_SUCCESS)
        {
          //could not start kernel with any parameters
          std::cerr << "ViennaCL: FATAL ERROR: Kernel start failed for '" << k.name() << "'." << std::endl;
          VIENNACL_ERR_CHECK(err);
        }

      }

      #if defined(VIENNACL_DEBUG_ALL) || defined(VIENNACL_DEBUG_KERNEL)
      queue.finish();
      std::cout << "ViennaCL: Kernel " << k.name() << " finished!" << std::endl;
      #endif
    } //enqueue()


    /** @brief Convenience function that enqueues the provided kernel into the first queue of the currently active device in the currently active context */
    template <typename KernelType>
    void enqueue(KernelType & k)
    {
      enqueue(k, k.context().get_queue());
    }

    inline void enqueue(viennacl::generator::custom_operation & op, viennacl::ocl::command_queue const & queue)
    {
      generator::enqueue_custom_op(op,queue);
    }

    inline void enqueue(viennacl::generator::custom_operation & op)
    {
      enqueue(op, viennacl::ocl::current_context().get_queue());
    }

  } // namespace ocl
} // namespace viennacl
#endif
