#ifndef __CL_BASE_CLASS_H__
#define __CL_BASE_CLASS_H__
#define __CL_ENABLE_EXCEPTIONS
#include <iostream> 
#include <string>
#include <sstream>
#include <CL/cl.hpp>

// A bunch of util methods borrowed from Ian's tutorial on opencl
class CLBaseClass 
{
    protected:
        unsigned int deviceUsed;
        std::vector<cl::Device> devices;

        // We use static so all of our inheriting classes can share buffers
        // across the context
        static cl::Context context;
        // Track if context was created so we dont accidentally make a new one
        static int contextCreated; 

        cl::CommandQueue queue;
        cl::Program program;
        cl::Kernel kernel;

        //debugging variables
        cl_int err;
        cl::Event event;
        cl::Event event2;
        cl::Event event3;

    public:

        //----------------------------------------------------------------------
        CLBaseClass(int rank=0);

        //----------------------------------------------------------------------
        std::string addExtension(std::string& source, std::string ext, bool
                enabled = true);

        //----------------------------------------------------------------------
        std::string getDeviceFP64Extension(int device_id=-1);

        //----------------------------------------------------------------------
        void loadProgram(std::string& kernel_source, bool enable_fp64 = false);


        std::string loadFileContents(const char* filename, bool searchEnvDir);

        //----------------------------------------------------------------------
        // Helper function to get error string
        // *********************************************************************
        const char* oclErrorString(cl_int error)
        {
            static const char* errorString[] = {
                "CL_SUCCESS",
                "CL_DEVICE_NOT_FOUND",
                "CL_DEVICE_NOT_AVAILABLE",
                "CL_COMPILER_NOT_AVAILABLE",
                "CL_MEM_OBJECT_ALLOCATION_FAILURE",
                "CL_OUT_OF_RESOURCES",
                "CL_OUT_OF_HOST_MEMORY",
                "CL_PROFILING_INFO_NOT_AVAILABLE",
                "CL_MEM_COPY_OVERLAP",
                "CL_IMAGE_FORMAT_MISMATCH",
                "CL_IMAGE_FORMAT_NOT_SUPPORTED",
                "CL_BUILD_PROGRAM_FAILURE",
                "CL_MAP_FAILURE",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "CL_INVALID_VALUE",
                "CL_INVALID_DEVICE_TYPE",
                "CL_INVALID_PLATFORM",
                "CL_INVALID_DEVICE",
                "CL_INVALID_CONTEXT",
                "CL_INVALID_QUEUE_PROPERTIES",
                "CL_INVALID_COMMAND_QUEUE",
                "CL_INVALID_HOST_PTR",
                "CL_INVALID_MEM_OBJECT",
                "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR",
                "CL_INVALID_IMAGE_SIZE",
                "CL_INVALID_SAMPLER",
                "CL_INVALID_BINARY",
                "CL_INVALID_BUILD_OPTIONS",
                "CL_INVALID_PROGRAM",
                "CL_INVALID_PROGRAM_EXECUTABLE",
                "CL_INVALID_KERNEL_NAME",
                "CL_INVALID_KERNEL_DEFINITION",
                "CL_INVALID_KERNEL",
                "CL_INVALID_ARG_INDEX",
                "CL_INVALID_ARG_VALUE",
                "CL_INVALID_ARG_SIZE",
                "CL_INVALID_KERNEL_ARGS",
                "CL_INVALID_WORK_DIMENSION",
                "CL_INVALID_WORK_GROUP_SIZE",
                "CL_INVALID_WORK_ITEM_SIZE",
                "CL_INVALID_GLOBAL_OFFSET",
                "CL_INVALID_EVENT_WAIT_LIST",
                "CL_INVALID_EVENT",
                "CL_INVALID_OPERATION",
                "CL_INVALID_GL_OBJECT",
                "CL_INVALID_BUFFER_SIZE",
                "CL_INVALID_MIP_LEVEL",
                "CL_INVALID_GLOBAL_WORK_SIZE",
            };

            const int errorCount = sizeof(errorString) / sizeof(errorString[0]);
            const int index = -error;
            return (index >= 0 && index < errorCount) ? errorString[index] : "";
        }

        // Split a string (for example, a list of extensions, given a character
        // deliminator)
        std::vector<std::string>& split(const std::string &s, char delim,
                std::vector<std::string> &elems);

        std::vector<std::string> split(const std::string &s, char delim);


        std::vector<std::string>& split(const std::string &s, const
                std::string delim, std::vector<std::string> &elems, bool
                keep_substr=false);
        std::vector<std::string> split(const std::string &s, const std::string delim, bool keep_substr=false); 

};

#endif 
