#ifndef __CL_BASE_CLASS_H__
#define __CL_BASE_CLASS_H__
#define __CL_ENABLE_EXCEPTIONS
#include <iostream> 
#include <CL/cl.hpp>

// A bunch of util methods borrowed from Ian's tutorial on opencl
class CLBaseClass 
{
    protected:
        //handles for creating an opencl context
        //cl_platform_id platform;

        //device variables
        //cl_device_id* devices;
        //cl_uint numDevices;
        unsigned int deviceUsed;
        std::vector<cl::Device> devices;

        //cl_context context;
        cl::Context context;

        //cl_command_queue command_queue;
        cl::CommandQueue queue;
        //cl_program program;
        cl::Program program;
        //cl_kernel kernel;
        cl::Kernel kernel;


        //debugging variables
        cl_int err;
        ///cl_event event;
        cl::Event event;

    public:

        CLBaseClass(int rank=0) {
            std::cout<<"Loading and compiling OpenCL source for DerivativeCL\n";

            printf("Initialize OpenCL object and context\n");
            //setup devices and context

            //this function is defined in util.cpp
            //it comes from the NVIDIA SDK example code
            ///err = oclGetPlatformID(&platform);
            //oclErrorString is also defined in util.cpp and comes from the NVIDIA SDK
            ///printf("oclGetPlatformID: %s\n", oclErrorString(err));
            std::vector<cl::Platform> platforms;
            std::cout << "Getting the platform" << std::endl;
            err = cl::Platform::get(&platforms);
            std::cout << "GOT PLATFORM" << std::endl; 
            printf("cl::Platform::get(): %s\n", oclErrorString(err));
            if (platforms.size() == 0) {
                printf("Platform size 0\n");
            }


            // Get the number of GPU devices available to the platform
            // we should probably expose the device type to the user
            // the other common option is CL_DEVICE_TYPE_CPU
            ///err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, NULL, &numDevices);
            ///printf("clGetDeviceIDs (get number of devices): %s\n", oclErrorString(err));


            // Create the device list
            ///devices = new cl_device_id [numDevices];
            ///err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, numDevices, devices, NULL);
            ///printf("clGetDeviceIDs (create device list): %s\n", oclErrorString(err));


            //for right now we just use the first available device
            //later you may have criteria (such as support for different extensions)
            //that you want to use to select the device
            //deviceUsed = 0;



            //create the context
            ///context = clCreateContext(0, 1, &devices[deviceUsed], NULL, NULL, &err);
            //context properties will be important later, for now we go with defualts
            cl_context_properties properties[] =
            { CL_CONTEXT_PLATFORM, (cl_context_properties)(platforms[0])(), 0};

            std::cout << "Creating cl::Context" << std::endl;
            context = cl::Context(CL_DEVICE_TYPE_GPU, properties);
            devices = context.getInfo<CL_CONTEXT_DEVICES>();

            //create the command queue we will use to execute OpenCL commands
            ///command_queue = clCreateCommandQueue(context, devices[deviceUsed], 0, &err);
            try{
                deviceUsed = rank % devices.size(); 
                queue = cl::CommandQueue(context, devices[deviceUsed], 0, &err);
                printf("[initialize] Using CL device: %d\n", deviceUsed);
                std::cout << "\tDevice Name: " << devices[deviceUsed].getInfo<CL_DEVICE_NAME>().c_str() << std::endl;
                std::cout << "\tDriver Version: " << devices[deviceUsed].getInfo<CL_DRIVER_VERSION>() << std::endl; 
                std::cout << "\tVendor: " << devices[deviceUsed].getInfo<CL_DEVICE_VENDOR_ID>() << std::endl; 
                std::cout << "\tMax Compute Units: " << devices[deviceUsed].getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>() << std::endl; 
                std::cout << "\tMax Work Item Dimensions: " << devices[deviceUsed].getInfo<CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS>() << std::endl; 
            }
            catch (cl::Error er) {
                printf("[initialize] ERROR: %s(%d)\n", er.what(), er.err());
            }
            std::cout << "Done with cl::Context setup..." << std::endl;	
        }


        void loadProgram(std::string kernel_source)
        {
            //Program Setup
            int pl;
            //size_t program_length;
            printf("load the program\n");

            pl = kernel_source.size();
            printf("kernel size: %d\n", pl);
            //printf("kernel: \n %s\n", kernel_source.c_str());
            try
            {
                cl::Program::Sources source(1,
                        std::make_pair(kernel_source.c_str(), pl));
                program = cl::Program(context, source);
            }
            catch (cl::Error er) {
                printf("[loadProgram] ERROR: %s(%s)\n", er.what(), oclErrorString(er.err()));
            }

            printf("build program\n");
            try
            {
                err = program.build(devices);
            }
            catch (cl::Error er) {
                printf("program.build: %s\n", oclErrorString(er.err()));
                //if(err != CL_SUCCESS){
            }
            printf("done building program\n");
            std::cout << "Build Status: " << program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(devices[0]) << std::endl;
            std::cout << "Build Options:\t" << program.getBuildInfo<CL_PROGRAM_BUILD_OPTIONS>(devices[0]) << std::endl;
            std::cout << "Build Log:\t " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[0]) << std::endl;

            }


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

        };

#endif 
