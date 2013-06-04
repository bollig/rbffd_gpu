#include <iostream>
#include <fstream>
#include <stdio.h>
#include <vector>
#include <stdlib.h>

#include <limits.h>

#include "cl_base_class.h"

// Static definition.
cl::Context CLBaseClass::context;
cl::CommandQueue CLBaseClass::queue;
int CLBaseClass::contextCreated = 0;

//----------------------------------------------------------------------
CLBaseClass::CLBaseClass(int rank) {
    printf("Initialize OpenCL object and context\n");

    if (!contextCreated) {
        //setup devices and context
        std::vector<cl::Platform> platforms;
        std::cout << "Getting the platform" << std::endl;
        err = cl::Platform::get(&platforms);
        std::cout << "GOT PLATFORM" << std::endl;
        printf("cl::Platform::get(): %s\n", oclErrorString(err));
        if (platforms.size() == 0) {
            printf("Platform size 0\n");
        }

        //create the context
        cl_context_properties properties[] =
        { CL_CONTEXT_PLATFORM, (cl_context_properties)(platforms[0])(), 0};

        std::cout << "Creating cl::Context (only selecting GPU devices)" << std::endl;
        //context = cl::Context(CL_DEVICE_TYPE_GPU, properties);
        context = cl::Context(CL_DEVICE_TYPE_ACCELERATOR, properties);
        // This prevents the context from being created again
        contextCreated++;
    }

    devices = context.getInfo<CL_CONTEXT_DEVICES>();

    //create the command queue we will use to execute OpenCL commands
    try{
        deviceUsed = rank % devices.size();
        queue = cl::CommandQueue(context, devices[deviceUsed], CL_QUEUE_PROFILING_ENABLE, &err);
        printf("[initialize] Using CL device: %d\n", deviceUsed);
        std::cout << "\tDevice Name: " <<
            devices[deviceUsed].getInfo<CL_DEVICE_NAME>().c_str() << std::endl;
        std::cout << "\tDriver Version: " <<
            devices[deviceUsed].getInfo<CL_DRIVER_VERSION>() << std::endl;
        std::cout << "\tVendor: " <<
            devices[deviceUsed].getInfo<CL_DEVICE_VENDOR_ID>() << std::endl;
        std::cout << "\tMax Compute Units: " <<
            devices[deviceUsed].getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>() <<
            std::endl;
        std::cout << "\tMax Work Item Dimensions: " <<
            devices[deviceUsed].getInfo<CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS>()
            << std::endl;
    }
    catch (cl::Error er) {
        printf("[initialize] ERROR: %s(%d)\n", er.what(), er.err());
    }
    std::cout << "Done with cl::Context setup..." << std::endl;
}

//----------------------------------------------------------------------
std::string CLBaseClass::addExtension(std::string& source, std::string ext,
        bool enabled) {
    std::ostringstream oss;
    oss << "#pragma OPENCL EXTENSION " << ext << ": ";
    if (enabled)
        oss << "enable\n";
    else
        oss << "disable\n";
#if 0
    if (enabled)
            oss << "typedef double FLOAT;\n";
    else
            oss << "typedef float FLOAT;\n";
#endif

    oss << "\n" << source;

    return oss.str();
}

//----------------------------------------------------------------------
std::string CLBaseClass::getDeviceFP64Extension(int device_id) {
    if (device_id < 0) {
        device_id = deviceUsed;
    }

    std::vector<std::string> d_exts =
        split(devices[device_id].getInfo<CL_DEVICE_EXTENSIONS>(), ' ');

    std::vector<std::string>::iterator d;
    int count = 0;

    std::string ext = "";
    for (d = d_exts.begin(); d != d_exts.end(); d++) {
        if ((*d).find("fp64") != std::string::npos) {
            std::cout << "FOUND MATCHING FP64 EXTENSION: " << *d << std::endl;
            ext = *d;
            count ++;
        }
    }
    if (count > 1) {
        // If we find multiple extensions ending in fp64 then we
        // want to return the API standard extension:
        return "cl_khr_fp64";
    }
    return ext;
}

//----------------------------------------------------------------------
void CLBaseClass::loadProgram(std::string& kernel_source, bool enable_fp64)
{
    //Program Setup
    int pl;
    //unsigned int program_length;

    std::string updated_source(kernel_source);

//    if (enable_fp64)
    {
        updated_source = addExtension(kernel_source, getDeviceFP64Extension(deviceUsed), enable_fp64);
    }

    // std::cout << updated_source << std::endl;
    pl = updated_source.size();
    printf("[CLBaseClass] building kernel source of size: %d\n", pl);
//    printf("KERNEL: \n %s\n", updated_source.c_str());
    try
    {
        cl::Program::Sources source(1,
                std::make_pair(updated_source.c_str(), pl));
        program = cl::Program(context, source);
    }
    catch (cl::Error er) {
        printf("[CLBaseClass::loadProgram] ERROR: %s(%s)\n", er.what(), oclErrorString(er.err()));
    }

    try
    {
            char* kernel_dir = getenv("CL_KERNELS");
            char* cwd = getenv("PWD");

            if (kernel_dir == NULL) {
                printf("\n**** [CLBaseClass] Error: You must set CL_KERNELS in your environment to run on the GPU!\n\n");
                exit(EXIT_FAILURE);
            }
     //       printf("Loading kernels from the directory: %s and .\n **** Specified by environment variable: CL_KERNELS\n", kernel_dir);
            char includes[PATH_MAX];
            sprintf(includes, "-I%s -I%s", kernel_dir, cwd);
            //sprintf(includes, "-cl-opt-disable -I%s -I%s", kernel_dir, cwd);
            err = program.build(devices, includes);
    }
    catch (cl::Error er) {
        printf("program.build: %s\n", oclErrorString(er.err()));
        std::cout << "Build Status: " <<
            program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(devices[deviceUsed])
            << std::endl;
        std::cout << "Build Options:\t" <<
            program.getBuildInfo<CL_PROGRAM_BUILD_OPTIONS>(devices[deviceUsed])
            << std::endl;
        std::cout << "Build Log:\t " <<
            program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[deviceUsed]) <<
            std::endl;
        exit(EXIT_FAILURE);
    }
    printf("[CLBaseClass] done building program\n");

#if 0
    std::cout << "Build Status: " <<
        program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(devices[deviceUsed]) <<
        std::endl;
    std::cout << "Build Options:\t" <<
        program.getBuildInfo<CL_PROGRAM_BUILD_OPTIONS>(devices[deviceUsed]) <<
        std::endl;
    std::cout << "Build Log:\t " <<
        program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[deviceUsed]) <<
        std::endl;
#endif
}


std::string CLBaseClass::loadFileContents(const char* filename, bool searchEnvDir) {

        if (searchEnvDir) {
                char* kernel_dir = getenv("CL_KERNELS");

                if (kernel_dir == NULL) {
                        printf("\n**** [CLBaseClass] Error: You must set CL_KERNELS in your environment to load kernel file contents!\n\n");
                        exit(EXIT_FAILURE);
                }
                char kernel_path[PATH_MAX];
                sprintf(kernel_path, "%s/%s", kernel_dir, filename);

                std::ifstream ifs(kernel_path);
                std::string str((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
                return str;
        } else {
                // Grab the whole file in one go using the iterators
                std::ifstream ifs(filename);
                std::string str((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
                return str;
        }
}


// Split a string (for example, a list of extensions, given a character
// deliminator)
std::vector<std::string>& CLBaseClass::split(const std::string &s, char delim, std::vector<std::string> &elems) {
    std::stringstream ss(s);
    std::string item;
    while(std::getline(ss, item, delim)) {
        elems.push_back(item);
    }
    return elems;
}

std::vector<std::string> CLBaseClass::split(const std::string &s, char delim) {
    std::vector<std::string> elems;
    return split(s, delim, elems);
}


// Split a string (for example, a list of extensions, given a string deliminator)
// NOTE: if keep_substr=false we discard the delim wherever it is matched
// if keep_substr=true we keep the delim in the substrings
std::vector<std::string>& CLBaseClass::split(const std::string &s, const std::string delim, std::vector<std::string> &elems, bool keep_substr) {
    size_t last_found = 0;
    size_t find_start = last_found;
    size_t found = 0;
    while(found != std::string::npos)
    {
        found = s.find(delim, find_start);
        std::string sub = s.substr(last_found, found - last_found);
        if (sub.length() > 0)
            elems.push_back(sub);
        if (keep_substr) {
            last_found = found;
            find_start = found + delim.length();
        } else {
            last_found = found + delim.length();
            find_start = found + delim.length();
        }
    }
    return elems;
}

std::vector<std::string> CLBaseClass::split(const std::string &s, const std::string delim, bool keep_substr) {
    std::vector<std::string> elems;
    return split(s, delim, elems, keep_substr);
}


