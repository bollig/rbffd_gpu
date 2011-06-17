#ifndef __CL_FILE_LOADER_H__
#define __CL_FILE_LOADER_H__

#include <string> 
#include <map>
#include <iostream>
#include <sstream>
#include <CL/cl.hpp> 

typedef std::map<std::string, std::pair<bool, int> > EXT_MAP;

class CLFileLoader 
{
    protected: 
        // EXTENSION_NAME : enabled/disabled
        // note: ONLY the extension name, not the full #pragma ... string here
        std::map<std::string, bool> extensions;  

    public: 

        // Setup loader to read multiple files from disk; 
        // fills extensions by querying the OpenCL platform 
        CLFileLoader(int platform_selection=0) {

            std::vector<cl::Platform> platforms;
            std::cout << "Getting the platform" << std::endl;
            cl_int err = cl::Platform::get(&platforms);
            std::cout << "Found " << platforms.size() << " platform(s) : " << oclErrorString(err) << std::endl; 

            std::vector<cl::Platform>::iterator i;
            if(platforms.size() > platform_selection)
            {
                i = platforms.begin() + platform_selection; 
                //for(i = platforms.begin(); i != platforms.end(); ++i)
                {
                    std::cout << "Platform: " << (*i).getInfo<CL_PLATFORM_NAME>() << ", Version: " << (*i).getInfo<CL_PLATFORM_VERSION>() << std::endl; 
                    std::vector<std::string> exts; 
                    split((*i).getInfo<CL_PLATFORM_EXTENSIONS>(), ' ', exts); 


                    cl_context_properties cps[3] = 
                    { 
                        CL_CONTEXT_PLATFORM, 
                        (cl_context_properties)(*i)(),
                        0 
                    };

                    cl::Context context = cl::Context(CL_DEVICE_TYPE_GPU, cps, NULL, NULL, &err);

                    std::vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();

                    if (devices.size() == 0) 
                    {
                        std::cout << "WARNING! No devices found" << std::endl;
                    }
                    std::vector<cl::Device>::iterator d; 
                    for (d = devices.begin(); d != devices.end(); ++d) {
                        std::vector<std::string> d_exts = split((*d).getInfo<CL_DEVICE_EXTENSIONS>(), " ", exts); 

                    }

                    // exts should now contain ALL extensions
                    std::vector<std::string>::iterator j; 
                    for (j = exts.begin(); j != exts.end(); ++j) {
//                        std::cout << "Adding extension: '" << (*j) << "' : disabled\n"; 
                        extensions[(*j)] = false; 
                    }
                }
            }
        }


        void printExtensions() {
            std::cout << "Available extensions: \n";
            std::map<std::string, bool>::iterator e; 
            for (e=extensions.begin(); e!=extensions.end(); ++e) {
                std::cout << "#pragma OPENCL EXTENSION " << e->first << " : ";
                if (e->second) {
                    std::cout << "enabled" << std::endl;
                } else {
                    std::cout << "disabled" << std::endl;
                }
            }
        }



        // Split a string (for example, a list of extensions, given a string deliminator)
        // NOTE: if keep_substr=false we discard the delim wherever it is matched
        // if keep_substr=true we keep the delim in the substrings
        std::vector<std::string>& split(const std::string &s, const std::string& delim, std::vector<std::string> &elems, bool keep_substr=false) {
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

        std::vector<std::string> split(const std::string &s, const std::string& delim) {
            std::vector<std::string> elems;
            return split(s, delim, elems);
        }

        // Split a string (for example, a list of extensions, given a character deliminator)
        std::vector<std::string>& split(const std::string &s, char delim, std::vector<std::string> &elems) {
            std::stringstream ss(s);
            std::string item;
            while(std::getline(ss, item, delim)) {
                elems.push_back(item);
            }
            return elems;
        }

        std::vector<std::string> split(const std::string &s, char delim) {
            std::vector<std::string> elems;
            return split(s, delim, elems);
        }

#if 0
        cl_context_properties cps[3] = 
        { 
            CL_CONTEXT_PLATFORM, 
            (cl_context_properties)(*i)(),
            0 
        };


        if(NULL == (*i)())
        {
            sampleCommon->error("NULL platform found so Exiting Application.");
            return SDK_FAILURE;
        }

        context = cl::Context(dType, cps, NULL, NULL, &status);
        if(!sampleCommon->checkVal(status, 
                    CL_SUCCESS,
                    "Context::Context() failed."))
            return SDK_FAILURE;

        devices = context.getInfo<CL_CONTEXT_DEVICES>();
        if(!sampleCommon->checkVal(status, CL_SUCCESS, "Context::getInfo() failed."))
            return SDK_FAILURE;

        if (devices.size() == 0) 
        {
            sampleCommon->error("No device available");
            return SDK_FAILURE;
        }

        std::string extensions = devices[0].getInfo<CL_DEVICE_EXTENSIONS>();
        if(!strstr(extensions.c_str(), "cl_amd_fp64"))
        {
            sampleCommon->error("Device does not support double precision extension!");
            exit(SDK_SUCCESS);
        }


}


// Read a *.cl file from disk and split the kernels within into separate strings
// any #pragma OPENCL EXTENSION strings should enable/disable elements of the extensions map above
CLFileLoader(std::string filename); 

// return the full kernel string for kernelName (matched with the name of the kernel routine)
// prepend with all extensions enabled in the extensions map above
// if the string contains "double" pre-pend the #pragma OPENCL EXTENSION for ATI or NVidia
// and warn the user that it was added
std::string getKernelString(std::string kernelName); 
#endif 


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

#endif // __cl_file_loader_h__
