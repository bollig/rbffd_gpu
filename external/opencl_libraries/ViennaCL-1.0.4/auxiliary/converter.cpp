/*
* Converts OpenCL sources to header file string constants
*/

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>

#include <boost/filesystem/operations.hpp>
#include <boost/filesystem/path.hpp>
#include <iostream>

namespace fs = boost::filesystem;

void writeSourceFile(std::ofstream & out_file, std::string & filename, const char * dirname, const char * alignment)
{
    std::string fullpath(dirname);
    fullpath += "/";
    fullpath += alignment;
    fullpath += "/";
    fullpath += filename;
    std::ifstream in_file(fullpath.c_str());
    std::string tmp;

    if (in_file.is_open())
    {
        //write variable declaration:
        out_file << "const char * const " << dirname << "_" << alignment << "_" << filename.substr(0, filename.size()-3) << " = " << std::endl;
    
        //write source string:
        while (getline(in_file, tmp, '\n'))
        {
            if (tmp.size() > 0)
            {
      	        //out_file << "\"" << tmp.replace(tmp.end()-1, tmp.end(), "\\n\"") << std::endl;
                if ( *(tmp.end()-1) == '\r')  //Windows line delimiter, \r\n
                    out_file << "\"" << tmp.replace(tmp.end()-1, tmp.end(), "\\n\"") << std::endl;
                else //Unix line delimiter \n
                    out_file << "\"" << tmp.append("\\n\"") << std::endl;
            }
        }
        out_file << "; //" << dirname << "_" << alignment << "_" << filename.substr(0, filename.size()-3)  << std::endl << std::endl;
        
    }
    else
        std::cerr << "Failed to open file " << filename << std::endl;
}

void createSourceFile(const char * dirname)
{
    //Step 1: Open source file
    std::string header_name(dirname);
    std::ofstream source_file(("../viennacl/linalg/kernels/" + header_name + "_source.h").c_str());

    //Step 2: Write source header file preamble
    std::string dirname_uppercase(dirname);
    std::transform(dirname_uppercase.begin(), dirname_uppercase.end(), dirname_uppercase.begin(), toupper);
    source_file << "#ifndef _VIENNACL_" << dirname_uppercase << "_SOURCE_HPP_" << std::endl;
    source_file << "#define _VIENNACL_" << dirname_uppercase << "_SOURCE_HPP_" << std::endl;
    source_file << "//Automatically generated file from aux-directory, do not edit manually!" << std::endl;
    source_file << "namespace viennacl" << std::endl;
    source_file << "{" << std::endl;
    source_file << " namespace linalg" << std::endl;
    source_file << " {" << std::endl;
    source_file << "  namespace kernels" << std::endl;
    source_file << "  {" << std::endl;

    //Step 3: Write all OpenCL kernel sources into header file
    fs::path filepath = fs::system_complete( fs::path( dirname ) );
    if ( fs::is_directory( filepath ) )
    {
        std::cout << "\nIn directory: " << filepath.directory_string() << std::endl;

        fs::directory_iterator end_iter;
        //write and register single precision sources:
        for ( fs::directory_iterator alignment_itr( filepath );
              alignment_itr != end_iter;
              ++alignment_itr )
        {
            if (fs::is_directory( alignment_itr->path() ))
            {
                std::cout << "\nIn directory: " << alignment_itr->path().directory_string() << std::endl;

                //write and register single precision sources:
                for ( fs::directory_iterator cl_itr( alignment_itr->path() );
                      cl_itr != end_iter;
                      ++cl_itr )
                {
                    std::string fname = cl_itr->path().filename();
                    std::string alignment = alignment_itr->path().filename();
                    if (fname.substr(fname.size()-3, 3) == ".cl")
                        writeSourceFile(source_file, fname, dirname, alignment.c_str());
                        //std::cout << alignment_itr->path().filename() << "/" << fname << std::endl;
                } //for                
            } //if is_directory
        } //for alignment_iterator
    } //if is_directory
    else
        std::cerr << "Cannot access directory " << dirname << std::endl;

    //Final Step: Write file tail:
    source_file << "  }  //namespace kernels" << std::endl;
    source_file << " }  //namespace linalg" << std::endl;
    source_file << "}  //namespace viennacl" << std::endl;
    source_file << "#endif" << std::endl;
    source_file.close();
}


unsigned int getBestKernel(const char * dirname, std::string & kernel_name, unsigned int alignment)
{
    unsigned int search_alignment = alignment;
    //std::cout << "Searching for best match for " << kernel_name << " with alignment " << alignment << std::endl;

    while (search_alignment > 1)
    {
        std::ostringstream oss;
        oss << dirname << "/align" << search_alignment;
        //std::cout << "Searching " << oss.str() << std::endl;

        //try to find kernel in directory:
        fs::path filepath = fs::system_complete( fs::path( oss.str() ) );
        if ( fs::is_directory( filepath ) ) //directory exists?
        {
            fs::directory_iterator end_iter;
            for ( fs::directory_iterator cl_itr( filepath );
                  cl_itr != end_iter;
                  ++cl_itr )
            {
                std::string fname = cl_itr->path().filename();
                if (fname == kernel_name)
                {
                  //std::cout << "Found matching kernel for " << kernel_name << " with alignment " << alignment << " at alignment " << search_alignment << std::endl;
                    return search_alignment;
                }
            }
        }

        search_alignment /= 2;
    }

    //std::cout << "Found alignment 1 only..." << std::endl;
    //nothing found: return alignment 1:
    return 1;
}


void writeKernelInit(std::ostream & kernel_file, const char * dirname, std::string & subfolder, bool is_float)
{
    //extract alignment information from subfolder string:
    std::istringstream stream(subfolder.substr(5, subfolder.size()-5));
    unsigned int alignment = 0;
    stream >> alignment;
    if (alignment == 0)
        std::cerr << "ERROR: Could not extract alignment from " << subfolder << std::endl;

    kernel_file << "    template <> void " << dirname;
    if (is_float)
        kernel_file << "<float, ";
    else
        kernel_file << "<double, ";
    kernel_file << alignment << ">::init()" << std::endl;
    kernel_file << "    {" << std::endl;
    kernel_file << "     static bool init_done = false;" << std::endl;
    kernel_file << "     if (!init_done)" << std::endl;
    kernel_file << "     {" << std::endl;
    kernel_file << "       std::string source;" << std::endl;
    kernel_file << "       viennacl::ocl::program prog;" << std::endl;

    //iterate over all kernels in align1-folder:
    std::string current_dir(dirname);
    current_dir += "/align1";
    fs::path filepath = fs::system_complete( fs::path( current_dir ) );

    fs::directory_iterator end_iter;
    //write and register single precision sources:
    for ( fs::directory_iterator cl_itr( filepath );
          cl_itr != end_iter;
          ++cl_itr )
    {
        std::string fname = cl_itr->path().filename();
        if (fname.substr(fname.size()-3, 3) == ".cl")
        {
            //add kernel source to program string:
            kernel_file << "       source.append(";
            if (!is_float)
                kernel_file << "viennacl::tools::make_double_kernel(";
            kernel_file << dirname << "_align" << getBestKernel(dirname, fname, alignment) << "_" << fname.substr(0, fname.size()-3);
            if (!is_float)
                kernel_file << ")";
            kernel_file << ");" << std::endl;
        }
    } //for                
    
    kernel_file << "       prog.create(source);" << std::endl << std::endl;
    
    //write and register single precision sources:
    for ( fs::directory_iterator cl_itr( filepath );
          cl_itr != end_iter;
          ++cl_itr )
    {
        std::string fname = cl_itr->path().filename();
        if (fname.substr(fname.size()-3, 3) == ".cl")
        {
            //initialize kernel:
            kernel_file << "       " << fname.substr(0, fname.size()-3) << ".prepareInit(";
            kernel_file << "\"" << fname.substr(0, fname.size()-3) << "\", ";
            kernel_file << "prog);" << std::endl;
        }
    } //for                
    
    kernel_file << "       init_done = true;" << std::endl;
    kernel_file << "     }" << std::endl;
    kernel_file << "    }" << std::endl;
}




void createKernelFile(const char * dirname)
{
    //Step 1: Open kernel file
    std::string header_name(dirname);
    std::ofstream kernel_file(("../viennacl/linalg/kernels/" + header_name + "_kernels.h").c_str());

    //Step 2: Write kernel header file preamble
    std::string dirname_uppercase(dirname);
    std::transform(dirname_uppercase.begin(), dirname_uppercase.end(), dirname_uppercase.begin(), toupper);
    kernel_file << "#ifndef _VIENNACL_" << dirname_uppercase << "_KERNELS_HPP_" << std::endl;
    kernel_file << "#define _VIENNACL_" << dirname_uppercase << "_KERNELS_HPP_" << std::endl;
    kernel_file << "#include \"viennacl/tools/tools.hpp\"" << std::endl;
    kernel_file << "#include \"viennacl/ocl/kernel.hpp\"" << std::endl;
    kernel_file << "#include \"viennacl/linalg/kernels/" << dirname << "_source.h\"" << std::endl;
    kernel_file << std::endl;
    kernel_file << "//Automatically generated file from aux-directory, do not edit manually!" << std::endl;
    kernel_file << "namespace viennacl" << std::endl;
    kernel_file << "{" << std::endl;
    kernel_file << " namespace linalg" << std::endl;
    kernel_file << " {" << std::endl;
    kernel_file << "  namespace kernels" << std::endl;
    kernel_file << "  {" << std::endl;


    //Step 3: Write class information:
    kernel_file << "   template<class TYPE, unsigned int alignment>" << std::endl;
    kernel_file << "   struct " << dirname << std::endl;
    kernel_file << "   {" << std::endl;
    
    std::string dir(dirname);
    fs::path filepath = fs::system_complete( fs::path( dir + "/align1" ) );
    if ( fs::is_directory( filepath ) )
    {
        fs::directory_iterator end_iter;
        //register kernels available in align1/:
        for ( fs::directory_iterator dir_itr( filepath );
              dir_itr != end_iter;
              ++dir_itr )
        {
            std::string fname = dir_itr->path().filename();
            if (fname.substr(fname.size()-3, 3) == ".cl")
                kernel_file << "    static viennacl::ocl::kernel " << fname.substr(0, fname.size()-3) << ";" << std::endl;
        }
    }
    else
        std::cerr << "Cannot access directory " << dirname << std::endl;

    kernel_file << std::endl << "    static void init();" << std::endl;
    kernel_file << "   };" << std::endl;
    kernel_file << std::endl;

    //Step 4: Write static instantiations:
    if ( fs::is_directory( filepath ) )
    {
        fs::directory_iterator end_iter;
        //register kernels available in align1/:
        for ( fs::directory_iterator dir_itr( filepath );
              dir_itr != end_iter;
              ++dir_itr )
        {
            std::string fname = dir_itr->path().filename();
            if (fname.substr(fname.size()-3, 3) == ".cl")
            {
                kernel_file << "   template <typename T, unsigned int ALIGNMENT> viennacl::ocl::kernel ";
                kernel_file << dirname << "<T, ALIGNMENT>::" << fname.substr(0, fname.size()-3) << ";" << std::endl;
            }
        }
    }
    else
        std::cerr << "Cannot access directory " << dirname << std::endl;

    kernel_file << std::endl;

    //Step 5: Write single precision kernels
    kernel_file << std::endl << "    /////////////// single precision kernels //////////////// " << std::endl;
    filepath = fs::system_complete( fs::path( dir ) );
    if ( fs::is_directory( filepath ) )
    {
        std::cout << "\nIn directory: " << filepath.directory_string() << std::endl;

        fs::directory_iterator end_iter;
        //write and register single precision sources:
        for ( fs::directory_iterator alignment_itr( filepath );
              alignment_itr != end_iter;
              ++alignment_itr )
        {
            if (fs::is_directory( alignment_itr->path() ))
            {
                std::string subfolder = alignment_itr->path().filename();
                writeKernelInit(kernel_file, dirname, subfolder, true);
            } //if is_directory
        } //for alignment_iterator
        kernel_file << std::endl;
    } //if is_directory
    else
        std::cerr << "Cannot access directory " << dirname << std::endl;

    //Step 6: Write double precision kernels
    kernel_file << std::endl << "    /////////////// double precision kernels //////////////// " << std::endl;
    filepath = fs::system_complete( fs::path( dir ) );
    if ( fs::is_directory( filepath ) )
    {
        std::cout << "\nIn directory: " << filepath.directory_string() << std::endl;

        fs::directory_iterator end_iter;
        //write and register single precision sources:
        for ( fs::directory_iterator alignment_itr( filepath );
              alignment_itr != end_iter;
              ++alignment_itr )
        {
            if (fs::is_directory( alignment_itr->path() ))
            {
                std::string subfolder = alignment_itr->path().filename();
                writeKernelInit(kernel_file, dirname, subfolder, false);
            } //if is_directory
        } //for alignment_iterator
        kernel_file << std::endl;
    } //if is_directory
    else
        std::cerr << "Cannot access directory " << dirname << std::endl;

    //Final Step: Write file tail:
    kernel_file << "  }  //namespace kernels" << std::endl;
    kernel_file << " }  //namespace linalg" << std::endl;
    kernel_file << "}  //namespace viennacl" << std::endl;
    kernel_file << "#endif" << std::endl;
    kernel_file.close();
}

void createHeaders(const char * dirname)
{
    createKernelFile(dirname);
    createSourceFile(dirname);
}

int main(int args, char * argsv[])
{

    createHeaders("compressed_matrix");
    createHeaders("coordinate_matrix");
    createHeaders("matrix");
    createHeaders("scalar");
    createHeaders("vector");

}
