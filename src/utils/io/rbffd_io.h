#ifndef _RBFFD_IO_H_
#define _RBFFD_IO_H_

#include <vector>
#include <string>

template <typename T>
class RBFFD_IO
{
private:
	bool ascii;

public:
        // Read/Write weights to disk in ascii matrix market format. 0 = binary (default), 1 = ascii 
		#if 0
		void setAsciiWeights(int binary_or_ascii) { asciiWeights = binary_or_ascii; }
		void writeAllWeightsToFile(); 
		void writeToFile(DerType which, std::string filename);
		void writeToAsciiFile(DerType which, std::string filename);
		void writeToBinaryFile(DerType which, std::string filename);
		void writeToFile(DerType which) { this->writeToFile(which, this->getFilename(which)); }
		int  loadAllWeightsFromFile();
		int  loadFromFile(DerType which, std::string filename);
		int  loadFromFile(DerType which){ return this->loadFromFile(which, this->getFilename(which)); }

        int  loadFromAsciiFile(DerType which, std::string filename);
        int  loadFromBinaryFile(DerType which, std::string filename);

        int  loadFromAsciiFile(DerType which, std::string filename);
        int  loadFromBinaryFile(DerType which, std::string filename);

        void setAsciiWeights(bool binary_or_ascii) { ascii = binary_or_ascii; }
		#endif


		// size of rows is number of nonzeros in the matrix
    	int loadFromAsciMMFile(std::vector<int>& rows, std::vector<int>& cols, 
			std::vector<T>& values, int& width, int& height, std::string& filename);

		#if 0
    	int writeToAsciMMFile(std::vector<int>& rows, std::vector<int>& cols, 
			std::vector<T>& value, int& nonzeros_per_row, std::string& filename);
		#endif

		int loadFromBinaryMMFile(std::vector<int>& rows, std::vector<int>& cols, 
				std::vector<T>& values,int& width, int& height, std::string& filename);

		int writeToBinaryMMFile(std::vector<int>& rows, std::vector<int>& cols, 
				std::vector<T>& values, int width, int height, std::string& filename);
};

#include "rbffd_io.cpp"
#endif

