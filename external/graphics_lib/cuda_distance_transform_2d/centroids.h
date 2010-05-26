#ifndef __CENTROIDS_H__
#define __CENTROIDS_H__

// for float4
#include <vector_types.h>
#include "local_types.h"
#include <vector>
#include "timege.h"

class DistanceTransformAcc;

class Centroids 
{
private:
	GE::Time* clock3;
	GE::Time* clock_gpu;
	GE::Time* clock_tmp;
	float4* bins_cpu;
	float4* bins_gpu;
	float4* bins_cpu_gpu;
	int nb_bins;
	std::vector<int4> newSeeds;
	DistanceTransformAcc* dtransf;

	/// voronoi bitmap, memory controlled by this class
	float4* data_cpu; 
	int szx, szy; // dimensions of voronoi raster

public:
	Centroids();
	Centroids(int nb_seeds);
	~Centroids();

	/// compute the centroids. No data is sent to the GPU from the host
	/// return bins (x,y,0,nbSeeds in this bin)
	float4* computeOnGPU();

	/// compute the centroids. No data is sent to the GPU from the host
	/// return bins (x,y,0,nbSeeds in this bin)
	float4* computeOnCPU(int szx, int szy);

	/// Run the GPU algorithm on the CPU to check whether 
	/// it is working as expected
	float4* simulateGPUonCPU(int szx, int szy);

	void printComputationCPU();
	void printComputationGPU();
	void printComputationGPUonCPU();
	void printSeeds(std::vector<Seed>& seeds);

	void printError(std::vector<Seed>& seeds);

	// depends on CUDA types
	float4* getBinsCPU() { return bins_cpu; }
	float4* getBinsGPU() { return bins_gpu; }
	void setNbBins(int nb_bins);
	unsigned int getNbBins() { return nb_bins; }
	void setDistanceTransform(DistanceTransformAcc& transf) 
		{ dtransf = &transf; }
	float4* getVoronoiBitmap();
	void getDims(int* szx_, int* szy_) {
		*szx_ = szx;
		*szy_ = szy;
	}
	std::vector<int4>& getNewSeeds() { return newSeeds; }

private:
	Centroids(const Centroids&);
};

#endif
