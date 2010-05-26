#include <vector_functions.h>
#include "centroids.h"
#include "distance_transform_2d.h"
#include "ping_pong_cuda.h"
#include "map.h"
#include "array_cuda_1d.h"

//----------------------------------------------------------------------
Centroids::Centroids()
{
	clock3 = new GE::Time("histogram");
	clock_gpu = new GE::Time("centroids on the gpu");
	clock_tmp = new GE::Time("Centroids::misc");
	nb_bins = 0;
	bins_cpu = 0;
	bins_gpu = 0;
	bins_cpu_gpu = 0;
	data_cpu = 0;
}
//----------------------------------------------------------------------
Centroids::Centroids(int nb_seeds)
{
	clock3 = new GE::Time("histogram");
	nb_bins = nb_seeds;
	bins_cpu = new float4[nb_seeds];
	data_cpu = 0;
}
//----------------------------------------------------------------------
void Centroids::setNbBins(int nb_bins)
{
	this->nb_bins = nb_bins;
	if (bins_cpu) delete [] bins_cpu;
	bins_cpu = new float4[nb_bins];
}
//----------------------------------------------------------------------
Centroids::~Centroids()
{
}
//----------------------------------------------------------------------
float4* Centroids::simulateGPUonCPU(int szx, int szy)
{
	int nb_seeds = newSeeds.size();
	printf("nb_seeds= %d\n", nb_seeds);

	if (!bins_cpu_gpu) {
		printf("allocated  memory for bins_cpu_gpu\n");
		bins_cpu_gpu = new float4 [nb_seeds];
	}

	float4* data = getVoronoiBitmap(); // from GPU
	printf("data= %d\n", data);

	for (int i=0; i < nb_seeds; i++) {
		int xseed = newSeeds[i].x;
		int yseed = newSeeds[i].y;
		int seed = newSeeds[i].w;

		int xorig = xseed - 8;
		int yorig = yseed - 8;

		int shift = xorig - ((xorig >> 2) << 2);
		if (shift == 1) xorig -= 1;
		else if (shift == 2) xorig += 2;
		else if (shift == 3) xorig += 1;

// results will be slightly off unless I apply same 
// adjustment to xorig as in the GPU program (divisible by 4)

		float xc= 0.;
		float yc= 0.;
		float cc = 0;

		for (int l = 0; l < 16; l++) {
		for (int k = 0; k < 16; k++) {
			int x = xorig + k;
			int y = yorig + l;
			if (x < 0 || x >= szx) continue;
			if (y < 0 || y >= szy) continue;
			int ix = x + szx * y;
			int ss = (int) (data[ix].w+0.1);
			//printf("seed= %d, ss= %d\n", seed, ss);
			if (seed != ss) continue;
			float xx = (float) x / szx;
			float yy = (float) y / szy;
			xc += xx;
			yc += yy;
			cc++;
		}}

		bins_cpu_gpu[seed].x = xc / cc;
		bins_cpu_gpu[seed].y = yc / cc;
		bins_cpu_gpu[seed].w = (float) cc;
	}

	return bins_cpu_gpu;
}
//----------------------------------------------------------------------
float4* Centroids::computeOnCPU(int szx, int szy)
{
		int nb_seeds = newSeeds.size();
		float4* data = getVoronoiBitmap(); // from GPU

		int nb_data = szx*szy;
	
		if (bins_cpu == 0) {
			printf("computeHistogramOnCPU:: no memory allocated for bins_cpu\n");
			exit(0);
		}

		clock3->start();
		float4* bins = bins_cpu;

		for (int i=0; i < nb_seeds; i++) {
			bins[i].x = 0.;
			bins[i].y = 0.;
			bins[i].w = 0.;
		}

		for (int j=0; j < szy; j++) {
		for (int i=0; i < szx; i++) {
			float xx = (float) i / szx;
			float yy = (float) j / szy;
			int ix = i + szx*j;
			float4& d = data[ix];
			int seed = (int) (d.w+.1);
			float4& f = bins[seed];
			f.x += xx;
			f.y += yy;
			f.w++;
		}}

		for (int i=0; i < nb_seeds; i++) {
			bins[i].x /= bins[i].w;
			bins[i].y /= bins[i].w;
		}

		clock3->end();

		clock3->print();
		clock3->reset();

		return bins;
}
//----------------------------------------------------------------------
void Centroids::printComputationCPU()
{
		int count = 0;
		printf("--- CPU centroid  computation --------------------\n");
		for (int i=0; i < nb_bins; i++) {
			if (i < 20) printf("-- bins_cpu(%d) = %f, %f, %f\n", i, bins_cpu[i].x, bins_cpu[i].y, bins_cpu[i].w);
			count += (int) bins_cpu[i].w;
		}

		printf("cpu histogram: count= %d, nb_bins= %d\n", count, nb_bins);
		printf("---------------------------------------------\n");
}
//----------------------------------------------------------------------
void Centroids::printComputationGPUonCPU()
{
		int count = 0;
		printf("--- GPU simulated on GPU centroid  computation --------------------\n");
		for (int i=0; i < nb_bins; i++) {
			float4& b = bins_cpu_gpu[i];
			if (i < 20) printf("-- bins_gpu_cpu(%d) = %f, %f, %f\n", i, b.x, b.y, b.w);
			count += (int) b.w;
		}

		printf("gpu histogram: count= %d, nb bins= %d\n", count, nb_bins);
		printf("---------------------------------------------\n");
}
//----------------------------------------------------------------------
void Centroids::printComputationGPU()
{
		int count = 0;
		printf("--- GPU centroid  computation --------------------\n");
		for (int i=0; i < nb_bins; i++) {
			if (i < 20) printf("-- bins_gpu(%d) = %f, %f, %f\n", i, bins_gpu[i].x, bins_gpu[i].y, bins_gpu[i].w);
			count += (int) bins_cpu[i].w;
		}

		printf("gpu histogram: count= %d, nb bins= %d\n", count, nb_bins);
		printf("---------------------------------------------\n");
}
//----------------------------------------------------------------------
void Centroids::printSeeds(std::vector<Seed>& seeds)
{
		int count = 0;
		printf("--- Voronoi cell seeds --------------------\n");
		for (int i=0; i < seeds.size(); i++) {
			Seed& s = seeds[i];
			if (i < 20) printf("-- seed(%d) = %f, %f, %f\n", i, s.x, s.y, s.w);
		}

		printf("---------------------------------------------\n");
}
//----------------------------------------------------------------------
void Centroids::printError(std::vector<Seed>& seeds)
{
	// compute absolute histogram error

	int nb_seeds = seeds.size();
	float4 err;

	printf("relative Error in percent (x,y centroid, nb seeds in cell\n");

	if (!bins_gpu) {
		printf("histogramError: bins_gpu = 0\n");
		exit(0);
	}

	for (int i=0; i < nb_seeds; i++) {
		err.x = bins_gpu[i].x - bins_cpu[i].x;
		err.y = bins_gpu[i].y - bins_cpu[i].y;
		err.w = bins_gpu[i].w - bins_cpu[i].w;

		err.x *= 100./bins_cpu[i].x;
		err.y *= 100./bins_cpu[i].y;
		err.w *= 100./bins_cpu[i].w;

		if (i < 20) printf("rel. error (%d): x,y,z= %2.2f, %2.2f, %2.2f\n", i, err.x, err.y, err.w);
	}

	// measure error with respect to original seed location

	printf("\nrelative displacement of CPU centroid in percent (x,y) with respect to initial seed location\n");

	// Errors are just about zero!!

	for (int i=0; i < nb_seeds; i++) {
		err.x = 100.*(bins_cpu[i].x - seeds[i].x) / seeds[i].x;
		err.y = 100.*(bins_cpu[i].y - seeds[i].y) / seeds[i].y;
		if (i < 20) printf("cpu rel. error (%d): x,y= %2.2f, %2.2f\n", i, err.x, err.y);
	}

	printf("\nrelative displacement of GPU centroid in percent (x,y) with respect to initial seed location\n");

	for (int i=0; i < nb_seeds; i++) {
		err.x = 100.*(bins_gpu[i].x - seeds[i].x) / seeds[i].x;
		err.y = 100.*(bins_gpu[i].y - seeds[i].y) / seeds[i].y;
		if (i < 20) printf("gpu rel. error (%d): x,y= %2.2f, %2.2f\n", i, err.x, err.y);
	}
}
//----------------------------------------------------------------------
float4* Centroids::getVoronoiBitmap()
{
	PingPong& ping = dtransf->getCurPingPong();
	szx = ping.getWidth();
	szy = ping.getHeight();
	data_cpu = new float4 [szx*szy];

	PingPongCuda* ppc = new PingPongCuda(ping); // create PBO
	float4* data = (float4*) ppc->begin(); // 4 channels, copy FBO to PBO
		copyFromDeviceToHost(data_cpu, data, szx*szy*sizeof(float4));
	ppc->end();

	return data_cpu;
}
//----------------------------------------------------------------------
float4* Centroids::computeOnGPU()
{
	printf("enter computeOnGPU\n");
	clock_gpu->begin();
	PingPong& ping = dtransf->getCurPingPong();

	szx = ping.getWidth();
	szy = ping.getHeight();
	int sz3d = 0;
	int edge = 8;

		std::vector<Seed>& seeds = dtransf->getSeeds();
		nb_bins = seeds.size();

		int width= ping.getWidth();
		int height= ping.getHeight();

		newSeeds.clear(); // empty container (potentially expensive)

		for (int i=0; i < nb_bins; i++) {
			Seed& s =seeds[i];
			int x = (int) (width *s.x);
			int y = (int) (height*s.y);
			int z = 0;
			int seed_nb = i; // make sure it is not zero
			newSeeds.push_back(make_int4(x,y,z,seed_nb));
		}

		setNbBins(newSeeds.size());

		// above takes 0.29 ms
		//
		// the entire method is clocked at 17.9 ms!!! (GPU only takes 2 ms)
		// the entire method minus call to centroid_voronoi_7 (takes 15 ms)
	    // the time between ppc->begin() and ppc->end() is 1.8 ms (with call to voronoi)
		// the time for ppc->begin(): between 5.3 ms and 14 ms (do not know why)
		// the time for ppc->end(): 0.7 ms

#if 1
	PingPongCuda* ppc = new PingPongCuda(ping); // create PBO

	// flat texture on GPU
	printf("before ppc->begin()\n");
	clock_tmp->begin();
	float4* data = (float4*) ppc->begin(); // 4 channels, copy FBO to PBO
	clock_tmp->end();

		// copy newSeeds to the device
		// problem with this structure
		ArrayCuda1D<int4> newSeeds_h(&newSeeds[0], newSeeds.size());
		newSeeds_h.copyToDevice();

		ArrayCuda1D<float4>* bins = new ArrayCuda1D<float4>(nb_bins);
		bins->clear();
		bins->copyToDevice();
		bins_gpu = centroid_voronoi_7(data, bins->getDevicePtr(), szx, szy, nb_bins, sz3d, 
			&newSeeds[0], edge, newSeeds.size()); // centroid computation
		bins->copyToHost();

	ppc->end();
	delete ppc;
#endif

	clock_gpu->end();

	clock_tmp->print();
	clock_tmp->reset();

	clock_gpu->print();
	clock_gpu->reset();

	printf("enter computeOnGPU\n");

	return bins_gpu;
}
//----------------------------------------------------------------------
