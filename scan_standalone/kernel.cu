
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include "utils.h"
#include "timer.h"
#include <stdlib.h>
#include <time.h>

#define MAX_BLOCK_SZ 1024

__global__
void gpu_sum_scan_naive(unsigned int* const d_out,
	const unsigned int* const d_in,
	const size_t numElems)
{
	// Using naive scan where each thread calculates a separate partial sum
	// Step complexity is still O(n) as the last thread will calculate the global sum

	unsigned int d_hist_idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (d_hist_idx == 0 || d_hist_idx >= numElems)
	{
		return;
	}
	unsigned int cdf_val = 0;
	for (int i = 0; i < d_hist_idx; ++i)
	{
		cdf_val = cdf_val + d_in[i];
	}
	d_out[d_hist_idx] = cdf_val;
}

__global__
void gpu_sum_scan_blelloch(unsigned int* const d_out,
	const unsigned int* const d_in,
	const size_t numElems)
{
	unsigned int glbl_t_idx = blockDim.x * blockIdx.x + threadIdx.x;

	// Copy d_in to d_out
	// This has to be modified when we decide to only use numElems/2 threads instead of numElems
	if (glbl_t_idx < numElems)
		d_out[glbl_t_idx] = d_in[glbl_t_idx];
	else
		return;

	if (glbl_t_idx >= (numElems / 2))
		return;

	__syncthreads();

	// Reduce step
	unsigned int max_steps = 0;
	while ((1 << max_steps) < numElems)
		max_steps = max_steps + 1;

	unsigned int r_idx = 0;
	unsigned int l_idx = 0;
	unsigned int sum = 0; // global sum can be passed to host if needed
	unsigned int t_active = 0;
	for (int s = 0; s < max_steps; ++s)
	{
		t_active = 0;

		// calculate necessary indexes
		// right index must be (t+1) * 2^(s+1)) - 1
		r_idx = ((glbl_t_idx + 1) * (1 << (s + 1))) - 1;
		if (r_idx >= 0 && r_idx < numElems)
			t_active = 1;

		if (t_active)
		{
			// left index must be r_idx - 2^s
			l_idx = r_idx - (1 << s);

			// do the actual add operation
			sum = d_out[l_idx] + d_out[r_idx];
		}
		__syncthreads();

		if (t_active)
			d_out[r_idx] = sum;
		__syncthreads();
	}

	// Reset last element to operation's identity (sum, 0)
	if (glbl_t_idx == 0)
		d_out[r_idx] = 0;

	__syncthreads();

	// Downsweep step
	// FIX INDICES
	for (int s = max_steps - 1; s >= 0; --s)
	{
		// calculate necessary indexes
		// right index must be (t+1) * 2^(s+1)) - 1
		r_idx = ((glbl_t_idx + 1) * (1 << (s + 1))) - 1;
		if (r_idx >= 0 && r_idx < numElems)
		{
			t_active = 1;
		}

		unsigned int r_cpy = 0;
		unsigned int lr_sum = 0;
		if (t_active)
		{
			// left index must be r_idx - 2^s
			l_idx = r_idx - (1 << s);

			// do the downsweep operation
			r_cpy = d_out[r_idx];
			lr_sum = d_out[l_idx] + d_out[r_idx];
		}
		__syncthreads();

		if (t_active)
		{
			d_out[l_idx] = r_cpy;
			d_out[r_idx] = lr_sum;
		}
		__syncthreads();

	}
}

void sum_scan_naive(unsigned int* const d_out,
	const unsigned int* const d_in,
	const size_t numElems)
{
	unsigned int blockSz = MAX_BLOCK_SZ;
	unsigned int gridSz = (unsigned int)ceil(float(numElems) / float(MAX_BLOCK_SZ));
	checkCudaErrors(cudaMemset(d_out, 0, numElems * sizeof(unsigned int)));
	gpu_sum_scan_naive << <gridSz, blockSz >> >(d_out, d_in, numElems);
	//gpu_sum_scan_blelloch << <gridSz, blockSz >> >(d_out, d_in, numElems);
}

void sum_scan_blelloch(unsigned int* const d_out,
	const unsigned int* const d_in,
	const size_t numElems)
{
	// Zero out d_out
	checkCudaErrors(cudaMemset(d_out, 0, numElems * sizeof(unsigned int)));

	// Set up number of threads and blocks
	// num of blocks must be ceiling of log of numElems
	// if numElems is not power of two, the remainder will still need a whole block
	unsigned int blockSz = MAX_BLOCK_SZ;
	unsigned int gridSz = (unsigned int)ceil(float(numElems) / float(MAX_BLOCK_SZ));
	checkCudaErrors(cudaMemset(d_out, 0, numElems * sizeof(unsigned int)));
	//gpu_sum_scan_naive << <gridSz, blockSz >> >(d_out, d_in, numElems);
	gpu_sum_scan_blelloch << <gridSz, blockSz >> >(d_out, d_in, numElems);
}

void cpu_sum_scan(unsigned int* const h_out,
	const unsigned int* const h_in,
	const size_t numElems)
{
	unsigned int run_sum = 0;
	for (int i = 0; i < numElems; ++i)
	{
		h_out[i] = run_sum;
		run_sum = run_sum + h_in[i];
	}
}

int main()
{
	srand(time(NULL));
	unsigned int h_in_len = 0;
	//unsigned int h_in_len = (1 << 26);
	//unsigned int h_in_len = 1920*1080;
	for (int k = 0; k < 12; ++k)
	{
		//GpuTimer timer;
		std::clock_t start;
		double duration;

		h_in_len = (1 << k);
		//h_in_len = 220480;
		std::cout << "h_in size: " << h_in_len << std::endl;

		unsigned int* h_in = new unsigned int[h_in_len];
		for (int i = 0; i < h_in_len; ++i)
		{
			//h_in[i] = rand() % 2;
			h_in[i] = i;
		}
		unsigned int* h_out_naive = new unsigned int[h_in_len];
		unsigned int* h_out_blelloch = new unsigned int[h_in_len];
		unsigned int* d_in;
		checkCudaErrors(cudaMalloc(&d_in, sizeof(unsigned int) * h_in_len));
		checkCudaErrors(cudaMemcpy(d_in, h_in, sizeof(unsigned int) * h_in_len, cudaMemcpyHostToDevice));
		unsigned int* d_out_naive;
		unsigned int* d_out_blelloch;
		checkCudaErrors(cudaMalloc(&d_out_naive, sizeof(unsigned int) * h_in_len));
		checkCudaErrors(cudaMalloc(&d_out_blelloch, sizeof(unsigned int) * h_in_len));
		//timer.Start();
		//sum_scan_naive(d_out_naive, d_in, h_in_len);
		start = std::clock();
		cpu_sum_scan(h_out_naive, h_in, h_in_len);
		duration = (std::clock() - start) / (double)CLOCKS_PER_SEC;
		std::cout << "CPU time: " << duration << std::endl;
		//std::cout << "Naive time: " << timer.Elapsed() << std::endl;
		//timer.Start();
		start = std::clock();
		sum_scan_blelloch(d_out_blelloch, d_in, h_in_len);
		duration = (std::clock() - start) / (double)CLOCKS_PER_SEC;
		std::cout << "GPU time: " << duration << std::endl;
		//timer.Stop();
		//std::cout << "Blelloch time: " << timer.Elapsed() << std::endl
		//checkCudaErrors(cudaMemcpy(h_out_naive, d_out_naive, sizeof(unsigned int) * h_in_len, cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy(h_out_blelloch, d_out_blelloch, sizeof(unsigned int) * h_in_len, cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaFree(d_out_blelloch));
		checkCudaErrors(cudaFree(d_out_naive));
		checkCudaErrors(cudaFree(d_in));

		//for (int i = 0; i < h_in_len; ++i)
		//{
		//	std::cout << h_in[i] << ", ";
		//}
		//std::cout << std::endl;
		//for (int i = 0; i < h_in_len; ++i)
		//{
		//	std::cout << h_out_naive[i] << ", ";
		//}
		//std::cout << std::endl;
		//for (int i = 0; i < h_in_len; ++i)
		//{
		//	std::cout << h_out_blelloch[i] << ", ";
		//}
		//std::cout << std::endl;
		bool match = true;
		int index_diff = 0;
		for (int i = 0; i < h_in_len; ++i)
		{
			if (h_out_naive[i] != h_out_blelloch[i])
			{
				match = false;
				index_diff = i;
				break;
			}
		}
		std::cout << "Match: " << match << std::endl;
		if (!match)
		{
			std::cout << "Difference in index: " << index_diff << std::endl;
			std::cout << "Naive: " << h_out_naive[index_diff] << std::endl;
			std::cout << "Blelloch: " << h_out_blelloch[index_diff] << std::endl;
			int window_sz = 10;

			std::cout << "Contents: " << std::endl;
			std::cout << "Naive: ";
			for (int i = -(window_sz / 2); i < (window_sz / 2); ++i)
			{
				std::cout << h_out_naive[index_diff + i] << ", ";
			}
			std::cout << std::endl;
			std::cout << "Blelloch: ";
			for (int i = -(window_sz / 2); i < (window_sz / 2); ++i)
			{
				std::cout << h_out_blelloch[index_diff + i] << ", ";
			}
			std::cout << std::endl;
		}

		delete[] h_in;
		delete[] h_out_naive;
		delete[] h_out_blelloch;

		std::cout << std::endl;
	}
}
