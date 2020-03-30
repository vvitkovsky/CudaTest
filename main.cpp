// CudaTest.cpp : Defines the entry point for the application.
//
#include "cudaWarp.h"
#include <fstream>
#include <iostream>
#include <iterator>
#include <vector>
#include <chrono>

using namespace std::chrono;

float2 mFocalLength = { 1.0f, 1.0f }; //1, 1
float2 mPrincipalPoint = { 0.0f, 0.0f }; //0, 0
float4 mDistortion = { 0.0f, 0.0f, 0.0f, 0.0f }; // 0, 0, 0, 0

unsigned int mIterations = 100;

uint32_t mWidth = 5120;
uint32_t mHeight = 5120;

void TestWithoutAlloc(const std::vector<float>& input_host, std::vector<float>& output_host, size_t sizeBytes) {

	cudaError_t err = cudaSuccess;
	float4* input = NULL;
	err = cudaMalloc((void**)&input, sizeBytes);
	if (err != cudaSuccess) {
		fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	err = cudaMemcpy(input, input_host.data(), sizeBytes, cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		fprintf(stderr, "Failed to copy vector A from host to device (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	float4* output = NULL;
	err = cudaMalloc((void**)&output, sizeBytes);
	if (err != cudaSuccess) {
		fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	auto start = high_resolution_clock::now();
	for (unsigned int i = 1; i <= mIterations; ++i) {
		err = cudaWarpIntrinsic(input, output, mWidth, mHeight, mFocalLength, mPrincipalPoint, mDistortion);
		if (err != cudaSuccess) {
			fprintf(stderr, "Failed to cudaWarpIntrinsic (error code %s)!\n", cudaGetErrorString(err));
			exit(EXIT_FAILURE);
		}
	}
	auto end = high_resolution_clock::now();
	auto elapsed = duration_cast<microseconds>(end - start).count();
	std::cout << "Test without alloc " << mIterations << " iterations, time " << elapsed << "μs" << std::endl;

	err = cudaMemcpy(output_host.data(), output, sizeBytes, cudaMemcpyDeviceToHost);
	if (err != cudaSuccess) {
		fprintf(stderr, "Failed to copy vector C from device to host (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	cudaFree(input);
	cudaFree(output);
}

void TestWithAlloc(const std::vector<float>& input_host, std::vector<float>& output_host, size_t sizeBytes) {

	auto start = high_resolution_clock::now();
	for (unsigned int i = 1; i <= mIterations; ++i) {
		cudaError_t err = cudaSuccess;
		float4* input = NULL;
		err = cudaMalloc((void**)&input, sizeBytes);
		if (err != cudaSuccess) {
			fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n", cudaGetErrorString(err));
			exit(EXIT_FAILURE);
		}

		err = cudaMemcpy(input, &input_host[0], sizeBytes, cudaMemcpyHostToDevice);
		if (err != cudaSuccess) {
			fprintf(stderr, "Failed to copy vector A from host to device (error code %s)!\n", cudaGetErrorString(err));
			exit(EXIT_FAILURE);
		}

		float4* output = NULL;
		err = cudaMalloc((void**)&output, sizeBytes);
		if (err != cudaSuccess) {
			fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n", cudaGetErrorString(err));
			exit(EXIT_FAILURE);
		}

		err = cudaWarpIntrinsic(input, output, mWidth, mHeight, mFocalLength, mPrincipalPoint, mDistortion);
		if (err != cudaSuccess) {
			fprintf(stderr, "Failed to cudaWarpIntrinsic (error code %s)!\n", cudaGetErrorString(err));
			exit(EXIT_FAILURE);
		}

		err = cudaMemcpy(&output_host[0], output, sizeBytes, cudaMemcpyDeviceToHost);
		if (err != cudaSuccess) {
			fprintf(stderr, "Failed to copy vector C from device to host (error code %s)!\n", cudaGetErrorString(err));
			exit(EXIT_FAILURE);
		}

		cudaFree(input);
		cudaFree(output);
	}

	auto end = high_resolution_clock::now();
	auto elapsed = duration_cast<milliseconds>(end - start).count();
	std::cout << "Test with alloc " << mIterations << " iterations, time " << elapsed << "ms" << std::endl;
}

void TestWithAllocZeroCopy(const std::vector<float>& input_host, std::vector<float>& output_host, size_t sizeBytes) {

	cudaError_t err = cudaSuccess;

	// Set flag to enable zero copy access
	err = cudaSetDeviceFlags(cudaDeviceMapHost);
	if (err != cudaSuccess) {
		fprintf(stderr, "Failed to set cudaSetDeviceFlags (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	// Host Arrays (CPU pointers)
	float4* h_in = NULL;
	float4* h_out = NULL;

	// Allocate host memory using CUDA allocation calls
	err = cudaHostAlloc((void**)&h_in, sizeBytes, cudaHostAllocMapped);
	if (err != cudaSuccess) {
		fprintf(stderr, "Failed to allocate device input memory (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	err = cudaHostAlloc((void**)&h_out, sizeBytes, cudaHostAllocMapped);
	if (err != cudaSuccess) {
		fprintf(stderr, "Failed to allocate device output memory (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	// Copy input bytes, just to operate with prepared data
	memcpy(h_in, input_host.data(), sizeBytes);

	auto start = high_resolution_clock::now();
	for (unsigned int i = 1; i <= mIterations; ++i) {

		// Device arrays (CPU pointers)
		float4* d_out, * d_in;
		// Get device pointer from host memory. No allocation or memcpy
		err = cudaHostGetDevicePointer((void**)&d_in, (void*)h_in, 0);
		if (err != cudaSuccess) {
			fprintf(stderr, "Failed to cudaHostGetDevicePointer input (error code %s)!\n", cudaGetErrorString(err));
			exit(EXIT_FAILURE);
		}

		err = cudaHostGetDevicePointer((void**)&d_out, (void*)h_out, 0);
		if (err != cudaSuccess) {
			fprintf(stderr, "Failed to cudaHostGetDevicePointer output (error code %s)!\n", cudaGetErrorString(err));
			exit(EXIT_FAILURE);
		}
		
		err = cudaWarpIntrinsic(d_in, d_out, mWidth, mHeight, mFocalLength, mPrincipalPoint, mDistortion);
		if (err != cudaSuccess) {
			fprintf(stderr, "Failed to cudaWarpIntrinsic (error code %s)!\n", cudaGetErrorString(err));
			exit(EXIT_FAILURE);
		}

		memcpy(&output_host[0], h_out, sizeBytes);

		cudaFree(d_in);
		cudaFree(d_out);
	}

	auto end = high_resolution_clock::now();
	auto elapsed = duration_cast<milliseconds>(end - start).count();
	std::cout << "Test with alloc " << mIterations << " iterations, time " << elapsed << "ms" << std::endl;
	
	cudaFree(h_in);
	cudaFree(h_out);
}

int main(int argc, char** argv)
{	
	if (argc < 2) {
		std::cout << "usage: " << "-i <iterations count> -f <file path> -o <output file path> -w <width> -h <height>" << std::endl;
	}

	std::string filePath = "frame.bin";
	std::string outFilePath;

	for (int i = 1; i < argc - 1; i++) {
		if (strcmp(argv[i], "-i") == 0) {
			mIterations = std::atoi(argv[i + 1]);
		}
		else if (strcmp(argv[i], "-f") == 0) {
			filePath = argv[i + 1];
		}
		else if (strcmp(argv[i], "-o") == 0) {
			outFilePath = argv[i + 1];
		}
		else if (strcmp(argv[i], "-w") == 0) {
			mWidth = std::atoi(argv[i + 1]);
		}
		else if (strcmp(argv[i], "-h") == 0) {
			mHeight = std::atoi(argv[i + 1]);
		}
	}

	std::ifstream is(filePath, std::ios::binary);
	is.seekg(0, std::ios::end);
	size_t filesize = is.tellg();
	is.seekg(0, std::ios::beg);

	auto size = filesize / sizeof(uint16_t);	
	std::vector<uint16_t> input(size, 0);
	is.read((char*)input.data(), filesize);
	is.close();

	std::vector<float> input_host(size * 4, 0);
	std::vector<float> output_host(size * 4, 0);

	size_t pos = 0;
	for (uint16_t val : input) {
		input_host[pos] = val;
		input_host[pos + 1u] = val;
		input_host[pos + 2u] = val;
		input_host[pos + 3u] = val;
		pos += 4;
	}

	auto sizeBytes = size * sizeof(float) * 4;

	TestWithAllocZeroCopy(input_host, output_host, sizeBytes);

	//TestWithoutAlloc(input_host, output_host, sizeBytes);

	if (!outFilePath.empty()) {
		std::vector<uint16_t> output;
		output.reserve(size);

		int j = 0;
		for (auto i = 0; i < output_host.size(); ++i) {
			output.push_back(output_host[i]);
			i += 3;
		}

		std::ofstream os(outFilePath, std::ios::out | std::ofstream::binary);
		os.write((char*)output.data(), output.size() * 2);
		os.flush();
		os.close();
	}

	//TestWithAlloc(input_host, output_host, sizeBytes);

	return 0;
}
