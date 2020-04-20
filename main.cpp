// CudaTest.cpp : Defines the entry point for the application.
//
#include "SumProcessor.h"

#include "cudaWarp.h"
#include "cudaSum.h"

#include <fstream>
#include <iostream>
#include <iterator>
#include <vector>
#include <chrono>
#include <thread>

using namespace std::chrono;
using namespace AVUN;

float2 mFocalLength = { 1.0f, 1.0f }; //1, 1
float2 mPrincipalPoint = { 0.0f, 0.0f }; //0, 0
float4 mDistortion = { 0.0f, 0.0f, 0.0f, 0.0f }; // 0, 0, 0, 0

unsigned int mIterations = 1;
int mDevice = 0;

uint32_t mWidth = 5120;
uint32_t mHeight = 5120;

void TestWithoutAlloc(const std::vector<float>& input_host, std::vector<float>& output_host, size_t sizeBytes) {

	cudaError_t err = cudaSuccess;
	float4* input = NULL;
	err = cudaMalloc((void**)&input, sizeBytes);
	if (err != cudaSuccess) {
		fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n", cudaGetErrorString(err));
		return;
	}

	err = cudaMemcpy(input, input_host.data(), sizeBytes, cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		fprintf(stderr, "Failed to copy vector A from host to device (error code %s)!\n", cudaGetErrorString(err));
		return;
	}

	float4* output = NULL;
	err = cudaMalloc((void**)&output, sizeBytes);
	if (err != cudaSuccess) {
		fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n", cudaGetErrorString(err));
		return;
	}

	auto start = high_resolution_clock::now();
	for (unsigned int i = 1; i <= mIterations; ++i) {
		err = cudaWarpIntrinsic(input, output, mWidth, mHeight, mFocalLength, mPrincipalPoint, mDistortion);
		if (err != cudaSuccess) {
			fprintf(stderr, "Failed to cudaWarpIntrinsic (error code %s)!\n", cudaGetErrorString(err));
			return;
		}
	}
	auto end = high_resolution_clock::now();
	auto elapsed = duration_cast<microseconds>(end - start).count();
	std::cout << "Test without alloc " << mIterations << " iterations, time " << elapsed << "μs" << std::endl;

	err = cudaMemcpy(output_host.data(), output, sizeBytes, cudaMemcpyDeviceToHost);
	if (err != cudaSuccess) {
		fprintf(stderr, "Failed to copy vector C from device to host (error code %s)!\n", cudaGetErrorString(err));
		return;
	}

	cudaFree(input);
	cudaFree(output);

	cudaDeviceSynchronize();
}

void TestWithAlloc(const std::vector<float>& input_host, std::vector<float>& output_host, size_t sizeBytes) {

	auto start = high_resolution_clock::now();
	for (unsigned int i = 1; i <= mIterations; ++i) {
		cudaError_t err = cudaSuccess;
		float4* input = NULL;
		err = cudaMalloc((void**)&input, sizeBytes);
		if (err != cudaSuccess) {
			fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n", cudaGetErrorString(err));
			return;
		}

		err = cudaMemcpy(input, input_host.data(), sizeBytes, cudaMemcpyHostToDevice);
		if (err != cudaSuccess) {
			fprintf(stderr, "Failed to copy vector A from host to device (error code %s)!\n", cudaGetErrorString(err));
			return;
		}

		float4* output = NULL;
		err = cudaMalloc((void**)&output, sizeBytes);
		if (err != cudaSuccess) {
			fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n", cudaGetErrorString(err));
			return;
		}

		err = cudaWarpIntrinsic(input, output, mWidth, mHeight, mFocalLength, mPrincipalPoint, mDistortion);
		if (err != cudaSuccess) {
			fprintf(stderr, "Failed to cudaWarpIntrinsic (error code %s)!\n", cudaGetErrorString(err));
			return;
		}

		err = cudaMemcpy(&output_host[0], output, sizeBytes, cudaMemcpyDeviceToHost);
		if (err != cudaSuccess) {
			fprintf(stderr, "Failed to copy vector C from device to host (error code %s)!\n", cudaGetErrorString(err));
			return;
		}

		cudaFree(input);
		cudaFree(output);
	}

	auto end = high_resolution_clock::now();
	auto elapsed = duration_cast<milliseconds>(end - start).count();
	std::cout << "Test with alloc " << mIterations << " iterations, time " << elapsed << "ms" << std::endl;

	cudaDeviceSynchronize();
}

bool CheckSupport() {
	cudaDeviceProp deviceProp;
	cudaError_t err = cudaGetDeviceProperties(&deviceProp, mDevice);
	if (err != cudaSuccess) {
		fprintf(stderr, "Failed to cudaGetDeviceProperties for device %d (error code %s)!\n", mDevice, cudaGetErrorString(err));
		return false;
	}

	if (!deviceProp.canMapHostMemory) {
		fprintf(stderr, "Device %d does not support mapping CPU host memory!\n", mDevice);
		return false;
	}

	// Set flag to enable zero copy access
	err = cudaSetDeviceFlags(cudaDeviceMapHost);
	if (err != cudaSuccess) {
		fprintf(stderr, "Failed to set cudaSetDeviceFlags (error code %s)!\n", cudaGetErrorString(err));
		return false;
	}

	return true;
}

void TestWithAllocZeroCopy(const std::vector<float>& input_host, std::vector<float>& output_host, size_t sizeBytes) {
	cudaError_t err = cudaSuccess;

	// Host Arrays (CPU pointers)
	float4* h_in = NULL;
	float4* h_out = NULL;

	// Allocate host memory using CUDA allocation calls
	err = cudaHostAlloc((void**)&h_in, sizeBytes, cudaHostAllocMapped);
	if (err != cudaSuccess) {
		fprintf(stderr, "Failed to allocate device input memory (error code %s)!\n", cudaGetErrorString(err));
		return;
	}

	err = cudaHostAlloc((void**)&h_out, sizeBytes, cudaHostAllocMapped);
	if (err != cudaSuccess) {
		fprintf(stderr, "Failed to allocate device output memory (error code %s)!\n", cudaGetErrorString(err));
		return;
	}

	// Copy input bytes, just to operate with prepared data
	err = cudaMemcpy(h_in, input_host.data(), sizeBytes, cudaMemcpyHostToHost);
	if (err != cudaSuccess) {
		fprintf(stderr, "Failed to cudaMemcpy input memory (error code %s)!\n", cudaGetErrorString(err));
		return;
	}

	auto start = high_resolution_clock::now();
	for (unsigned int i = 1; i <= mIterations; ++i) {

		// Device arrays (CPU pointers)
		float4* d_in = nullptr;
		// Get device pointer from host memory. No allocation or memcpy
		err = cudaHostGetDevicePointer((void**)&d_in, (void*)h_in, 0);
		if (err != cudaSuccess) {
			fprintf(stderr, "Failed to cudaHostGetDevicePointer input (error code %s)!\n", cudaGetErrorString(err));
			return;
		}

		float4* d_out = nullptr;
		err = cudaHostGetDevicePointer((void**)&d_out, (void*)h_out, 0);
		if (err != cudaSuccess) {
			fprintf(stderr, "Failed to cudaHostGetDevicePointer output (error code %s)!\n", cudaGetErrorString(err));
			return;
		}
		
		err = cudaWarpIntrinsic(d_in, d_out, mWidth, mHeight, mFocalLength, mPrincipalPoint, mDistortion);
		if (err != cudaSuccess) {
			fprintf(stderr, "Failed to cudaWarpIntrinsic (error code %s)!\n", cudaGetErrorString(err));
			return;
		}
	}

	auto end = high_resolution_clock::now();
	auto elapsed = duration_cast<microseconds>(end - start).count();
	std::cout << "Test zero copy " << mIterations << " iterations, time " << elapsed << "μs" << std::endl;
	
	cudaDeviceSynchronize();

	err = cudaMemcpy(&output_host[0], h_out, sizeBytes, cudaMemcpyHostToHost);
	if (err != cudaSuccess) {
		fprintf(stderr, "Failed to cudaMemcpy input memory (error code %s)!\n", cudaGetErrorString(err));
		return;
	}

	cudaFreeHost(h_in);
	cudaFreeHost(h_out);
}

void TestWithAllocZeroCopy(const std::vector<uint16_t>& input_host, std::vector<uint16_t>& output_host, size_t sizeBytes) {
	cudaError_t err = cudaSuccess;

	// Host Arrays (CPU pointers)
	uint8_t* h_in = NULL;
	uint8_t* h_out = NULL;

	// Allocate host memory using CUDA allocation calls
	err = cudaHostAlloc((void**)&h_in, sizeBytes, cudaHostAllocMapped);
	if (err != cudaSuccess) {
		fprintf(stderr, "Failed to allocate device input memory (error code %s)!\n", cudaGetErrorString(err));
		return;
	}

	// Copy input bytes, just to operate with prepared data
	err = cudaMemcpy(h_in, input_host.data(), sizeBytes, cudaMemcpyHostToHost);
	if (err != cudaSuccess) {
		fprintf(stderr, "Failed to cudaMemcpy input memory (error code %s)!\n", cudaGetErrorString(err));
		return;
	}

	err = cudaHostAlloc((void**)&h_out, sizeBytes, cudaHostAllocMapped);
	if (err != cudaSuccess) {
		fprintf(stderr, "Failed to allocate device output memory (error code %s)!\n", cudaGetErrorString(err));
		return;
	}

	auto start = high_resolution_clock::now();
	for (unsigned int i = 1; i <= mIterations; ++i) {
		// Device arrays (CPU pointers)
		ushort1* d_in = nullptr;
		// Get device pointer from host memory. No allocation or memcpy
		err = cudaHostGetDevicePointer((void**)&d_in, (void*)h_in, 0);
		if (err != cudaSuccess) {
			fprintf(stderr, "Failed to cudaHostGetDevicePointer input (error code %s)!\n", cudaGetErrorString(err));
			return;
		}

		ushort1* d_out = nullptr;
		err = cudaHostGetDevicePointer((void**)&d_out, (void*)h_out, 0);
		if (err != cudaSuccess) {
			fprintf(stderr, "Failed to cudaHostGetDevicePointer output (error code %s)!\n", cudaGetErrorString(err));
			return;
		}

		err = cudaWarpIntrinsic(d_in, d_out, mWidth, mHeight, mFocalLength, mPrincipalPoint, mDistortion);
		if (err != cudaSuccess) {
			fprintf(stderr, "Failed to cudaWarpIntrinsic (error code %s)!\n", cudaGetErrorString(err));
			return;
		}
	}

	err = cudaDeviceSynchronize();
	//err = cudaStreamSynchronize(0);
	if (err != cudaSuccess) {
		fprintf(stderr, "Failed to cudaDeviceSynchronize (error code %s)!\n", cudaGetErrorString(err));
		return;
	}

	auto end = high_resolution_clock::now();
	auto elapsed = duration_cast<milliseconds>(end - start).count();
	std::cout << "Test distorsion " << mIterations << " iterations, time " << elapsed << "ms" << std::endl;

	err = cudaMemcpy(&output_host[0], h_out, sizeBytes, cudaMemcpyHostToHost);
	if (err != cudaSuccess) {
		fprintf(stderr, "Failed to cudaMemcpy input memory (error code %s)!\n", cudaGetErrorString(err));
		return;
	}

	cudaFreeHost(h_in);
	cudaFreeHost(h_out);
}

void TestSum(const std::vector<uint16_t>& input_host, std::vector<uint16_t>& output_host, size_t sizeBytes) {
	cudaError_t err = cudaSuccess;

	// Host Arrays (CPU pointers)
	uint8_t* h_in = NULL;
	uint8_t* h_out = NULL;

	// Allocate host memory using CUDA allocation calls
	err = cudaHostAlloc((void**)&h_in, sizeBytes, cudaHostAllocMapped);
	if (err != cudaSuccess) {
		fprintf(stderr, "Failed to allocate device input memory (error code %s)!\n", cudaGetErrorString(err));
		return;
	}

	// Copy input bytes, just to operate with prepared data
	err = cudaMemcpy(h_in, input_host.data(), sizeBytes, cudaMemcpyHostToHost);
	if (err != cudaSuccess) {
		fprintf(stderr, "Failed to cudaMemcpy input memory (error code %s)!\n", cudaGetErrorString(err));
		return;
	}

	err = cudaHostAlloc((void**)&h_out, sizeBytes, cudaHostAllocMapped);
	if (err != cudaSuccess) {
		fprintf(stderr, "Failed to allocate device output memory (error code %s)!\n", cudaGetErrorString(err));
		return;
	}

	// Device arrays (CPU pointers)
	uint16_t* d_in = nullptr;
	// Get device pointer from host memory. No allocation or memcpy
	err = cudaHostGetDevicePointer((void**)&d_in, (void*)h_in, 0);
	if (err != cudaSuccess) {
		fprintf(stderr, "Failed to cudaHostGetDevicePointer input (error code %s)!\n", cudaGetErrorString(err));
		return;
	}

	uint16_t* d_out = nullptr;
	err = cudaHostGetDevicePointer((void**)&d_out, (void*)h_out, 0);
	if (err != cudaSuccess) {
		fprintf(stderr, "Failed to cudaHostGetDevicePointer output (error code %s)!\n", cudaGetErrorString(err));
		return;
	}

	auto start = high_resolution_clock::now();
	for (int i = 0; i < mIterations; i++) {
		err = cudaWarpSum(d_out, d_in, sizeBytes / 2);
		if (err != cudaSuccess) {
			fprintf(stderr, "Failed to cudaWarpIntrinsic (error code %s)!\n", cudaGetErrorString(err));
			return;
		}
	}

	err = cudaDeviceSynchronize();
	//err = cudaStreamSynchronize(0);
	if (err != cudaSuccess) {
		fprintf(stderr, "Failed to cudaDeviceSynchronize (error code %s)!\n", cudaGetErrorString(err));
		return;
	}

	auto end = high_resolution_clock::now();
	auto elapsed = duration_cast<milliseconds>(end - start).count();
	std::cout << "Test cudaWarpSum, time " << elapsed << "ms" << std::endl;

	err = cudaMemcpy(&output_host[0], h_out, sizeBytes, cudaMemcpyHostToHost);
	if (err != cudaSuccess) {
		fprintf(stderr, "Failed to cudaMemcpy input memory (error code %s)!\n", cudaGetErrorString(err));
		return;
	}

	cudaFreeHost(h_in);
	cudaFreeHost(h_out);
}

void TestSumStd(const std::vector<uint16_t>& input_host, std::vector<uint16_t>& output_host, size_t sizeBytes) {
	
	output_host.assign(input_host.begin(), input_host.end());
	
	std::vector<std::unique_ptr<SumProcessor>> mSumProcessors;

	auto processorCount = std::thread::hardware_concurrency();

	for (int i = 0; i < processorCount; ++i) {
		mSumProcessors.emplace_back(std::make_unique<SumProcessor>(i));
	}

	auto start = high_resolution_clock::now();

	auto src = (uint16_t*)input_host.data();
	auto dst = (uint16_t*)output_host.data();

	auto sumDataSize = sizeBytes / 2;
	auto blockSize = sumDataSize / processorCount;

	size_t offset = 0;
	for (auto& processor : mSumProcessors) {
		processor->Process(src + offset, dst + offset, blockSize);
		offset += blockSize;
	}

	for (auto& processor : mSumProcessors) {
		processor->WaitForComplete();
	}

	auto end = high_resolution_clock::now();
	auto elapsed = duration_cast<milliseconds>(end - start).count();
	std::cout << "Test sumStd, time " << elapsed << "ms" << std::endl;
}

void TestSumAndDist(const std::vector<uint16_t>& input_host, std::vector<uint16_t>& output_host, size_t sizeBytes) {
	cudaError_t err = cudaSuccess;

	// Host Arrays (CPU pointers)
	uint8_t* h_in = NULL;
	uint8_t* h_out = NULL;

	// Allocate host memory using CUDA allocation calls
	err = cudaHostAlloc((void**)&h_in, sizeBytes, cudaHostAllocMapped);
	if (err != cudaSuccess) {
		fprintf(stderr, "Failed to allocate device input memory (error code %s)!\n", cudaGetErrorString(err));
		return;
	}

	// Copy input bytes, just to operate with prepared data
	err = cudaMemcpy(h_in, input_host.data(), sizeBytes, cudaMemcpyHostToHost);
	if (err != cudaSuccess) {
		fprintf(stderr, "Failed to cudaMemcpy input memory (error code %s)!\n", cudaGetErrorString(err));
		return;
	}

	err = cudaHostAlloc((void**)&h_out, sizeBytes, cudaHostAllocMapped);
	if (err != cudaSuccess) {
		fprintf(stderr, "Failed to allocate device output memory (error code %s)!\n", cudaGetErrorString(err));
		return;
	}

	// Host Arrays (CPU pointers)
	uint8_t* h_sum_out = NULL;
	err = cudaHostAlloc((void**)&h_sum_out, sizeBytes, cudaHostAllocMapped);
	if (err != cudaSuccess) {
		fprintf(stderr, "Failed to allocate device output memory (error code %s)!\n", cudaGetErrorString(err));
		return;
	}

	auto start = high_resolution_clock::now();
	for (int i = 0; i < mIterations; i++) {
		{
			// Device arrays (CPU pointers)
			ushort1* d_in = nullptr;
			// Get device pointer from host memory. No allocation or memcpy
			err = cudaHostGetDevicePointer((void**)&d_in, (void*)h_in, 0);
			if (err != cudaSuccess) {
				fprintf(stderr, "Failed to cudaHostGetDevicePointer input (error code %s)!\n", cudaGetErrorString(err));
				return;
			}

			ushort1* d_out = nullptr;
			err = cudaHostGetDevicePointer((void**)&d_out, (void*)h_out, 0);
			if (err != cudaSuccess) {
				fprintf(stderr, "Failed to cudaHostGetDevicePointer output (error code %s)!\n", cudaGetErrorString(err));
				return;
			}

			err = cudaWarpIntrinsic(d_in, d_out, mWidth, mHeight, mFocalLength, mPrincipalPoint, mDistortion);
			if (err != cudaSuccess) {
				fprintf(stderr, "Failed to cudaWarpIntrinsic (error code %s)!\n", cudaGetErrorString(err));
				return;
			}
			//err = cudaDeviceSynchronize();
		}

		// Device arrays (CPU pointers)
		uint16_t* d_in = nullptr;
		// Get device pointer from host memory. No allocation or memcpy
		err = cudaHostGetDevicePointer((void**)&d_in, (void*)h_out, 0);
		if (err != cudaSuccess) {
			fprintf(stderr, "Failed to cudaHostGetDevicePointer input (error code %s)!\n", cudaGetErrorString(err));
			return;
		}

		uint16_t* d_out = nullptr;
		err = cudaHostGetDevicePointer((void**)&d_out, (void*)h_sum_out, 0);
		if (err != cudaSuccess) {
			fprintf(stderr, "Failed to cudaHostGetDevicePointer output (error code %s)!\n", cudaGetErrorString(err));
			return;
		}

		err = cudaWarpSum(d_out, d_in, sizeBytes / 2);
		if (err != cudaSuccess) {
			fprintf(stderr, "Failed to cudaWarpIntrinsic (error code %s)!\n", cudaGetErrorString(err));
			return;
		}
		//err = cudaDeviceSynchronize();
	}
	err = cudaDeviceSynchronize();
	if (err != cudaSuccess) {
		fprintf(stderr, "Failed to cudaDeviceSynchronize (error code %s)!\n", cudaGetErrorString(err));
		return;
	}

	auto end = high_resolution_clock::now();
	auto elapsed = duration_cast<milliseconds>(end - start).count();
	std::cout << "Test cudaHostAlloc, time " << elapsed << "ms" << std::endl;
}

int main(int argc, char** argv)
{	
	if (argc < 2) {
		std::cout << "usage: " << "-i <iterations count> -f <file path> -o <output file path> -w <width> -h <height> -d <device>" << std::endl;
	}

	std::string filePath = "frame.bin";
	std::string outFilePath = "out.bin";

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
		else if (strcmp(argv[i], "-d") == 0) {
			mDevice = std::atoi(argv[i + 1]);
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

	/*
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

	TestWithoutAlloc(input_host, output_host, sizeBytes);

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

	TestWithAlloc(input_host, output_host, sizeBytes);
	*/

	std::vector<uint16_t> output_host(size, 0);
	auto sizeBytes = size * sizeof(uint16_t);

	if (!CheckSupport()) {
		std::cout << "Zero copy support is not found!" << std::endl;
		return -1;
	}

	TestWithAllocZeroCopy(input, output_host, sizeBytes);
	TestSum(input, output_host, sizeBytes);
	TestSumStd(input, output_host, sizeBytes);
	
	TestSumAndDist(input, output_host, sizeBytes);

	if (!outFilePath.empty()) {
		std::ofstream os(outFilePath, std::ios::out | std::ofstream::binary);
		os.write((char*)output_host.data(), output_host.size() * 2);
		os.flush();
		os.close();
	}

	return 0;
}
