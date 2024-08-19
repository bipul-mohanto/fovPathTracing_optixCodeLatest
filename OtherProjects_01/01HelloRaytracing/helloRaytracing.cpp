#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stack_size.h>
#include <optix_stubs.h>

#include <sutil/CUDAOutputBuffer.h>
#include <sutil/Exception.h>
#include <sutil/sutil.h>
#include <sampleConfig.h>

#include "helloRaytracing.h"

#include <iomanip>
#include <iostream>
#include <string>

void initOptix() {
	cudaFree(0);
	int numDevices;
	cudaGetDeviceCount(&numDevices);
	if (numDevices == 0)
		throw std::runtime_error("no Cuda capable devices found!");

	OPTIX_CHECK(optixInit());
}

int main(int argc, char* argv[])
{
	initOptix();

	return 0;
}