// ======================================================================== //
// Copyright 2018-2019 Ingo Wald                                            //
//                                                                          //
// Licensed under the Apache License, Version 2.0 (the "License");          //
// you may not use this file except in compliance with the License.         //
// You may obtain a copy of the License at                                  //
//                                                                          //
//     http://www.apache.org/licenses/LICENSE-2.0                           //
//                                                                          //
// Unless required by applicable law or agreed to in writing, software      //
// distributed under the License is distributed on an "AS IS" BASIS,        //
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. //
// See the License for the specific language governing permissions and      //
// limitations under the License.                                           //
// ======================================================================== //

#include "toneMap.h"

#include <cuda_runtime_api.h>
#include <sutil/vec_math.h>

#include <math.h>

inline __device__ float4 sqrt(float4 f)
{
    return make_float4(sqrtf(f.x),
        sqrtf(f.y),
        sqrtf(f.z),
        sqrtf(f.w));
}
inline __device__ float  clampf(float f) { return min(1.f, max(0.f, f)); }
inline __device__ float4 clamp(float4 f)
{
    return make_float4(clampf(f.x),
        clampf(f.y),
        clampf(f.z),
        clampf(f.w));
}

/*! runs a cuda kernel that performs gamma correction and float4-to-rgba conversion */
__global__ void computeFinalPixelColorsKernel(uint32_t* finalColorBuffer, float4* denoisedBuffer, int2 size)
{
    int pixelX = threadIdx.x + blockIdx.x * blockDim.x;
    int pixelY = threadIdx.y + blockIdx.y * blockDim.y;
    if (pixelX >= size.x) return;
    if (pixelY >= size.y) return;

    int pixelID = pixelX + size.x * pixelY;

    float4 f4 = denoisedBuffer[pixelID];
    f4 = clamp(sqrt(f4));
    uint32_t rgba = 0;
    rgba |= (uint32_t)(f4.x * 255.9f) << 0;
    rgba |= (uint32_t)(f4.y * 255.9f) << 8;
    rgba |= (uint32_t)(f4.z * 255.9f) << 16;
    rgba |= (uint32_t)255 << 24;
    finalColorBuffer[pixelID] = rgba;
}

extern "C" __declspec(dllexport) void computeFinalPixelColors(const int2& fbSize, float4* denoisedBuffer, uint32_t* finalColorBuffer)
{
    
    int2 blockSize = make_int2(32);
    int2 numBlocks = make_int2(make_float2((fbSize + blockSize - 1)) / make_float2(blockSize));// divRoundUp(fbSize, blockSize);
    computeFinalPixelColorsKernel
        <<<dim3(numBlocks.x, numBlocks.y), dim3(blockSize.x, blockSize.y) >>>
        (finalColorBuffer,
            denoisedBuffer,
            fbSize);
}

