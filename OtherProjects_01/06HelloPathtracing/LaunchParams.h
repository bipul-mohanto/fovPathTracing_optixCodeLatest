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

#pragma once
 
#include "Util.h"
#include "Maths.h"
#include <optix.h>

// for this simple example, we have a single ray type
enum {
	RAY_TYPE_RADIANCE = 0,
	RAY_TYPE_OCCLUSION = 1,
	RAY_TYPE_COUNT
};

struct Material
{
	Material()
	{
		color = make_float3(0.8f, 0.8f, 0.8f);
		emission = make_float3(0.0f);
		absorption = make_float3(0.0);

		// when eta is zero the index of refraction will be inferred from the specular component
		eta = 0.0f;

		metallic = 0.0f;
		subsurface = 0.0f;
		specular = 0.5f;
		roughness = 0.5f;
		specularTint = 0.0f;
		anisotropic = 0.0f;
		sheen = 0.0f;
		sheenTint = 0.0f;
		clearcoat = 0.0f;
		clearcoatGloss = 1.0f;
		transmission = 0.0f;
		bump = 0.0f;
		bumpTile = make_float3(10.0f);

	}

	CUDA_CALLABLE __forceinline__ float GetIndexOfRefraction() const
	{
		if (eta == 0.0f)
			return 2.0f / (1.0f - sqrt(0.08f * specular)) - 1.0f;
		else
			return eta;
	}

	float3 emission;
	float3 color;
	float3 absorption;

	float eta;
	float metallic;
	float subsurface;
	float specular;
	float roughness;
	float specularTint;
	float anisotropic;
	float sheen;
	float sheenTint;
	float clearcoat;
	float clearcoatGloss;
	float transmission;

	//Texture bumpMap;
	float bump;
	float3 bumpTile;
};


struct TriangleMeshSBTData {
    float3* vertex;
    float3* normal;
    float2* texcoord;
    uint3* index;
    bool                hasTexture;
    cudaTextureObject_t texture;

	Material material;
};

struct Params
{
    struct {
        float4* accum_buffer;
		uchar4* frame_buffer;

		float4* color_buffer;
		float4* normal_buffer;
		float4* albedo_buffer;
        
        int2     size;
        unsigned int subframe_index;
    } frame;
    
    struct {
        float3       eye;
        float3       U;
        float3       V;
        float3       W;
    } camera;   

	unsigned int maxDepth;
    unsigned int samples_per_launch;

    OptixTraversableHandle traversable;
};

struct RayGenData
{
    void* data;
};

struct MissData
{
    void* data;
};

struct HitGroupData
{
    TriangleMeshSBTData data;
};



