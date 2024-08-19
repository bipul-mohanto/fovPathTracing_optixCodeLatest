#include "LaunchParams.h"
#include "Disney.cuh"

#include <optix_device.h>

struct Sampler {

};

Spectrum PathIntegrator(Sampler& sampler, int depth) {    
    Spectrum L = make_float3(0.f), beta = make_float3(1.f);
    //RayDifferential ray(r);
    bool specularBounce = false;
    int bounces;
    // Added after book publication: etaScale tracks the accumulated effect
    // of radiance scaling due to rays passing through refractive
    // boundaries (see the derivation on p. 527 of the third edition). We
    // track this value in order to remove it from beta when we apply
    // Russian roulette; this is worthwhile, since it lets us sometimes
    // avoid terminating refracted rays that are about to be refracted back
    // out of a medium and thus have their beta value increased.
    Float etaScale = 1;

    for (bounces = 0;; ++bounces) {
        // Find next path vertex and accumulate contribution
        

        // Intersect _ray_ with scene and store intersection in _isect_
        SurfaceInteraction isect;
        bool foundIntersection = scene.Intersect(ray, &isect);

        // Possibly add emitted light at intersection
        if (bounces == 0 || specularBounce) {
            // Add emitted light at path vertex or from the environment
            if (foundIntersection) {
                L += beta * isect.Le(-ray.d);
                VLOG(2) << "Added Le -> L = " << L;
            }
            else {
                for (const auto& light : scene.infiniteLights)
                    L += beta * light->Le(ray);
                //VLOG(2) << "Added infinite area lights -> L = " << L;
            }
        }

        // Terminate path if ray escaped or _maxDepth_ was reached
        if (!foundIntersection || bounces >= maxDepth) break;

        // Compute scattering functions and skip over medium boundaries
        isect.ComputeScatteringFunctions(ray, arena, true);
        if (!isect.bsdf) {
            //VLOG(2) << "Skipping intersection due to null bsdf";
            ray = isect.SpawnRay(ray.d);
            bounces--;
            continue;
        }

        const Distribution1D* distrib = lightDistribution->Lookup(isect.p);

        // Sample illumination from lights to find path contribution.
        // (But skip this for perfectly specular BSDFs.)
        if (isect.bsdf->NumComponents(BxDFType(BSDF_ALL & ~BSDF_SPECULAR)) >
            0) {
            ++totalPaths;
            Spectrum Ld = beta * UniformSampleOneLight(isect, scene, arena,
                sampler, false, distrib);
            //VLOG(2) << "Sampled direct lighting Ld = " << Ld;
            if (Ld.IsBlack()) ++zeroRadiancePaths;
            //CHECK_GE(Ld.y(), 0.f);
            L += Ld;
        }

        // Sample BSDF to get new path direction
        Vector3f wo = -ray.d, wi;
        Float pdf;
        BxDFType flags;
        Spectrum f = isect.bsdf->Sample_f(wo, &wi, sampler.Get2D(), &pdf,
            BSDF_ALL, &flags);
        //VLOG(2) << "Sampled BSDF, f = " << f << ", pdf = " << pdf;
        if (f.IsBlack() || pdf == 0.f) break;
        beta *= f * AbsDot(wi, isect.shading.n) / pdf;
        //VLOG(2) << "Updated beta = " << beta;
        //CHECK_GE(beta.y(), 0.f);
        //DCHECK(!std::isinf(beta.y()));
        specularBounce = (flags & BSDF_SPECULAR) != 0;
        if ((flags & BSDF_SPECULAR) && (flags & BSDF_TRANSMISSION)) {
            Float eta = isect.bsdf->eta;
            // Update the term that tracks radiance scaling for refraction
            // depending on whether the ray is entering or leaving the
            // medium.
            etaScale *= (Dot(wo, isect.n) > 0) ? (eta * eta) : 1 / (eta * eta);
        }
        ray = isect.SpawnRay(wi);

        // Account for subsurface scattering, if applicable
        if (isect.bssrdf && (flags & BSDF_TRANSMISSION)) {
            // Importance sample the BSSRDF
            SurfaceInteraction pi;
            Spectrum S = isect.bssrdf->Sample_S(
                scene, sampler.Get1D(), sampler.Get2D(), arena, &pi, &pdf);
            DCHECK(!std::isinf(beta.y()));
            if (S.IsBlack() || pdf == 0) break;
            beta *= S / pdf;

            // Account for the direct subsurface scattering component
            L += beta * UniformSampleOneLight(pi, scene, arena, sampler, false,
                lightDistribution->Lookup(pi.p));

            // Account for the indirect subsurface scattering component
            Spectrum f = pi.bsdf->Sample_f(pi.wo, &wi, sampler.Get2D(), &pdf,
                BSDF_ALL, &flags);
            if (f.IsBlack() || pdf == 0) break;
            beta *= f * AbsDot(wi, pi.shading.n) / pdf;
            DCHECK(!std::isinf(beta.y()));
            specularBounce = (flags & BSDF_SPECULAR) != 0;
            ray = pi.SpawnRay(wi);
        }

        // Possibly terminate the path with Russian roulette.
        // Factor out radiance scaling due to refraction in rrBeta.
        Spectrum rrBeta = beta * etaScale;
        if (rrBeta.MaxComponentValue() < rrThreshold && bounces > 3) {
            Float q = std::max((Float).05, 1 - rrBeta.MaxComponentValue());
            if (sampler.Get1D() < q) break;
            beta /= 1 - q;
            DCHECK(!std::isinf(beta.y()));
        }
    }
    ReportValue(pathLength, bounces);
    return L;
}

struct RadiancePRD
{
    int bounces;    
    float3 L;
    float3 beta;
    float etaScale;
    bool specularBounce;

    bool done;
};

extern "C" {
    __constant__ Params params;
}

static __forceinline__ __device__ void* unpackPointer(unsigned int i0, unsigned int i1)
{
    const unsigned long long uptr = static_cast<unsigned long long>(i0) << 32 | i1;
    void* ptr = reinterpret_cast<void*>(uptr);
    return ptr;
}


static __forceinline__ __device__ void  packPointer(void* ptr, unsigned int& i0, unsigned int& i1)
{
    const unsigned long long uptr = reinterpret_cast<unsigned long long>(ptr);
    i0 = uptr >> 32;
    i1 = uptr & 0x00000000ffffffff;
}

template<typename T>
static __forceinline__ __device__ T* getPRD()
{
    const unsigned int u0 = optixGetPayload_0();
    const unsigned int u1 = optixGetPayload_1();
    return reinterpret_cast<T*>(unpackPointer(u0, u1));
}


static __forceinline__ __device__ void setPayloadOcclusion(bool occluded)
{
    optixSetPayload_0(static_cast<unsigned int>(occluded));
}

extern "C" __global__ void __anyhit__occlusion()
{
    //setPayloadOcclusion(true);
}

extern "C" __global__ void __closesthit__occlusion()
{

}

extern "C" __global__ void __miss__occlusion()
{
    //setPayloadOcclusion(false);
}

extern "C" __global__ void __raygen__renderFrame()
{
    RadiancePRD prd;
    prd.L = make_float3(0.f);
    prd.beta = make_float3(0.f);
    prd.etaScale = 1.f;
    prd.specularBounce = false;
    prd.done = false;
}

extern "C" __global__ void __miss__radiance()
{
    RadiancePRD* prd = getPRD<RadiancePRD>();

    if (prd->bounces == 0 || prd->specularBounce) {
        for (const auto& light : scene.infiniteLights)
            prd->L += prd->beta * light->Le(ray);
    }

    prd->done = true;
}

extern "C" __global__ void __closesthit__radiance()
{
    RadiancePRD* prd = getPRD<RadiancePRD>();

    if (prd->bounces == 0 || prd->specularBounce) {
        // Add emitted light at path vertex or from the environment
        prd->L += prd->beta * isect.Le(-ray.d);
    }

    // Compute scattering functions and skip over medium boundaries
    isect.ComputeScatteringFunctions(ray, arena, true);
    if (!isect.bsdf) {
        ray = isect.SpawnRay(ray.d);
        prd->bounces--;
        return;
    }

    const Distribution1D* distrib = lightDistribution->Lookup(isect.p);

    // Sample illumination from lights to find path contribution.
    // (But skip this for perfectly specular BSDFs.)
    if (isect.bsdf->NumComponents(BxDFType(BSDF_ALL & ~BSDF_SPECULAR)) >
        0) {
        ++totalPaths;
        Spectrum Ld = beta * UniformSampleOneLight(isect, scene, arena,
            sampler, false, distrib);
        //VLOG(2) << "Sampled direct lighting Ld = " << Ld;
        if (IsBlack(Ld)) ++zeroRadiancePaths;
        //CHECK_GE(Ld.y(), 0.f);
        prd->L += Ld;
    }

    // Sample BSDF to get new path direction
    Vector3f wo = -ray.d, wi;
    Float pdf;
    BxDFType flags;
    Spectrum f = isect.bsdf->Sample_f(wo, &wi, sampler.Get2D(), &pdf, BSDF_ALL, &flags);

    if (IsBlack(f) || pdf == 0.f) {
        prd->done = true;
        return;
    }
    prd->beta *= f * AbsDot(wi, isect.shading.n) / pdf;
    prd->specularBounce = (flags & BSDF_SPECULAR) != 0;
    if ((flags & BSDF_SPECULAR) && (flags & BSDF_TRANSMISSION)) {
        Float eta = isect.bsdf->eta;
        // Update the term that tracks radiance scaling for refraction
        // depending on whether the ray is entering or leaving the
        // medium.
        prd->etaScale *= (dot(wo, isect.n) > 0) ? (eta * eta) : 1 / (eta * eta);
    }
    ray = isect.SpawnRay(wi);

    // Account for subsurface scattering, if applicable

    /*if (isect.bssrdf && (flags & BSDF_TRANSMISSION)) {
        // Importance sample the BSSRDF
        SurfaceInteraction pi;
        //Spectrum S = isect.bssrdf->Sample_S(scene, sampler.Get1D(), sampler.Get2D(), arena, &pi, &pdf);
        if (S.IsBlack() || pdf == 0) break;
            beta *= S / pdf;

        // Account for the direct subsurface scattering component
        prd->L += prd->beta * UniformSampleOneLight(pi, scene, arena, sampler, false,
            lightDistribution->Lookup(pi.p));

        // Account for the indirect subsurface scattering component
        Spectrum f = pi.bsdf->Sample_f(pi.wo, &wi, sampler.Get2D(), &pdf,
            BSDF_ALL, &flags);
        if (f.IsBlack() || pdf == 0) break;
        beta *= f * AbsDot(wi, pi.shading.n) / pdf;
        DCHECK(!std::isinf(beta.y()));
        specularBounce = (flags & BSDF_SPECULAR) != 0;
        ray = pi.SpawnRay(wi);
    }*/

    // Possibly terminate the path with Russian roulette.
    // Factor out radiance scaling due to refraction in rrBeta.
    Spectrum rrBeta = prd->beta * prd->etaScale;
    if (rrBeta.MaxComponentValue() < rrThreshold && bounces > 3) {
        Float q = std::max((Float).05, 1 - rrBeta.MaxComponentValue());
        if (sampler.Get1D() < q) break;
        beta /= 1 - q;
        DCHECK(!std::isinf(beta.y()));
    }

}