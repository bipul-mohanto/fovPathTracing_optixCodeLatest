#include "Maths.h"

// BSDF Declarations
enum BxDFType {
    BSDF_REFLECTION = 1 << 0,
    BSDF_TRANSMISSION = 1 << 1,
    BSDF_DIFFUSE = 1 << 2,
    BSDF_GLOSSY = 1 << 3,
    BSDF_SPECULAR = 1 << 4,
    BSDF_ALL = BSDF_DIFFUSE | BSDF_GLOSSY | BSDF_SPECULAR | BSDF_REFLECTION |
    BSDF_TRANSMISSION,
};

struct SurfaceInteraction{
    float3 p;
    Float time;
    Vector3f pError;
    Vector3f wo;
    float3 n;
    //MediumInterface mediumInterface;

    // SurfaceInteraction Public Data
    Point2f uv;
    Vector3f dpdu, dpdv;
    float3 dndu, dndv;
    //const Shape* shape = nullptr;
    struct {
        float3 n;
        Vector3f dpdu, dpdv;
        float3 dndu, dndv;
    } shading;
    //const Primitive* primitive = nullptr;
    BSDF* bsdf = nullptr;
    //BSSRDF* bssrdf = nullptr;
    mutable Vector3f dpdx, dpdy;
    mutable Float dudx = 0, dvdx = 0, dudy = 0, dvdy = 0;
};

CUDA_CALLABLE CUDA_INLINE bool MatchesFlags(BxDFType type, BxDFType flags) { return (type & flags) == type; }

// BxDF Declarations
struct BxDF {
public:
    // BxDF Interface
    virtual ~BxDF() {}
    BxDF() {}    
    virtual Spectrum Eval(const float3& wo, const float3& wi) const = 0;
    
    virtual void Sample(const Vector3f& wo, Vector3f& wi, const Point2f& sample) const {
        // Cosine-sample the hemisphere, flipping the direction if necessary
        wi = CosineSampleHemisphere(sample);
        if (wo.z < 0) wi.z *= -1;
    };
    virtual float Pdf(const Vector3f& wo, const Vector3f& wi) const {
        return SameHemisphere(wo, wi) ? AbsCosTheta(wi) * InvPi : 0;
    };

    virtual Spectrum Sample_f(const Vector3f& wo, Vector3f& wi, const Point2f& sample, float& pdf) const {
        Sample(wo, wi, sample);
        pdf = Pdf(wo, wi);
        return Eval(wo, wi);
    };
};



class BSDF {
public:
    // BSDF Public Methods
    BSDF(const SurfaceInteraction& si, Float eta = 1)
        : eta(eta),
        ns(si.shading.n),
        ng(si.n),
        ss(normalize(si.shading.dpdu)),
        ts(cross(ns, ss)) {}
    void Add(BxDF* b) {
        bxdfs[nBxDFs++] = b;
    }
    int NumComponents(BxDFType flags = BSDF_ALL) const;
    Vector3f WorldToLocal(const Vector3f& v) const {
        return make_float3(dot(v, ss), dot(v, ts), dot(v, ns));
    }
    Vector3f LocalToWorld(const Vector3f& v) const {
        return make_float3(ss.x * v.x + ts.x * v.y + ns.x * v.z,
            ss.y * v.x + ts.y * v.y + ns.y * v.z,
            ss.z * v.x + ts.z * v.y + ns.z * v.z);
    }
    Spectrum f(const Vector3f& woW, const Vector3f& wiW,
        BxDFType flags = BSDF_ALL) const;    
    Spectrum Sample_f(const Vector3f& wo, Vector3f* wi, const Point2f& u,
        Float* pdf, BxDFType type = BSDF_ALL,
        BxDFType* sampledType = nullptr) const;
    Float Pdf(const Vector3f& wo, const Vector3f& wi,
        BxDFType flags = BSDF_ALL) const;

    // BSDF Public Data
    const Float eta;

private:
    // BSDF Private Methods
    ~BSDF() {}

    // BSDF Private Data
    const float3 ns, ng;
    const Vector3f ss, ts;
    int nBxDFs = 0;
    static const int MaxBxDFs = 8;
    BxDF* bxdfs[MaxBxDFs];
    friend class MixMaterial;
};

inline int BSDF::NumComponents(BxDFType flags) const {
    int num = 0;
    for (int i = 0; i < nBxDFs; ++i)
        if (bxdfs[i]->MatchesFlags(flags)) ++num;
    return num;
}

Spectrum BSDF::f(const Vector3f& woW, const Vector3f& wiW,
    BxDFType flags) const {
    //ProfilePhase pp(Prof::BSDFEvaluation);
    Vector3f wi = WorldToLocal(wiW), wo = WorldToLocal(woW);
    if (wo.z == 0) return make_float3(0.f);
    bool reflect = dot(wiW, ng) * dot(woW, ng) > 0;
    Spectrum f = make_float3(0.f);
    for (int i = 0; i < nBxDFs; ++i)
        if (bxdfs[i]->MatchesFlags(flags) &&
            ((reflect && (bxdfs[i]->type & BSDF_REFLECTION)) ||
                (!reflect && (bxdfs[i]->type & BSDF_TRANSMISSION))))
            f += bxdfs[i]->f(wo, wi);
    return f;
}

Spectrum BSDF::Sample_f(const Vector3f& woWorld, Vector3f* wiWorld,
    const Point2f& u, Float* pdf, BxDFType type,
    BxDFType* sampledType) const {
    //ProfilePhase pp(Prof::BSDFSampling);
    // Choose which _BxDF_ to sample
    int matchingComps = NumComponents(type);
    if (matchingComps == 0) {
        *pdf = 0;
        if (sampledType) *sampledType = BxDFType(0);
        return make_float3(0);
    }
    int comp =
        min((int)std::floor(u.x * matchingComps), matchingComps - 1);

    // Get _BxDF_ pointer for chosen component
    BxDF* bxdf = nullptr;
    int count = comp;
    for (int i = 0; i < nBxDFs; ++i)
        if (bxdfs[i]->MatchesFlags(type) && count-- == 0) {
            bxdf = bxdfs[i];
            break;
        }    

    // Remap _BxDF_ sample _u_ to $[0,1)^2$
    Point2f uRemapped = make_float2(min(u.x * matchingComps - comp, OneMinusEpsilon), u.y);

    // Sample chosen _BxDF_
    Vector3f wi, wo = WorldToLocal(woWorld);
    if (wo.z == 0) return make_float3(0.f);
    *pdf = 0;
    if (sampledType) *sampledType = bxdf->type;
    Spectrum f = bxdf->Sample_f(wo, &wi, uRemapped, pdf, sampledType);
    
    if (*pdf == 0) {
        if (sampledType) *sampledType = BxDFType(0);
        return make_float3(0.f);
    }
    *wiWorld = LocalToWorld(wi);

    // Compute overall PDF with all matching _BxDF_s
    if (!(bxdf->type & BSDF_SPECULAR) && matchingComps > 1)
        for (int i = 0; i < nBxDFs; ++i)
            if (bxdfs[i] != bxdf && bxdfs[i]->MatchesFlags(type))
                *pdf += bxdfs[i]->Pdf(wo, wi);
    if (matchingComps > 1) *pdf /= matchingComps;

    // Compute value of BSDF for sampled direction
    if (!(bxdf->type & BSDF_SPECULAR)) {
        bool reflect = dot(*wiWorld, ng) * dot(woWorld, ng) > 0;
        f = make_float3(0.);
        for (int i = 0; i < nBxDFs; ++i)
            if (bxdfs[i]->MatchesFlags(type) &&
                ((reflect && (bxdfs[i]->type & BSDF_REFLECTION)) ||
                    (!reflect && (bxdfs[i]->type & BSDF_TRANSMISSION))))
                f += bxdfs[i]->f(wo, wi);
    }    
    return f;
}

Float BSDF::Pdf(const Vector3f& woWorld, const Vector3f& wiWorld,
    BxDFType flags) const {
    //ProfilePhase pp(Prof::BSDFPdf);
    if (nBxDFs == 0.f) return 0.f;
    Vector3f wo = WorldToLocal(woWorld), wi = WorldToLocal(wiWorld);
    if (wo.z == 0) return 0.;
    Float pdf = 0.f;
    int matchingComps = 0;
    for (int i = 0; i < nBxDFs; ++i)
        if (bxdfs[i]->MatchesFlags(flags)) {
            ++matchingComps;
            pdf += bxdfs[i]->Pdf(wo, wi);
        }
    Float v = matchingComps > 0 ? pdf / matchingComps : 0.f;
    return v;
}