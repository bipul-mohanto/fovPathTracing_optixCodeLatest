#include "Maths.h"
/*
// BSSRDF Declarations
class BSSRDF {
public:
    // BSSRDF Public Methods
    BSSRDF(const SurfaceInteraction& po, Float eta) : po(po), eta(eta) {}
    virtual ~BSSRDF() {}

    // BSSRDF Interface
    virtual Spectrum S(const SurfaceInteraction& pi, const Vector3f& wi) = 0;
    virtual Spectrum Sample_S(const Scene& scene, Float u1, const Point2f& u2,
        MemoryArena& arena, SurfaceInteraction* si,
        Float* pdf) const = 0;

protected:
    // BSSRDF Protected Data
    const SurfaceInteraction& po;
    Float eta;
};

class SeparableBSSRDF : public BSSRDF {
    friend class SeparableBSSRDFAdapter;

public:
    // SeparableBSSRDF Public Methods
    SeparableBSSRDF(const SurfaceInteraction& po, Float eta,
        const Material* material, TransportMode mode)
        : BSSRDF(po, eta),
        ns(po.shading.n),
        ss(Normalize(po.shading.dpdu)),
        ts(Cross(ns, ss)),
        material(material),
        mode(mode) {}
    Spectrum S(const SurfaceInteraction& pi, const Vector3f& wi) {
        ProfilePhase pp(Prof::BSSRDFEvaluation);
        Float Ft = FrDielectric(CosTheta(po.wo), 1, eta);
        return (1 - Ft) * Sp(pi) * Sw(wi);
    }
    Spectrum Sw(const Vector3f& w) const {
        Float c = 1 - 2 * FresnelMoment1(1 / eta);
        return (1 - FrDielectric(CosTheta(w), 1, eta)) / (c * Pi);
    }
    Spectrum Sp(const SurfaceInteraction& pi) const {
        return Sr(Distance(po.p, pi.p));
    }
    Spectrum Sample_S(const Scene& scene, Float u1, const Point2f& u2,
        MemoryArena& arena, SurfaceInteraction* si,
        Float* pdf) const;
    Spectrum Sample_Sp(const Scene& scene, Float u1, const Point2f& u2,
        MemoryArena& arena, SurfaceInteraction* si,
        Float* pdf) const;
    Float Pdf_Sp(const SurfaceInteraction& si) const;

    // SeparableBSSRDF Interface
    virtual Spectrum Sr(Float d) const = 0;
    virtual Float Sample_Sr(int ch, Float u) const = 0;
    virtual Float Pdf_Sr(int ch, Float r) const = 0;

private:
    // SeparableBSSRDF Private Data
    const Normal3f ns;
    const Vector3f ss, ts;
    const Material* material;
    const TransportMode mode;
};*/