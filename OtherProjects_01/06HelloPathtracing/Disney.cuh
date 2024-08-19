
#include <random.h>

#include "Maths.h"
#include "BSDF.cuh"

namespace pbrt {   
    

    enum class TransportMode { Radiance, Importance };    


    // https://seblagarde.wordpress.com/2013/04/29/memo-on-fresnel-equations/
    //
    // The Schlick Fresnel approximation is:
    //
    // R = R(0) + (1 - R(0)) (1 - cos theta)^5,
    //
    // where R(0) is the reflectance at normal indicence.
    __device__ __forceinline__ float SchlickWeight(float cosTheta) {
        float m = Clamp(1 - cosTheta, 0.f, 1.f);
        return (m * m) * (m * m) * m;
    }

    __device__ __forceinline__ float FrSchlick(float R0, float cosTheta) {
        return Lerp(SchlickWeight(cosTheta), R0, 1);
    }

    __device__ __forceinline__ Spectrum FrSchlick(const Spectrum& R0, float cosTheta) {
        return Lerp(SchlickWeight(cosTheta), R0, make_float3(0.f));
    }

    // For a dielectric, R(0) = (eta - 1)^2 / (eta + 1)^2, assuming we're
    // coming from air..
    __device__ __forceinline__ float SchlickR0FromEta(float eta) { return sqr(eta - 1.f) / sqr(eta + 1.f); }

    ///////////////////////////////////////////////////////////////////////////
    // DisneyDiffuse

    struct DisneyDiffuse {
        static const BxDFType type = BxDFType(BSDF_REFLECTION | BSDF_DIFFUSE);

        static Spectrum Eval(const float3& wo, const float3& wi, const float3 R) {
            Float Fo = SchlickWeight(AbsCosTheta(wo)),
                Fi = SchlickWeight(AbsCosTheta(wi));

            return R * InvPi * (1 - Fo / 2) * (1 - Fi / 2);
        }

        static float Pdf(const float3& wi) {
            return AbsCosTheta(wi) * InvPi;
        }

        static void Sample(const Vector3f& wo, Vector3f& wi, const Point2f& sample) {
            // Cosine-sample the hemisphere, flipping the direction if necessary
            wi = CosineSampleHemisphere(sample);
            if (wo.z < 0) wi.z *= -1;            
        };
    };

    struct LambertianTransmission  {
        static const BxDFType type = BxDFType(BSDF_TRANSMISSION | BSDF_DIFFUSE);

        static Spectrum Eval(const float3& T) {
            return T * InvPi;
        }

        static float Pdf(const float3& wi) {
            return AbsCosTheta(wi) * InvPi;
        }

        static void Sample(const Vector3f& wo, Vector3f& wi, const Point2f& sample) {
            // Cosine-sample the hemisphere, flipping the direction if necessary
            wi = CosineSampleHemisphere(sample);
            if (wo.z < 0) wi.z *= -1;
        };
    };    

    ///////////////////////////////////////////////////////////////////////////
    // DisneyFakeSS

    // "Fake" subsurface scattering lobe, based on the Hanrahan-Krueger BRDF
    // approximation of the BSSRDF.

    struct DisneyFakeSS { 

        static const BxDFType type = BxDFType(BSDF_REFLECTION | BSDF_DIFFUSE);

        static Spectrum Eval(const float3& wo, const float3& wi, const float3 R, const float roughness) {
            Float Fo = SchlickWeight(AbsCosTheta(wo)),
                Fi = SchlickWeight(AbsCosTheta(wi));

            float3 wh = wi + wo;
            if (wh.x == 0 && wh.y == 0 && wh.z == 0) return make_float3(0.f);
            float cosThetaD = dot(wi, wh);

            // Fss90 used to "flatten" retroreflection based on roughness
            float Fss90 = cosThetaD * cosThetaD * roughness;
            float Fss = Lerp(Fo, 1.0, Fss90) * Lerp(Fi, 1.0, Fss90);
            // 1.25 scale is used to (roughly) preserve albedo
            float ss =
                1.25f * (Fss * (1 / (AbsCosTheta(wo) + AbsCosTheta(wi)) - .5f) + .5f);

            return R * InvPi * ss;
        }

        static float Pdf(const float3& wi) {
            return AbsCosTheta(wi) * InvPi;
        }

        static void Sample(const Vector3f& wo, Vector3f& wi, const Point2f& sample) {
            // Cosine-sample the hemisphere, flipping the direction if necessary
            wi = CosineSampleHemisphere(sample);
            if (wo.z < 0) wi.z *= -1;
        };
    };    

    ///////////////////////////////////////////////////////////////////////////
    // DisneyRetro

    struct DisneyRetro {

        static const BxDFType type = BxDFType(BSDF_REFLECTION | BSDF_DIFFUSE); 

        static Spectrum Eval(const float3& wo, const float3& wi, const float3& R, const float& roughness) {
            Float Fo = SchlickWeight(AbsCosTheta(wo)),
                Fi = SchlickWeight(AbsCosTheta(wi));

            float3 wh = wi + wo;
            if (wh.x == 0 && wh.y == 0 && wh.z == 0) return make_float3(0.f);
            wh = normalize(wh);
            float cosThetaD = dot(wi, wh);

            float Rr = 2 * roughness * cosThetaD * cosThetaD;

            // Burley 2015, eq (4).
            return R * InvPi * Rr * (Fo + Fi + Fo * Fi * (Rr - 1));
        }

        static float Pdf(const float3& wi) {
            return AbsCosTheta(wi) * InvPi;
        }

        static void Sample(const Vector3f& wo, Vector3f& wi, const Point2f& sample) {
            // Cosine-sample the hemisphere, flipping the direction if necessary
            wi = CosineSampleHemisphere(sample);
            if (wo.z < 0) wi.z *= -1;
        };
    };



    ///////////////////////////////////////////////////////////////////////////
    // DisneySheen

    struct DisneySheen {       

        static const BxDFType type = BxDFType(BSDF_REFLECTION | BSDF_DIFFUSE);                

        static Spectrum Eval(const float3& wo, const float3& wi, const float3& R) {
            float3 wh = wi + wo;
            if (wh.x == 0 && wh.y == 0 && wh.z == 0) return make_float3(0.f);
            wh = normalize(wh);
            float cosThetaD = dot(wi, wh);

            return R * SchlickWeight(cosThetaD);
        }

        static float Pdf(const float3& wi) {
            return AbsCosTheta(wi) * InvPi;
        }

        static void Sample(const Vector3f& wo, Vector3f& wi, const Point2f& sample) {
            // Cosine-sample the hemisphere, flipping the direction if necessary
            wi = CosineSampleHemisphere(sample);
            if (wo.z < 0) wi.z *= -1;
        };
    };


    ///////////////////////////////////////////////////////////////////////////
    // DisneyClearcoat

    inline float GTR1(float cosTheta, float alpha) {
        float alpha2 = alpha * alpha;
        return (alpha2 - 1) /
            (Pi * std::log(alpha2) * (1 + (alpha2 - 1) * cosTheta * cosTheta));
    }

    // Smith masking/shadowing term.
    inline float smithG_GGX(float cosTheta, float alpha) {
        float alpha2 = alpha * alpha;
        float cosTheta2 = cosTheta * cosTheta;
        return 1 / (cosTheta + sqrt(alpha2 + cosTheta2 - alpha2 * cosTheta2));
    }

    struct DisneyClearcoat {       

        static const BxDFType type = BxDFType(BSDF_REFLECTION | BSDF_GLOSSY);

        static Spectrum Eval(const float3& wo, const float3& wi, const float& weight, const float& gloss)
        {
            float3 wh = wi + wo;
            if (wh.x == 0 && wh.y == 0 && wh.z == 0) return make_float3(0.f);
            wh = normalize(wh);

            // Clearcoat has ior = 1.5 hardcoded -> F0 = 0.04. It then uses the
            // GTR1 distribution, which has even fatter tails than Trowbridge-Reitz
            // (which is GTR2).
            float Dr = GTR1(AbsCosTheta(wh), gloss);
            float Fr = FrSchlick(.04, dot(wo, wh));
            // The geometric term always based on alpha = 0.25.
            float Gr =
                smithG_GGX(AbsCosTheta(wo), .25) * smithG_GGX(AbsCosTheta(wi), .25);

            return make_float3(weight * Gr * Fr * Dr / 4);
        }

        static Spectrum Sample(const float3& wo, float3* wi, const float2& u, const float& gloss) 
        {
            // TODO: double check all this: there still seem to be some very
            // occasional fireflies with clearcoat; presumably there is a bug
            // somewhere.
            if (wo.z == 0) return make_float3(0.f);

            float alpha2 = gloss * gloss;
            float cosTheta = sqrt(max(0.f, (1.f - pow(alpha2, 1 - u.x)) / (1 - alpha2)));
            float sinTheta = sqrt(max(0.f, 1.f - cosTheta * cosTheta));
            float phi = 2 * Pi * u.y;
            float3 wh = SphericalDirection(sinTheta, cosTheta, phi);
            if (!SameHemisphere(wo, wh)) wh = -wh;

            *wi = reflect(wo, wh);
            if (!SameHemisphere(wo, *wi)) return make_float3(0.f);
        }

        static float Pdf(const float3& wo, const float3& wi, const float& gloss)
        {
            float3 wh = wi + wo;
            if (wh.x == 0 && wh.y == 0 && wh.z == 0) return 0;
            wh = normalize(wh);

            // The sampling routine samples wh exactly from the GTR1 distribution.
            // Thus, the final value of the PDF is just the value of the
            // distribution for wh converted to a mesure with respect to the
            // surface normal.
            float Dr = GTR1(AbsCosTheta(wh), gloss);
            return Dr * AbsCosTheta(wh) / (4 * dot(wo, wh));
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    // DisneyFresnel

    // Specialized Fresnel function used for the specular component, based on
    // a mixture between dielectric and the Schlick Fresnel approximation.

    struct FresnelDielectric {

        static Spectrum Evaluate(Float cosThetaI, Float etaI, Float etaT)
        {
            return make_float3(FrDielectric(cosThetaI, etaI, etaT));
        }
    };

    struct DisneyFresnel {

        static Spectrum Evaluate(const Spectrum& R0, const float& metallic, const float& eta, const float& cosI) {
            return Lerp(metallic, make_float3(FrDielectric(cosI, 1, eta)),
                FrSchlick(R0, cosI));
        }
    };    

    ///////////////////////////////////////////////////////////////////////////
    // DisneyMicrofacetDistribution

    static void TrowbridgeReitzSample11(Float cosTheta, Float U1, Float U2,
        Float* slope_x, Float* slope_y) {
        // special case (normal incidence)
        if (cosTheta > .9999) {
            Float r = sqrt(U1 / (1 - U1));
            Float phi = 6.28318530718 * U2;
            *slope_x = r * cos(phi);
            *slope_y = r * sin(phi);
            return;
        }

        Float sinTheta =
            sqrt(max((Float)0, (Float)1 - cosTheta * cosTheta));
        Float tanTheta = sinTheta / cosTheta;
        Float a = 1 / tanTheta;
        Float G1 = 2 / (1 + std::sqrt(1.f + 1.f / (a * a)));

        // sample slope_x
        Float A = 2 * U1 / G1 - 1;
        Float tmp = 1.f / (A * A - 1.f);
        if (tmp > 1e10) tmp = 1e10;
        Float B = tanTheta;
        Float D = std::sqrt(
            max(Float(B * B * tmp * tmp - (A * A - B * B) * tmp), Float(0)));
        Float slope_x_1 = B * tmp - D;
        Float slope_x_2 = B * tmp + D;
        *slope_x = (A < 0 || slope_x_2 > 1.f / tanTheta) ? slope_x_1 : slope_x_2;

        // sample slope_y
        Float S;
        if (U2 > 0.5f) {
            S = 1.f;
            U2 = 2.f * (U2 - .5f);
        }
        else {
            S = -1.f;
            U2 = 2.f * (.5f - U2);
        }
        Float z =
            (U2 * (U2 * (U2 * 0.27385f - 0.73369f) + 0.46341f)) /
            (U2 * (U2 * (U2 * 0.093073f + 0.309420f) - 1.000000f) + 0.597999f);
        *slope_y = S * z * std::sqrt(1.f + *slope_x * *slope_x);
    }

    static Vector3f TrowbridgeReitzSample(const Vector3f& wi, Float alpha_x,
        Float alpha_y, Float U1, Float U2) {
        // 1. stretch wi
        Vector3f wiStretched =
            normalize(make_float3(alpha_x * wi.x, alpha_y * wi.y, wi.z));

        // 2. simulate P22_{wi}(x_slope, y_slope, 1, 1)
        Float slope_x, slope_y;
        TrowbridgeReitzSample11(CosTheta(wiStretched), U1, U2, &slope_x, &slope_y);

        // 3. rotate
        Float tmp = CosPhi(wiStretched) * slope_x - SinPhi(wiStretched) * slope_y;
        slope_y = SinPhi(wiStretched) * slope_x + CosPhi(wiStretched) * slope_y;
        slope_x = tmp;

        // 4. unstretch
        slope_x = alpha_x * slope_x;
        slope_y = alpha_y * slope_y;

        // 5. compute normal
        return normalize(make_float3(-slope_x, -slope_y, 1.));
    }

    struct MicrofacetDistribution {

        static Float G1(const float& lambda_w) {
            //    if (Dot(w, wh) * CosTheta(w) < 0.) return 0.;
            return 1 / (1 + lambda_w);
        }

        static Float G(const float& lambda_wo, const float &lambda_wi) {
            return 1 / (1 + lambda_wo + lambda_wi);
        }        
    };

    struct TrowbridgeReitzDistribution {

        // MicrofacetDistribution Inline Methods
        static Float RoughnessToAlpha(Float roughness) {
            roughness = max(roughness, (Float)1e-3);
            Float x = log(roughness);
            return 1.62142f + 0.819955f * x + 0.1734f * x * x + 0.0171201f * x * x * x +
                0.000640711f * x * x * x * x;
        }        

        static Vector3f Sample_wh(const Vector3f& wo, const Point2f& u, const float& alphax, const float& alphay) {
            float3 wh;            
            bool flip = wo.z < 0;
            wh = TrowbridgeReitzSample(flip ? -wo : wo, alphax, alphay, u.x, u.y);
            if (flip) wh = -wh;
            
            return wh;
        };

        // TrowbridgeReitzDistribution Private Methods
        static Float Lambda(const Vector3f& w, const float& alphax, const float& alphay) {
            Float absTanTheta = abs(TanTheta(w));
            if (isinf(absTanTheta)) return 0.;
            // Compute _alpha_ for direction _w_
            Float alpha =
                sqrt(Cos2Phi(w) * alphax * alphax + Sin2Phi(w) * alphay * alphay);
            Float alpha2Tan2Theta = (alpha * absTanTheta) * (alpha * absTanTheta);
            return (-1 + sqrt(1.f + alpha2Tan2Theta)) / 2;
        };

        static Float D(const float3& wh, const float& alphax, const float& alphay) {
            float tan2Theta = Tan2Theta(wh);
            if (std::isinf(tan2Theta)) return 0.;
            const float cos4Theta = Cos2Theta(wh) * Cos2Theta(wh);
            float e =
                (Cos2Phi(wh) / (alphax * alphax) + Sin2Phi(wh) / (alphay * alphay)) *
                tan2Theta;
            return 1 / (Pi * alphax * alphay * cos4Theta * (1 + e) * (1 + e));
        }

        static float Pdf(const float3& wo, const float3& wh, const float& alphax, const float& alphay)
        {
            float lambda_wo = TrowbridgeReitzDistribution::Lambda(wo, alphax, alphay);
            return TrowbridgeReitzDistribution::D(wh, alphax, alphay) * MicrofacetDistribution::G1(lambda_wo) * AbsDot(wo, wh) / AbsCosTheta(wo);
        }
    };

    struct DisneyMicrofacetDistribution {

        static float Lambda(const float3& w, const float& alphax, const float& alphay) {
            Float absTanTheta = abs(TanTheta(w));
            if (isinf(absTanTheta)) return 0.;
            // Compute _alpha_ for direction _w_
            Float alpha =
                sqrt(Cos2Phi(w) * alphax * alphax + Sin2Phi(w) * alphay * alphay);
            Float alpha2Tan2Theta = (alpha * absTanTheta) * (alpha * absTanTheta);
            return (-1 + sqrt(1.f + alpha2Tan2Theta)) / 2;
        }

        static float G(const float3& wo, const float3& wi, const float& alphax, const float& alphay) {
            return (1 / (1 + Lambda(wo, alphax, alphay))) * (1 / (1 + Lambda(wi, alphax, alphay)));
        }

        static float D(const float3& wh, const float& alphax, const float& alphay) {
            float tan2Theta = Tan2Theta(wh);
            if (std::isinf(tan2Theta)) return 0.;
            const float cos4Theta = Cos2Theta(wh) * Cos2Theta(wh);
            float e =
                (Cos2Phi(wh) / (alphax * alphax) + Sin2Phi(wh) / (alphay * alphay)) *
                tan2Theta;
            return 1 / (Pi * alphax * alphay * cos4Theta * (1 + e) * (1 + e));
        }

        static float Pdf(const float3& wo, const float3& wh, const float& alphax, const float& alphay)
        {
            float lambda_wo = DisneyMicrofacetDistribution::Lambda(wo, alphax, alphay);
            return DisneyMicrofacetDistribution::D(wh, alphax, alphay) * MicrofacetDistribution::G1(lambda_wo) * AbsDot(wo, wh) / AbsCosTheta(wo);
        }
    };


    //////////////////////////////////////////////////////////////////////////
    // Microfacet

    struct MicrofacetReflection {
        static const BxDFType type = BxDFType(BSDF_REFLECTION | BSDF_GLOSSY);

        static Spectrum Eval_DisneyMicroDist(const float3& wo, const float3& wi, const float& alphax, const float& alphay, const float3 Fr) {
            Float cosThetaO = AbsCosTheta(wo), cosThetaI = AbsCosTheta(wi);
            Vector3f wh = wi + wo;
            // Handle degenerate cases for microfacet reflection
            if (cosThetaI == 0 || cosThetaO == 0) return make_float3(0.);
            if (wh.x == 0 && wh.y == 0 && wh.z == 0) return make_float3(0.);
            wh = normalize(wh);
            // For the Fresnel call, make sure that wh is in the same hemisphere
            // as the surface normal, so that TIR is handled correctly.
            
            return DisneyMicrofacetDistribution::D(wh, alphax, alphay) * DisneyMicrofacetDistribution::G(wo, wi, alphax, alphay) * Fr / (4 * cosThetaI * cosThetaO);
        }        

        static Float Pdf_DisneyMicroDist(const Vector3f& wo, const Vector3f& wi, const float& alphax, const float& alphay) {
            Vector3f wh = normalize(wo + wi);
            return DisneyMicrofacetDistribution::Pdf(wo, wh, alphax, alphay) / (4 * dot(wo, wh));
        };

        static void Sample_DisneyMicroDist(const Vector3f& wo, Vector3f* wi, const Point2f& u, const float& alphax, const float& alphay) {
            // Sample microfacet orientation $\wh$ and reflected direction $\wi$
            if (wo.z == 0) return ;
            Vector3f wh = TrowbridgeReitzDistribution::Sample_wh(wo, u, alphax, alphay);
            if (dot(wo, wh) < 0) return;   // Should be rare
            *wi = reflect(wo, wh);            
        }    

        static Spectrum Sample_f_DisneyMicroDist(const Vector3f& wo, Vector3f* wi, const Point2f& u, const float& alphax, const float& alphay, const float3& Fr, float& pdf) {
            // Sample microfacet orientation $\wh$ and reflected direction $\wi$
            if (wo.z == 0) return make_float3(0.f);
            Vector3f wh = TrowbridgeReitzDistribution::Sample_wh(wo, u, alphax, alphay);
            if (dot(wo, wh) < 0) return make_float3(0.f);   // Should be rare
            *wi = reflect(wo, wh);
            if (!SameHemisphere(wo, *wi)) return make_float3(0.f);

            // Compute PDF of _wi_ for microfacet reflection
            pdf = DisneyMicrofacetDistribution::Pdf(wo, wh, alphax, alphay) / (4 * dot(wo, wh));
            
            return MicrofacetReflection::Eval_DisneyMicroDist(wo, *wi, alphax, alphay, Fr);
        }        
    };

    struct MicrofacetTransmission : public BxDF {
        static const BxDFType type = BxDFType(BSDF_TRANSMISSION | BSDF_GLOSSY);

        static Spectrum Eval_DisneyMicroDist(const Spectrum& T, const Vector3f& wo, const Vector3f& wi, const float& etaA, const float& etaB, const float& alphax, const float& alphay, const Spectrum& Fr, TransportMode mode) {
            if (SameHemisphere(wo, wi)) return make_float3(0);  // transmission only

            Float cosThetaO = CosTheta(wo);
            Float cosThetaI = CosTheta(wi);
            if (cosThetaI == 0 || cosThetaO == 0) return make_float3(0);

            // Compute $\wh$ from $\wo$ and $\wi$ for microfacet transmission
            Float eta = CosTheta(wo) > 0 ? (etaB / etaA) : (etaA / etaB);
            Vector3f wh = normalize(wo + wi * eta);
            if (wh.z < 0) wh = -wh;

            // Same side?
            if (dot(wo, wh) * dot(wi, wh) > 0) return make_float3(0);


            Float sqrtDenom = dot(wo, wh) + eta * dot(wi, wh);
            Float factor = (mode == TransportMode::Radiance) ? (1 / eta) : 1;

            return (make_float3(1.f) - Fr) * T *
                std::abs(DisneyMicrofacetDistribution::D(wh, alphax, alphay) * DisneyMicrofacetDistribution::G(wo, wi, alphax, alphay) * eta * eta *
                    AbsDot(wi, wh) * AbsDot(wo, wh) * factor * factor /
                    (cosThetaI * cosThetaO * sqrtDenom * sqrtDenom));
        };        

        static Float Pdf_DisneyMicroDist(const Vector3f& wo, const Vector3f& wi, const float& etaA, const float& etaB, const float& alphax, const float& alphay) {
            // Compute $\wh$ from $\wo$ and $\wi$ for microfacet transmission
            Float eta = CosTheta(wo) > 0 ? (etaB / etaA) : (etaA / etaB);
            Vector3f wh = normalize(wo + wi * eta);

            if (dot(wo, wh) * dot(wi, wh) > 0) return 0;

            // Compute change of variables _dwh\_dwi_ for microfacet transmission
            Float sqrtDenom = dot(wo, wh) + eta * dot(wi, wh);
            Float dwh_dwi =
                abs((eta * eta * dot(wi, wh)) / (sqrtDenom * sqrtDenom));
            return DisneyMicrofacetDistribution::Pdf(wo, wh, alphax, alphay) * dwh_dwi;
        }

        static Spectrum Sample_DisneyMicroDist(const Spectrum& T, const Vector3f& wo, Vector3f* wi, const Point2f u, const float& etaA, const float& etaB, const float& alphax, const float& alphay, float& pdf, const Spectrum& Fr, TransportMode mode)
        {
            if (wo.z == 0) return make_float3(0.);
            Vector3f wh = TrowbridgeReitzDistribution::Sample_wh(wo, u, alphax, alphay);
            if (dot(wo, wh) < 0) return make_float3(0.);  // Should be rare

            Float eta = CosTheta(wo) > 0 ? (etaA / etaB) : (etaB / etaA);
            if (!Refract(wo, wh, eta, wi)) return make_float3(0);

            pdf = MicrofacetTransmission::Pdf_DisneyMicroDist(wo, *wi, etaA, etaB, alphax, alphay);
            return MicrofacetTransmission::Eval_DisneyMicroDist(T, wo, *wi, etaA, etaB, alphax, alphay, Fr, mode);
        }

        static Spectrum Eval_TrowReitzDist(const Spectrum& T, const Vector3f& wo, const Vector3f& wi, const float& etaA, const float& etaB, const float& alphax, const float& alphay, const Spectrum& Fr, TransportMode mode) {
            if (SameHemisphere(wo, wi)) return make_float3(0);  // transmission only

            Float cosThetaO = CosTheta(wo);
            Float cosThetaI = CosTheta(wi);
            if (cosThetaI == 0 || cosThetaO == 0) return make_float3(0);

            // Compute $\wh$ from $\wo$ and $\wi$ for microfacet transmission
            Float eta = CosTheta(wo) > 0 ? (etaB / etaA) : (etaA / etaB);
            Vector3f wh = normalize(wo + wi * eta);
            if (wh.z < 0) wh = -wh;

            // Same side?
            if (dot(wo, wh) * dot(wi, wh) > 0) return make_float3(0);


            Float sqrtDenom = dot(wo, wh) + eta * dot(wi, wh);
            Float factor = (mode == TransportMode::Radiance) ? (1 / eta) : 1;

            float lambda_wo = TrowbridgeReitzDistribution::Lambda(wo, alphax, alphay);
            float lambda_wi = TrowbridgeReitzDistribution::Lambda(wi, alphax, alphay);

            return (make_float3(1.f) - Fr) * T *
                std::abs(TrowbridgeReitzDistribution::D(wh, alphax, alphay) * MicrofacetDistribution::G(lambda_wo, lambda_wi) * eta * eta *
                    AbsDot(wi, wh) * AbsDot(wo, wh) * factor * factor /
                    (cosThetaI * cosThetaO * sqrtDenom * sqrtDenom));
        };

        static Float Pdf_TrowReitzDist(const Vector3f& wo, const Vector3f& wi, const float& etaA, const float& etaB, const float& alphax, const float& alphay) {
            // Compute $\wh$ from $\wo$ and $\wi$ for microfacet transmission
            Float eta = CosTheta(wo) > 0 ? (etaB / etaA) : (etaA / etaB);
            Vector3f wh = normalize(wo + wi * eta);

            if (dot(wo, wh) * dot(wi, wh) > 0) return 0;

            // Compute change of variables _dwh\_dwi_ for microfacet transmission
            Float sqrtDenom = dot(wo, wh) + eta * dot(wi, wh);
            Float dwh_dwi =
                abs((eta * eta * dot(wi, wh)) / (sqrtDenom * sqrtDenom));
            return TrowbridgeReitzDistribution::Pdf(wo, wh, alphax, alphay) * dwh_dwi;
        }

        static Spectrum Sample_TrowReitzDist(const Spectrum& T, const Vector3f& wo, Vector3f* wi, const Point2f u, const float& etaA, const float& etaB, const float& alphax, const float& alphay, float& pdf, const Spectrum& Fr, TransportMode mode)
        {
            if (wo.z == 0) return make_float3(0.);
            Vector3f wh = TrowbridgeReitzDistribution::Sample_wh(wo, u, alphax, alphay);
            if (dot(wo, wh) < 0) return make_float3(0.);  // Should be rare

            Float eta = CosTheta(wo) > 0 ? (etaA / etaB) : (etaB / etaA);
            if (!Refract(wo, wh, eta, wi)) return make_float3(0);

            pdf = MicrofacetTransmission::Pdf_TrowReitzDist(wo, *wi, etaA, etaB, alphax, alphay);
            return MicrofacetTransmission::Eval_TrowReitzDist(T, wo, *wi, etaA, etaB, alphax, alphay, Fr, mode);
        }
    };

    ///////////////////////////////////////////////////////////////////////////
// DisneyMaterial

    // DisneyMaterial Declarations
    struct DisneyMaterial{    
        
        Spectrum color;
        float metallic, eta;
        float roughness, specularTint, anisotropic, sheen;
        float sheenTint, clearcoat, clearcoatGloss;
        float specTrans;
        Spectrum scatterDistance;
        bool thin;
        float flatness, diffTrans, bumpMap;
    };

    float MicrofacetDistributionPDF(const float3& wo, const float3& wi, const float3& wh, const float alphax, const float alphay) {
        float D = 0;
        Float tan2Theta = Tan2Theta(wh);
        if (!isinf(tan2Theta)) {
            const Float cos4Theta = Cos2Theta(wh) * Cos2Theta(wh);
            Float e = (Cos2Phi(wh) / (alphax * alphax) + Sin2Phi(wh) / (alphay * alphay)) *
                tan2Theta;
            float D = 1.f / (Pi * alphax * alphay * cos4Theta * (1.f + e) * (1.f + e));
        }

        float Lambda = 0;
        float absTanTheta = abs(TanTheta(wo));
        if (!isinf(absTanTheta)) {
            Float alpha = std::sqrt(Cos2Phi(wo) * alphax * alphax + Sin2Phi(wo) * alphay * alphay);
            Float alpha2Tan2Theta = (alpha * absTanTheta) * (alpha * absTanTheta);
            Lambda = (-1 + sqrt(1.f + alpha2Tan2Theta)) / 2;
        }
        float G1 = 1 / (1 + Lambda);

        return (D * G1 * AbsDot(wo, wh) / AbsCosTheta(wo));
    }

    void DisneyPDF(DisneyMaterial& material, const float3& wo, const float3& wi, float& pdf) {

        // Perform bump mapping with _bumpMap_, if present
        //if (bumpMap) Bump(bumpMap, si);

        // Evaluate textures for _DisneyMaterial_ material and allocate BRDF

        // Diffuse
        Spectrum c = material.color;
        Float metallicWeight = material.metallic;
        Float e = material.eta;
        Float strans = material.specTrans;
        Float diffuseWeight = (1 - metallicWeight) * (1 - strans);
        Float dt = material.diffTrans / 2;  // 0: all diffuse is reflected -> 1, transmitted
        Float rough = material.roughness;
        Float lum = Luminance(c);
        // normalize lum. to isolate hue+sat
        Spectrum Ctint = lum > 0 ? (c / lum) : make_float3(1.f);

        float etaA = 1.0f;
        float etaB = e;

        Float sheenWeight = material.sheen;
        Spectrum Csheen;
        if (sheenWeight > 0) {
            Float stint = material.sheenTint;
            Csheen = Lerp(stint, make_float3(1.f), Ctint);
        }

        Vector3f wh = normalize(wo + wi);
        Float aspect = sqrt(1 - material.anisotropic * .9);
        Float alphax = max(Float(.001), sqr(rough) / aspect);
        Float alphay = max(Float(.001), sqr(rough) * aspect);

        //float microfacetPDF = MicrofacetDistributionPDF(wo, wi, wh, alphax, alphay);

        if (SameHemisphere(wo, wi)) {
            if (diffuseWeight > 0) {                

                if (material.thin) {
                    // Disney Diffuse
                    pdf += DisneyDiffuse::Pdf(wi);

                    // Disney Fake SS
                    pdf += DisneyFakeSS::Pdf(wi);
                }
                else {
                    Spectrum sd = material.scatterDistance;
                    if (IsBlack(sd)) {
                        // Disney Diffuse
                        pdf += DisneyDiffuse::Pdf(wi);
                    }
                    else {
                        // Specular Transmission PDF = return 0;
                        // bssrdf DisneyBSSRDF
                    }                    
                }

                //Disney Retro
                pdf += DisneyRetro::Pdf(wi);

                // Disney Sheen
                if (sheenWeight > 0)
                    pdf += DisneySheen::Pdf(wi);
            }

            // clearcoat
            Float cc = material.clearcoat;
            if (cc > 0) {
                float gloss = Lerp(material.clearcoatGloss, .1f, .001f);
                pdf += DisneyClearcoat::Pdf(wo, wi, gloss);                
            }

            // Microfacet Reflection
            pdf += MicrofacetReflection::Pdf_DisneyMicroDist(wo, wi, alphax, alphay);
        }
        else
        {
            // Transmission
            if (strans > 0) {
                Spectrum T = strans * sqrt(c);
                if(material.thin){
                    pdf += MicrofacetTransmission::Pdf_TrowReitzDist(wo, wi, etaA, etaB, alphax, alphay);
                }
                else {
                    pdf += MicrofacetTransmission::Pdf_DisneyMicroDist(wo, wi, etaA, etaB, alphax, alphay);
                }
            }

            // Lambert Transmission
            if (material.thin) {
                pdf += LambertianTransmission::Pdf(wi);
            }
        }
    }

    float3 DisneyEval(DisneyMaterial& material, const float3& wo, const float3& wi, TransportMode mode) {
        float3 f = make_float3(0.f);

        Spectrum c = material.color;
        Float metallicWeight = material.metallic;
        Float e = material.eta;
        Float strans = material.specTrans;
        Float diffuseWeight = (1 - metallicWeight) * (1 - strans);
        Float dt = material.diffTrans / 2;  // 0: all diffuse is reflected -> 1, transmitted
        Float rough = material.roughness;
        Float lum = Luminance(c);
        // normalize lum. to isolate hue+sat
        Spectrum Ctint = lum > 0 ? (c / lum) : make_float3(1.f);

        float etaA = 1.0f;
        float etaB = e;

        Float sheenWeight = material.sheen;
        Spectrum Csheen;
        if (sheenWeight > 0) {
            Float stint = material.sheenTint;
            Csheen = Lerp(stint, make_float3(1.f), Ctint);
        }

        Vector3f wh = normalize(wo + wi);

        if (diffuseWeight > 0) {
            if (material.thin) {     
                Float flat = material.flatness;
                f += DisneyDiffuse::Eval(wo, wi, diffuseWeight * (1 - flat) * (1 - dt) * c);
                f += DisneyFakeSS::Eval(wo, wi, diffuseWeight * flat * (1 - dt) * c, rough);
            }
            else {
                Spectrum sd = material.scatterDistance;
                if (IsBlack(sd)) {
                    f += DisneyDiffuse::Eval(wo, wi, diffuseWeight * c);
                }
                else {
                    // Specular Transmission
                    // BSSRDF
                }
            }

            f += DisneyRetro::Eval(wo, wi, diffuseWeight * c, rough);

            if (sheenWeight > 0)
                f += DisneySheen::Eval(wo, wi, diffuseWeight * sheenWeight * Csheen);
        }

        Float aspect = std::sqrt(1 - material.anisotropic * .9);
        Float alphax = max(Float(.001), sqr(rough) / aspect);
        Float alphay = max(Float(.001), sqr(rough) * aspect);

        Float specTint = material.specularTint;
        Spectrum Cspec0 = Lerp(metallicWeight, SchlickR0FromEta(e) * Lerp(specTint, make_float3(1.f), Ctint), c);
        Spectrum F = DisneyFresnel::Evaluate(Cspec0, metallicWeight, e, dot(wi, Faceforward(wh, make_float3(0, 0, 1))));
        f += MicrofacetReflection::Eval_DisneyMicroDist(wo, wi, alphax, alphay, F);

        Float cc = material.clearcoat;
        if (cc > 0) {
            f += DisneyClearcoat::Eval(wo, wi, cc, Lerp(material.clearcoatGloss, .1, .001));
        }

        if (strans > 0) {
            Spectrum T = strans * sqrt(c);
            float3 Fr = FresnelDielectric::Evaluate(dot(wo, wh), etaA, etaB);
            if (material.thin) {                
                f += MicrofacetTransmission::Eval_TrowReitzDist(T, wo, wi, etaA, etaB, alphax, alphay, Fr, mode);
            }
            else {
                f += MicrofacetTransmission::Eval_DisneyMicroDist(T, wo, wi, etaA, etaB, alphax, alphay, Fr, mode);
            }
        }

        if (material.thin)
            f += LambertianTransmission::Eval(dt * c);

        return f;
    }

    float3 DisneySample(DisneyMaterial& material, const float3& wo, float3& wi, float& pdf, const float2& sample) {

        Spectrum c = material.color;
        Float metallicWeight = material.metallic;
        Float e = material.eta;
        Float strans = material.specTrans;
        Float diffuseWeight = (1 - metallicWeight) * (1 - strans);
        Float dt = material.diffTrans / 2;  // 0: all diffuse is reflected -> 1, transmitted
        Float rough = material.roughness;
        Float lum = Luminance(c);
        // normalize lum. to isolate hue+sat
        Spectrum Ctint = lum > 0 ? (c / lum) : make_float3(1.f);

        float etaA = 1.0f;
        float etaB = e;

        Float sheenWeight = material.sheen;
        Spectrum Csheen;
        if (sheenWeight > 0) {
            Float stint = material.sheenTint;
            Csheen = Lerp(stint, make_float3(1.f), Ctint);
        }

        Vector3f wh = normalize(wo + wi);

        Float Fo = SchlickWeight(AbsCosTheta(wo)),
            Fi = SchlickWeight(AbsCosTheta(wi));


        DisneyDiffuse::Sample(wo, wi, sample);
    }
}