#version 150 compatibility

#include "/lib/npu_policy.glsl"

uniform sampler2D colortex0;
uniform sampler2D colortex1;
uniform sampler2D colortex2;
uniform sampler2D colortex4;
uniform sampler2D depthtex0;
uniform sampler2D npuAssist;

uniform vec3 sunPosition;
uniform vec3 upPosition;
uniform mat4 gbufferProjectionInverse;
uniform mat4 gbufferModelViewInverse;
uniform float viewWidth;
uniform float viewHeight;

in vec2 texCoord;

// final consumes NPU-generated GI policy and turns it into:
// - short screen-space GI
// - water-only reflection shaping
// - volumetric fog and light scattering
// General SSR is intentionally disabled to avoid the stretched-geometry artifacts
// seen on opaque terrain and objects.
const bool ENABLE_WATER_SSR = true;
const bool ENABLE_BLOOM = true;

float saturate1(float value) {
    return clamp(value, 0.0, 1.0);
}

vec2 saturate2(vec2 value) {
    return clamp(value, vec2(0.0), vec2(1.0));
}

vec3 saturate3(vec3 value) {
    return clamp(value, vec3(0.0), vec3(1.0));
}

vec2 safeNormalize2(vec2 value) {
    float lengthSquared = dot(value, value);
    if (lengthSquared <= 0.000001) {
        return vec2(0.0, -1.0);
    }
    return value * inversesqrt(lengthSquared);
}

vec3 safeNormalize3(vec3 value) {
    float lengthSquared = dot(value, value);
    if (lengthSquared <= 0.000001) {
        return vec3(0.0, 1.0, 0.0);
    }
    return value * inversesqrt(lengthSquared);
}

float luma(vec3 color) {
    return dot(color, vec3(0.2126, 0.7152, 0.0722));
}

int budgetTapCount(float budget, int minTaps, int maxTaps) {
    float clampedBudget = saturate1(budget);
    return int(floor(mix(float(minTaps), float(maxTaps) + 0.999, clampedBudget)));
}

vec3 acesTonemap(vec3 color) {
    const float a = 2.51;
    const float b = 0.03;
    const float c = 2.43;
    const float d = 0.59;
    const float e = 0.14;
    return saturate3((color * (a * color + b)) / (color * (c * color + d) + e));
}

vec3 toSrgb(vec3 color) {
    return pow(max(color, vec3(0.0)), vec3(1.0 / 2.2));
}

vec3 projectAndDivide(mat4 projectionMatrix, vec3 position) {
    vec4 homPos = projectionMatrix * vec4(position, 1.0);
    return homPos.xyz / max(homPos.w, 0.0001);
}

vec3 sampleHdr(vec2 uv) {
    return texture2D(colortex0, saturate2(uv)).rgb;
}

float sampleSceneDepth(vec2 uv) {
    return texture2D(depthtex0, saturate2(uv)).r;
}

float sampleEdgeFade(vec2 uv, float margin) {
    vec2 clampedUv = saturate2(uv);
    vec2 edgeDistance = min(clampedUv, vec2(1.0) - clampedUv);
    return smoothstep(0.0, max(margin, 0.0001), min(edgeDistance.x, edgeDistance.y));
}

vec3 decodeNormal(vec3 encoded) {
    return normalize(encoded * 2.0 - 1.0);
}

vec3 sampleWorldNormal(vec2 uv) {
    return decodeNormal(texture2D(colortex1, saturate2(uv)).rgb);
}

vec3 getViewPos(vec2 uv, float depth) {
    return projectAndDivide(gbufferProjectionInverse, vec3(uv, depth) * 2.0 - 1.0);
}

vec3 getViewNormal(vec3 worldNormal) {
    return safeNormalize3(transpose(mat3(gbufferModelViewInverse)) * worldNormal);
}

float sampleDepthDiscontinuity(vec2 uv, float centerDepth) {
    vec2 pixel = vec2(1.0 / max(viewWidth, 1.0), 1.0 / max(viewHeight, 1.0));
    float maxGap = 0.0;
    maxGap = max(maxGap, abs(centerDepth - sampleSceneDepth(uv + vec2(pixel.x, 0.0))));
    maxGap = max(maxGap, abs(centerDepth - sampleSceneDepth(uv - vec2(pixel.x, 0.0))));
    maxGap = max(maxGap, abs(centerDepth - sampleSceneDepth(uv + vec2(0.0, pixel.y))));
    maxGap = max(maxGap, abs(centerDepth - sampleSceneDepth(uv - vec2(0.0, pixel.y))));
    float farFactor = smoothstep(0.60, 0.985, centerDepth);
    float minThreshold = mix(0.0012, 0.00016, farFactor);
    float maxThreshold = mix(0.022, 0.0045, farFactor);
    return smoothstep(minThreshold, maxThreshold, maxGap);
}

float sampleNormalDiscontinuity(vec2 uv, vec3 centerNormal, float centerDepth) {
    vec2 pixel = vec2(1.0 / max(viewWidth, 1.0), 1.0 / max(viewHeight, 1.0));
    float minDot = 1.0;
    minDot = min(minDot, dot(centerNormal, sampleWorldNormal(uv + vec2(pixel.x, 0.0))));
    minDot = min(minDot, dot(centerNormal, sampleWorldNormal(uv - vec2(pixel.x, 0.0))));
    minDot = min(minDot, dot(centerNormal, sampleWorldNormal(uv + vec2(0.0, pixel.y))));
    minDot = min(minDot, dot(centerNormal, sampleWorldNormal(uv - vec2(0.0, pixel.y))));
    float farFactor = smoothstep(0.60, 0.985, centerDepth);
    float minThreshold = mix(0.04, 0.02, farFactor);
    float maxThreshold = mix(0.42, 0.22, farFactor);
    return smoothstep(minThreshold, maxThreshold, 1.0 - clamp(minDot, -1.0, 1.0));
}

float sampleLumaDiscontinuity(vec2 uv, float centerLuma, float centerDepth) {
    vec2 pixel = vec2(1.0 / max(viewWidth, 1.0), 1.0 / max(viewHeight, 1.0));
    float maxGap = 0.0;
    maxGap = max(maxGap, abs(centerLuma - luma(sampleHdr(uv + vec2(pixel.x, 0.0)))));
    maxGap = max(maxGap, abs(centerLuma - luma(sampleHdr(uv - vec2(pixel.x, 0.0)))));
    maxGap = max(maxGap, abs(centerLuma - luma(sampleHdr(uv + vec2(0.0, pixel.y)))));
    maxGap = max(maxGap, abs(centerLuma - luma(sampleHdr(uv - vec2(0.0, pixel.y)))));
    float farFactor = smoothstep(0.60, 0.985, centerDepth);
    float minThreshold = mix(0.06, 0.015, farFactor);
    float maxThreshold = mix(0.42, 0.14, farFactor);
    return smoothstep(minThreshold, maxThreshold, maxGap);
}

vec3 traceGiPath(
    vec2 uv,
    vec2 trajectoryDir,
    float curvature,
    float spread,
    float energy,
    vec3 centerColor,
    float centerDepth,
    vec3 centerNormal,
    int sampleCount
) {
    vec2 tangent = safeNormalize2(trajectoryDir);
    vec2 normal = vec2(-tangent.y, tangent.x);
    float nearField = 1.0 - smoothstep(0.58, 0.985, centerDepth);
    float stepScale = mix(0.0012, 0.0048 + spread * 0.010, nearField);
    float curveScale = mix(0.0010, 0.0046 + spread * 0.006, nearField);
    vec3 accumulated = vec3(0.0);
    float weightSum = 0.0;

    for (int index = 1; index <= 4; ++index) {
        if (index > sampleCount) {
            break;
        }
        float t = float(index) / 4.0;
        vec2 offset = tangent * stepScale * float(index) + normal * curvature * t * t * curveScale;
        vec2 sampleUv = saturate2(uv + offset);
        vec3 sampleColor = sampleHdr(sampleUv);
        float sampleDepth = sampleSceneDepth(sampleUv);
        vec3 sampleNormal = sampleWorldNormal(sampleUv);
        float depthTolerance = mix(0.0008, 0.016 + spread * 0.020, nearField);
        float depthMatch = 1.0 - smoothstep(0.0002, depthTolerance, abs(sampleDepth - centerDepth));
        float normalMatch = pow(clamp(dot(centerNormal, sampleNormal), 0.0, 1.0), 2.5);
        float colorMatch = 1.0 - smoothstep(0.08, 0.55 + spread * 0.30, distance(sampleColor, centerColor));
        float weight = exp(-t * 1.0)
            * smoothstep(0.35, 1.45 + energy * 1.40, luma(sampleColor))
            * mix(0.04, 1.0, depthMatch)
            * mix(0.08, 1.0, normalMatch)
            * mix(0.02, 1.0, colorMatch);
        accumulated += sampleColor * weight;
        weightSum += weight;
    }

    if (weightSum <= 0.0001) {
        return vec3(0.0);
    }
    return accumulated / weightSum;
}

float gaussianWeight(float x, float sigma) {
    return exp(-(x * x) / max(2.0 * sigma * sigma, 0.0001));
}

vec3 extractHighlight(vec3 color, float threshold, float knee) {
    float brightness = luma(color);
    float soft = smoothstep(threshold - knee, threshold + knee, brightness);
    return color * soft * max(brightness - threshold + knee, 0.0);
}

void accumulateReflectionTap(
    vec2 mirrorUv,
    vec2 offset,
    float baseWeight,
    vec3 centerSample,
    float centerLuma,
    float centerDepth,
    float roughness,
    float edgeFade,
    inout vec3 accumulated,
    inout float weightSum
) {
    vec2 sampleUv = saturate2(mirrorUv + offset);
    vec3 sampleColor = sampleHdr(sampleUv);
    float sampleLuma = luma(sampleColor);
    float sampleDepth = sampleSceneDepth(sampleUv);
    float lumaMatch = 1.0 - smoothstep(0.08, 0.72 + roughness * 0.28, abs(sampleLuma - centerLuma));
    float chromaMatch = 1.0 - smoothstep(0.16, 1.10 + roughness * 0.35, distance(sampleColor, centerSample));
    float depthMatch = 1.0 - smoothstep(0.0015, 0.028 + roughness * 0.040, abs(sampleDepth - centerDepth));
    float tapEdgeFade = sampleEdgeFade(sampleUv, 0.09);
    float weight = baseWeight
        * mix(0.58, 1.0, lumaMatch)
        * mix(0.72, 1.0, chromaMatch)
        * mix(0.20, 1.0, depthMatch)
        * mix(0.25, 1.0, tapEdgeFade)
        * edgeFade;
    accumulated += sampleColor * weight;
    weightSum += weight;
}

vec3 sampleWaterReflectionField(
    vec2 uv,
    vec2 flow,
    vec2 trajectoryDir,
    float roughness,
    float energy,
    float surfaceFacing,
    int tapCount
) {
    vec2 flowDir = safeNormalize2(flow * 0.64 + trajectoryDir * 0.36);
    vec2 sideDir = vec2(-flowDir.y, flowDir.x);
    vec2 mirrorUv = saturate2(vec2(uv.x, 1.0 - uv.y));
    float sigma = mix(1.10, 0.74, surfaceFacing) + roughness * 0.55 + energy * 0.05;

    vec2 axisLong = flowDir * (0.0025 + roughness * 0.012 + energy * 0.0035);
    vec2 axisShort = sideDir * (0.0015 + roughness * 0.008 + (1.0 - surfaceFacing) * 0.0025);
    vec2 axisDiag = safeNormalize2(flowDir + sideDir * 0.72) * (0.002 + roughness * 0.010 + energy * 0.003);
    vec2 axisLongFar = axisLong * 1.55;

    vec3 centerSample = sampleHdr(mirrorUv);
    float centerWeight = gaussianWeight(0.0, sigma);
    float centerLuma = luma(centerSample);
    float centerDepth = sampleSceneDepth(mirrorUv);
    float edgeFade = sampleEdgeFade(mirrorUv, 0.11);
    vec3 accumulated = centerSample * centerWeight;
    float weightSum = centerWeight;

    if (tapCount >= 1) accumulateReflectionTap(mirrorUv, axisLong, gaussianWeight(1.0, sigma), centerSample, centerLuma, centerDepth, roughness, edgeFade, accumulated, weightSum);
    if (tapCount >= 2) accumulateReflectionTap(mirrorUv, -axisLong, gaussianWeight(1.0, sigma), centerSample, centerLuma, centerDepth, roughness, edgeFade, accumulated, weightSum);
    if (tapCount >= 3) accumulateReflectionTap(mirrorUv, axisShort, gaussianWeight(1.0, sigma) * 0.90, centerSample, centerLuma, centerDepth, roughness, edgeFade, accumulated, weightSum);
    if (tapCount >= 4) accumulateReflectionTap(mirrorUv, -axisShort, gaussianWeight(1.0, sigma) * 0.90, centerSample, centerLuma, centerDepth, roughness, edgeFade, accumulated, weightSum);
    if (tapCount >= 5) accumulateReflectionTap(mirrorUv, axisDiag, gaussianWeight(1.25, sigma), centerSample, centerLuma, centerDepth, roughness, edgeFade, accumulated, weightSum);
    if (tapCount >= 6) accumulateReflectionTap(mirrorUv, -axisDiag, gaussianWeight(1.25, sigma), centerSample, centerLuma, centerDepth, roughness, edgeFade, accumulated, weightSum);
    if (tapCount >= 7) accumulateReflectionTap(mirrorUv, (axisLong + axisShort) * 0.78, gaussianWeight(1.32, sigma), centerSample, centerLuma, centerDepth, roughness, edgeFade, accumulated, weightSum);
    if (tapCount >= 8) accumulateReflectionTap(mirrorUv, -((axisLong + axisShort) * 0.78), gaussianWeight(1.32, sigma), centerSample, centerLuma, centerDepth, roughness, edgeFade, accumulated, weightSum);
    if (tapCount >= 9) accumulateReflectionTap(mirrorUv, axisLongFar, gaussianWeight(1.55, sigma) * 0.52, centerSample, centerLuma, centerDepth, roughness, edgeFade, accumulated, weightSum);
    if (tapCount >= 10) accumulateReflectionTap(mirrorUv, -axisLongFar, gaussianWeight(1.55, sigma) * 0.52, centerSample, centerLuma, centerDepth, roughness, edgeFade, accumulated, weightSum);

    vec3 blurred = accumulated / max(weightSum, 0.0001);
    float blurMix = (0.18 + roughness * 0.20 + (1.0 - surfaceFacing) * 0.08) * edgeFade;
    return mix(centerSample, blurred, blurMix);
}

vec3 diluteReflectionColor(
    vec3 reflected,
    vec3 baseHdr,
    vec3 skyTint,
    vec3 receiverTint,
    float roughness,
    float reflectivity,
    float fresnel,
    float shadowReceiver
) {
    vec3 mediumColor = mix(baseHdr, skyTint, 0.24 + roughness * 0.22);
    mediumColor *= mix(vec3(1.0), receiverTint, 0.16 + shadowReceiver * 0.20);

    float dilution = clamp(0.46 + roughness * 0.30 + (1.0 - reflectivity) * 0.14 - fresnel * 0.08, 0.32, 0.82);
    vec3 softened = reflected * (0.54 + reflectivity * 0.12) + mediumColor * 0.46;
    vec3 diluted = mix(reflected, softened, dilution);
    float preserveBrights = smoothstep(0.80, 2.10, luma(reflected));
    return mix(diluted, reflected, preserveBrights * (0.10 + fresnel * 0.18));
}

vec3 sampleWaterSparkle(
    vec2 uv,
    vec2 flow,
    vec2 trajectoryDir,
    float roughness,
    float energy
) {
    vec2 dir = safeNormalize2(flow * 0.65 + trajectoryDir * 0.55);
    vec2 side = vec2(-dir.y, dir.x);
    vec2 base = vec2(uv.x, 1.0 - uv.y);
    float radius = 0.003 + (1.0 - roughness) * 0.010 + energy * 0.004;

    vec3 sampleA = sampleHdr(base + dir * radius);
    vec3 sampleB = sampleHdr(base - dir * radius * 0.75);
    vec3 sampleC = sampleHdr(base + side * radius * 0.85);
    vec3 sampleD = sampleHdr(base - side * radius * 0.65);
    vec3 sparkle = (sampleA + sampleB + sampleC + sampleD) * 0.25;

    float sparkleMask = smoothstep(0.85, 1.75 + energy * 1.25, luma(sparkle));
    return sparkle * sparkleMask;
}

vec3 sampleWaterReflectionBloom(
    vec2 mirrorUv,
    vec2 reflectionAxis,
    vec2 waterFlow,
    float roughness,
    float energy,
    float edgeFade
) {
    vec2 axis = safeNormalize2(reflectionAxis * 0.72 + waterFlow * 0.28);
    vec2 side = vec2(-axis.y, axis.x);
    float radiusLong = 0.004 + roughness * 0.016 + energy * 0.004;
    float radiusShort = 0.002 + roughness * 0.008;
    vec2 longStep = axis * radiusLong;
    vec2 shortStep = side * radiusShort;
    vec2 diagStep = safeNormalize2(axis + side * 0.62) * (radiusLong * 0.82);

    vec3 center = extractHighlight(sampleHdr(mirrorUv), 0.78, 0.18) * 0.28;
    vec3 bloom = center;
    bloom += extractHighlight(sampleHdr(mirrorUv + longStep), 0.82, 0.20) * 0.18;
    bloom += extractHighlight(sampleHdr(mirrorUv - longStep), 0.82, 0.20) * 0.18;
    bloom += extractHighlight(sampleHdr(mirrorUv + shortStep), 0.88, 0.22) * 0.10;
    bloom += extractHighlight(sampleHdr(mirrorUv - shortStep), 0.88, 0.22) * 0.10;
    bloom += extractHighlight(sampleHdr(mirrorUv + diagStep), 0.86, 0.20) * 0.13;
    bloom += extractHighlight(sampleHdr(mirrorUv - diagStep), 0.86, 0.20) * 0.13;
    return bloom * edgeFade;
}

vec3 postProcessWaterReflection(
    vec3 reflected,
    vec3 baseWaterHdr,
    vec2 mirrorUv,
    vec2 reflectionAxis,
    vec2 waterFlow,
    vec3 skyTint,
    vec3 receiverTint,
    float roughness,
    float reflectivity,
    float fresnel,
    float surfaceFacing,
    float energy,
    float budget,
    float edgeFade
) {
    vec3 bloom = ENABLE_BLOOM
        ? sampleWaterReflectionBloom(mirrorUv, reflectionAxis, waterFlow, roughness, energy, edgeFade)
        : vec3(0.0);
    float highlightMask = smoothstep(0.52, 1.55 + energy * 0.90, luma(reflected));
    float clarity = clamp(0.16 + (1.0 - roughness) * 0.22 + budget * 0.10, 0.0, 0.34);
    vec3 sharpened = mix(reflected, reflected * (1.04 + fresnel * 0.06) - baseWaterHdr * 0.04, clarity);
    vec3 sheen = skyTint * (0.03 + fresnel * 0.05 + (1.0 - surfaceFacing) * 0.03) * edgeFade;
    vec3 mediumLift = receiverTint * bloom * (0.12 + energy * 0.10 + reflectivity * 0.06);
    vec3 post = sharpened;
    post += bloom * skyTint * (0.16 + highlightMask * 0.24 + reflectivity * 0.08);
    post += mediumLift;
    post += sheen;
    post = mix(post, reflected, roughness * 0.18);
    return max(post, vec3(0.0));
}

float sampleSunShaftOcclusion(vec2 uv, vec2 sunScreenDir, float radiusPixels, int sampleCount) {
    if (length(sunScreenDir) < 0.0001) {
        return 1.0;
    }

    vec2 dir = normalize(-sunScreenDir);
    vec2 pixel = vec2(1.0 / max(viewWidth, 1.0), 1.0 / max(viewHeight, 1.0));
    float transmittance = 0.0;
    for (int index = 1; index <= 5; ++index) {
        if (index > sampleCount) {
            break;
        }
        float travel = float(index) * radiusPixels;
        vec2 sampleUv = saturate2(uv + dir * pixel * travel);
        float sampleDepth = texture2D(depthtex0, sampleUv).r;
        transmittance += sampleDepth > 0.999 ? 1.0 : 0.0;
    }
    return transmittance / max(float(sampleCount), 1.0);
}

vec3 computeVolumetricFog(
    vec2 uv,
    vec3 hdr,
    vec3 viewPos,
    vec2 bakedLight,
    vec2 trajectoryDir,
    float energy,
    float spread,
    float shadowReceiver,
    int shaftSamples
) {
    float distance = length(viewPos);
    float fogAmount = 1.0 - exp(-distance * (0.014 + spread * 0.010));
    float sunHeight = saturate1(dot(normalize(sunPosition), normalize(upPosition)) * 0.5 + 0.5);
    vec3 skyFog = mix(vec3(0.18, 0.24, 0.34), vec3(0.46, 0.58, 0.72), sunHeight);
    vec3 torchFog = vec3(1.15, 0.72, 0.28);

    float sunShaft = sampleSunShaftOcclusion(uv, trajectoryDir, mix(12.0, 36.0, spread), shaftSamples);
    float sunScatter = sunShaft * (0.14 + energy * 0.18) * bakedLight.y;
    float blockScatter = bakedLight.x * bakedLight.x * (0.08 + shadowReceiver * 0.14);

    vec3 fogColor = skyFog * sunScatter + torchFog * blockScatter;
    vec3 baseFog = mix(vec3(0.0), fogColor, fogAmount);
    vec3 localExtinction = hdr * mix(0.96, 0.78, fogAmount * 0.55);
    return mix(localExtinction, localExtinction + baseFog, saturate1(fogAmount));
}

vec3 presentColor(vec3 graded, vec2 uv) {
    vec3 toneMapped = acesTonemap(graded);
    float vignette = 1.0 - smoothstep(0.30, 1.45, dot(uv * 2.0 - 1.0, uv * 2.0 - 1.0));
    toneMapped *= mix(0.88, 1.0, vignette);
    return toSrgb(toneMapped);
}

void main() {
    vec2 uv = texCoord;
    vec3 hdr = texture2D(colortex0, uv).rgb;
    float depth = texture2D(depthtex0, uv).r;

    if (depth >= 0.999999) {
        gl_FragData[0] = vec4(presentColor(hdr * 0.35, uv), 1.0);
        return;
    }

    vec3 worldNormal = decodeNormal(texture2D(colortex1, uv).rgb);
    vec4 lightData = texture2D(colortex2, uv);
    vec2 bakedLight = lightData.rg;
    float entityMask = lightData.b;
    vec4 waterData = texture2D(colortex4, uv);
    vec3 viewPos = getViewPos(uv, depth);
    vec3 viewDir = safeNormalize3(-viewPos);
    vec3 viewNormal = getViewNormal(worldNormal);

    float distanceWeight = smoothstep(0.12, 0.98, depth);
    float baseSceneLuma = luma(hdr);
    NpuPolicy policy = mergeScreenNpuPolicy(npuAssist, uv, depth, baseSceneLuma, bakedLight);
    vec2 npuDirection = policy.trajectory;
    float npuEnergy = policy.energy;
    float curvature = policy.curvature;
    vec2 sunScreenDir = safeNormalize2(vec2(sunPosition.x, -sunPosition.y));
    float farFieldSuppression = smoothstep(0.72, 0.985, depth);
    float depthDiscontinuity = sampleDepthDiscontinuity(uv, depth);
    float normalDiscontinuity = sampleNormalDiscontinuity(uv, worldNormal, depth);
    float lumaDiscontinuity = sampleLumaDiscontinuity(uv, baseSceneLuma, depth);
    float silhouetteSuppression = saturate1(max(
        entityMask,
        max(
            depthDiscontinuity * 0.92 + normalDiscontinuity * 0.70,
            lumaDiscontinuity * 0.82 + farFieldSuppression * 0.60
        )
    ));
    float assistFade = saturate1(silhouetteSuppression * 0.88 + farFieldSuppression * 0.36);
    float effectiveBudget = mix(policy.budget, min(policy.budget, 0.26), assistFade);
    float effectiveEnergy = mix(npuEnergy, min(npuEnergy, 0.16 + baseSceneLuma * 0.10), assistFade);
    float npuSpread = mix(0.14, 0.68, effectiveBudget) * mix(1.0, 0.60, farFieldSuppression);
    int giSamples = budgetTapCount(effectiveBudget, 1, 3);
    int shaftSamples = budgetTapCount(effectiveBudget, 1, 4);
    int reflectionTapCount = budgetTapCount(effectiveBudget, 4, 10);
    vec2 trajectoryDir = safeNormalize2(mix(npuDirection, sunScreenDir, 0.28 + (1.0 - effectiveBudget) * 0.10 + silhouetteSuppression * 0.26 + farFieldSuppression * 0.22));
    float shadowReceiver = smoothstep(0.20, 0.86, 1.0 - baseSceneLuma);
    vec3 groundTint = normalize(max(pow(max(hdr, vec3(0.0)), vec3(0.42)) + vec3(0.08), vec3(0.0001)));
    vec3 receiverTint = mix(vec3(1.0), groundTint * (1.05 + shadowReceiver * 0.35), 0.40 + shadowReceiver * 0.36);

    vec3 giTrace = traceGiPath(uv, trajectoryDir, curvature, npuSpread, effectiveEnergy, hdr, depth, worldNormal, giSamples);
    float giMask = (1.0 - silhouetteSuppression * 0.92) * (1.0 - farFieldSuppression * 0.76) * (1.0 - lumaDiscontinuity * 0.38);
    giTrace *= giMask;
    float giAmount = (0.012 + effectiveEnergy * 0.18 + shadowReceiver * 0.08 + effectiveBudget * 0.10) * giMask;
    hdr += giTrace * receiverTint * giAmount;

    float exposure = mix(0.98, 1.24, effectiveEnergy) * mix(0.97, 1.04, distanceWeight) * mix(0.99, 1.06, effectiveBudget);
    exposure = mix(exposure, mix(0.98, 1.06, baseSceneLuma), silhouetteSuppression * 0.34 + farFieldSuppression * 0.14);
    vec3 graded = hdr * exposure;
    vec3 palette = normalize(max(vec3(0.24 + policy.rawAssist.r, 0.28 + policy.rawAssist.b, 0.22 + policy.rawAssist.g), vec3(0.0001)));
    graded *= mix(vec3(0.985), vec3(0.90) + palette * 0.12, (0.04 + effectiveBudget * 0.10) * (1.0 - assistFade * 0.72));

    float waterReflectivity = waterData.b;
    float waterRoughness = waterData.a;
    if (ENABLE_WATER_SSR && waterReflectivity > 0.05) {
        vec3 baseWaterHdr = hdr;
        float reflectionEdgeFade = sampleEdgeFade(vec2(uv.x, 1.0 - uv.y), 0.12);
        vec2 mirrorUv = saturate2(vec2(uv.x, 1.0 - uv.y));
        vec2 waterFlow = waterData.rg * 2.0 - 1.0;
        float sunHeight = saturate1(dot(normalize(sunPosition), normalize(upPosition)) * 0.5 + 0.5);
        vec3 waterNormal = safeNormalize3(vec3(
            waterFlow.x * (0.76 + npuSpread * 0.14) + trajectoryDir.x * 0.14,
            1.36 - waterRoughness * 0.40,
            waterFlow.y * (0.76 + npuSpread * 0.14) + trajectoryDir.y * 0.14
        ));
        vec3 waterReflectionDir = reflect(-viewDir, waterNormal);
        float surfaceFacing = saturate1(abs(viewNormal.y));
        vec2 reflectionAxis = safeNormalize2(waterReflectionDir.xy * 0.50 + waterFlow * 0.34 + trajectoryDir * 0.16);
        float fresnel = 0.02 + 0.98 * pow(1.0 - saturate1(dot(waterNormal, viewDir)), 5.0);

        vec3 reflected = sampleWaterReflectionField(uv, waterFlow, reflectionAxis, waterRoughness, effectiveEnergy, surfaceFacing, reflectionTapCount);
        vec3 sparkle = ENABLE_BLOOM ? sampleWaterSparkle(uv, waterFlow, reflectionAxis, waterRoughness, npuEnergy) : vec3(0.0);
        vec3 waterTrail = traceGiPath(
            uv,
            safeNormalize2(waterFlow + trajectoryDir * 0.28),
            curvature,
            0.14 + effectiveBudget * 0.32,
            effectiveEnergy,
            hdr,
            depth,
            worldNormal,
            max(1, giSamples - 1)
        );
        waterTrail *= giMask;

        vec3 skyTint = mix(vec3(0.24, 0.38, 0.55), vec3(0.72, 0.86, 1.05), sunHeight);
        float horizonMix = clamp(0.34 + waterRoughness * 0.28 + (1.0 - abs(reflectionAxis.y)) * 0.14, 0.0, 1.0);
        reflected = mix(reflected, skyTint, horizonMix);
        reflected += waterTrail * receiverTint * (0.08 + effectiveEnergy * 0.12 + shadowReceiver * 0.10);
        reflected += sparkle * skyTint * (0.08 + (1.0 - waterRoughness) * 0.18 + effectiveEnergy * 0.05);
        reflected = diluteReflectionColor(reflected, baseWaterHdr, skyTint, receiverTint, waterRoughness, waterReflectivity, fresnel, shadowReceiver);
        reflected = postProcessWaterReflection(
            reflected,
            baseWaterHdr,
            mirrorUv,
            reflectionAxis,
            waterFlow,
            skyTint,
            receiverTint,
            waterRoughness,
            waterReflectivity,
            fresnel,
            surfaceFacing,
            effectiveEnergy,
            effectiveBudget,
            reflectionEdgeFade
        );

        vec3 viewSunDir = safeNormalize3(normalize(sunPosition));
        float glint = ENABLE_BLOOM
            ? pow(saturate1(dot(waterReflectionDir, safeNormalize3(-viewSunDir))), 18.0 + (1.0 - waterRoughness) * 22.0) * (0.08 + sunHeight * 0.22)
            : 0.0;
        float reflectionMix = clamp(waterReflectivity * fresnel * (0.34 + effectiveBudget * 0.18) * reflectionEdgeFade, 0.0, 0.46);
        hdr = mix(baseWaterHdr, reflected, reflectionMix);
        hdr += skyTint * glint * (0.05 + waterReflectivity * 0.20);
        graded = hdr * exposure;
        graded *= mix(vec3(0.985), vec3(0.90) + palette * 0.12, (0.04 + effectiveBudget * 0.10) * (1.0 - assistFade * 0.72));
    }

    graded = computeVolumetricFog(uv, graded, viewPos, bakedLight, trajectoryDir, effectiveEnergy, npuSpread, shadowReceiver, shaftSamples);
    graded *= 0.35;
    gl_FragData[0] = vec4(presentColor(graded, uv), 1.0);
}
