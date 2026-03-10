#version 150 compatibility

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

vec3 decodeNormal(vec3 encoded) {
    return normalize(encoded * 2.0 - 1.0);
}

vec3 getViewPos(vec2 uv, float depth) {
    return projectAndDivide(gbufferProjectionInverse, vec3(uv, depth) * 2.0 - 1.0);
}

vec3 getViewNormal(vec3 worldNormal) {
    return safeNormalize3(transpose(mat3(gbufferModelViewInverse)) * worldNormal);
}

vec3 traceGiPath(vec2 uv, vec2 trajectoryDir, float curvature, float spread, float energy) {
    vec2 tangent = safeNormalize2(trajectoryDir);
    vec2 normal = vec2(-tangent.y, tangent.x);
    float stepScale = 0.006 + spread * 0.016;
    vec3 accumulated = vec3(0.0);
    float weightSum = 0.0;

    for (int index = 1; index <= 4; ++index) {
        float t = float(index) / 4.0;
        vec2 offset = tangent * stepScale * float(index) + normal * curvature * t * t * (0.005 + spread * 0.008);
        vec3 sampleColor = sampleHdr(uv + offset);
        float weight = exp(-t * 1.0) * smoothstep(0.35, 1.45 + energy * 1.40, luma(sampleColor));
        accumulated += sampleColor * weight;
        weightSum += weight;
    }

    if (weightSum <= 0.0001) {
        return vec3(0.0);
    }
    return accumulated / weightSum;
}

vec3 sampleWaterReflectionField(
    vec2 uv,
    vec2 flow,
    vec2 trajectoryDir,
    float roughness,
    float energy
) {
    vec2 flowDir = safeNormalize2(flow * 0.76 + trajectoryDir * 0.42);
    vec2 sideDir = vec2(-flowDir.y, flowDir.x);
    vec2 mirrorUv = vec2(uv.x, 1.0 - uv.y);

    vec2 rippleOffsetA = flowDir * (0.010 + roughness * 0.018 + energy * 0.010);
    vec2 rippleOffsetB = sideDir * (0.005 + roughness * 0.020);
    vec2 rippleOffsetC = safeNormalize2(flowDir + sideDir * 0.65) * (0.004 + roughness * 0.012);

    vec3 sampleA = sampleHdr(mirrorUv + rippleOffsetA);
    vec3 sampleB = sampleHdr(mirrorUv - rippleOffsetA * 0.55);
    vec3 sampleC = sampleHdr(mirrorUv + rippleOffsetB);
    vec3 sampleD = sampleHdr(mirrorUv - rippleOffsetB * 0.75);
    vec3 sampleE = sampleHdr(mirrorUv + rippleOffsetC * 1.6);
    vec3 sampleF = sampleHdr(mirrorUv - rippleOffsetC * 1.2);
    vec3 sampleG = sampleHdr(mirrorUv + (rippleOffsetA + rippleOffsetB) * 0.65);

    return sampleA * 0.24
        + sampleB * 0.16
        + sampleC * 0.16
        + sampleD * 0.14
        + sampleE * 0.10
        + sampleF * 0.10
        + sampleG * 0.10;
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

float sampleSunShaftOcclusion(vec2 uv, vec2 sunScreenDir, float radiusPixels) {
    if (length(sunScreenDir) < 0.0001) {
        return 1.0;
    }

    vec2 dir = normalize(-sunScreenDir);
    vec2 pixel = vec2(1.0 / max(viewWidth, 1.0), 1.0 / max(viewHeight, 1.0));
    float transmittance = 0.0;
    for (int index = 1; index <= 5; ++index) {
        float travel = float(index) * radiusPixels;
        vec2 sampleUv = saturate2(uv + dir * pixel * travel);
        float sampleDepth = texture2D(depthtex0, sampleUv).r;
        transmittance += sampleDepth > 0.999 ? 1.0 : 0.0;
    }
    return transmittance / 5.0;
}

vec3 computeVolumetricFog(
    vec2 uv,
    vec3 hdr,
    vec3 viewPos,
    vec2 bakedLight,
    vec2 trajectoryDir,
    float energy,
    float spread,
    float shadowReceiver
) {
    float distance = length(viewPos);
    float fogAmount = 1.0 - exp(-distance * (0.014 + spread * 0.010));
    float sunHeight = saturate1(dot(normalize(sunPosition), normalize(upPosition)) * 0.5 + 0.5);
    vec3 skyFog = mix(vec3(0.18, 0.24, 0.34), vec3(0.46, 0.58, 0.72), sunHeight);
    vec3 torchFog = vec3(1.15, 0.72, 0.28);

    float sunShaft = sampleSunShaftOcclusion(uv, trajectoryDir, mix(12.0, 36.0, spread));
    float sunScatter = sunShaft * (0.14 + energy * 0.18) * bakedLight.y;
    float blockScatter = bakedLight.x * bakedLight.x * (0.08 + shadowReceiver * 0.14);

    vec3 fogColor = skyFog * sunScatter + torchFog * blockScatter;
    vec3 baseFog = mix(vec3(0.0), fogColor, fogAmount);
    vec3 localExtinction = hdr * mix(0.96, 0.78, fogAmount * 0.55);
    return mix(localExtinction, localExtinction + baseFog, saturate1(fogAmount));
}

void main() {
    vec2 uv = texCoord;
    vec3 hdr = texture2D(colortex0, uv).rgb;
    vec3 worldNormal = decodeNormal(texture2D(colortex1, uv).rgb);
    vec2 bakedLight = texture2D(colortex2, uv).rg;
    vec4 waterData = texture2D(colortex4, uv);
    vec4 assist = texture2D(npuAssist, saturate2(uv));
    float depth = texture2D(depthtex0, uv).r;

    vec3 viewPos = getViewPos(uv, depth);
    vec3 viewDir = safeNormalize3(-viewPos);
    vec3 viewNormal = getViewNormal(worldNormal);

    vec2 npuDirection = assist.rg * 2.0 - 1.0;
    float npuEnergy = assist.b;
    float npuSpread = assist.a;
    float curvature = (assist.r - assist.g) * 0.85;
    float distanceWeight = smoothstep(0.12, 0.98, depth);
    vec2 sunScreenDir = safeNormalize2(vec2(sunPosition.x, -sunPosition.y));
    vec2 trajectoryDir = safeNormalize2(mix(npuDirection, sunScreenDir, 0.58));
    float baseSceneLuma = luma(hdr);
    float shadowReceiver = smoothstep(0.20, 0.86, 1.0 - baseSceneLuma);
    vec3 groundTint = normalize(max(pow(max(hdr, vec3(0.0)), vec3(0.42)) + vec3(0.08), vec3(0.0001)));
    vec3 receiverTint = mix(vec3(1.0), groundTint * (1.05 + shadowReceiver * 0.35), 0.40 + shadowReceiver * 0.36);

    vec3 giTrace = traceGiPath(uv, trajectoryDir, curvature, npuSpread, npuEnergy);
    float giAmount = 0.05 + npuEnergy * 0.18 + shadowReceiver * 0.24;
    hdr += giTrace * receiverTint * giAmount;

    float exposure = mix(0.94, 1.42, npuEnergy) * mix(0.92, 1.08, distanceWeight);
    vec3 graded = hdr * exposure;
    vec3 palette = normalize(max(vec3(0.26 + assist.r, 0.30 + assist.b, 0.24 + assist.g), vec3(0.0001)));
    graded *= mix(vec3(0.95), vec3(0.82) + palette * 0.28, 0.18 + npuSpread * 0.20);

    float waterReflectivity = waterData.b;
    float waterRoughness = waterData.a;
    if (waterReflectivity > 0.05) {
        vec2 waterFlow = waterData.rg * 2.0 - 1.0;
        float sunHeight = saturate1(dot(normalize(sunPosition), normalize(upPosition)) * 0.5 + 0.5);
        vec3 waterNormal = safeNormalize3(vec3(
            waterFlow.x * (0.72 + npuSpread * 0.16) + trajectoryDir.x * 0.16,
            1.36 - waterRoughness * 0.40,
            waterFlow.y * (0.72 + npuSpread * 0.16) + trajectoryDir.y * 0.16
        ));
        vec3 waterReflectionDir = reflect(-viewDir, waterNormal);
        vec2 reflectionAxis = safeNormalize2(waterFlow * 0.72 + trajectoryDir * 0.52 + waterReflectionDir.xy * 0.20);
        float fresnel = 0.02 + 0.98 * pow(1.0 - saturate1(dot(waterNormal, viewDir)), 5.0);

        vec3 reflected = sampleWaterReflectionField(uv, waterFlow, reflectionAxis, waterRoughness, npuEnergy);
        vec3 sparkle = sampleWaterSparkle(uv, waterFlow, reflectionAxis, waterRoughness, npuEnergy);
        vec3 waterTrail = traceGiPath(uv, safeNormalize2(waterFlow + trajectoryDir * 0.42), curvature, 0.28 + npuSpread * 0.52, npuEnergy);

        vec3 skyTint = mix(vec3(0.24, 0.38, 0.55), vec3(0.72, 0.86, 1.05), sunHeight);
        float horizonMix = clamp(0.34 + waterRoughness * 0.28 + (1.0 - abs(reflectionAxis.y)) * 0.14, 0.0, 1.0);
        reflected = mix(reflected, skyTint, horizonMix);
        reflected += waterTrail * receiverTint * (0.10 + npuEnergy * 0.18 + shadowReceiver * 0.12);
        reflected += sparkle * skyTint * (0.08 + (1.0 - waterRoughness) * 0.18 + npuEnergy * 0.06);

        vec3 viewSunDir = safeNormalize3(normalize(sunPosition));
        float glint = pow(saturate1(dot(waterReflectionDir, safeNormalize3(-viewSunDir))), 18.0 + (1.0 - waterRoughness) * 22.0) * (0.06 + sunHeight * 0.18);
        float reflectionMix = clamp(waterReflectivity * fresnel * (0.58 + npuSpread * 0.12), 0.0, 0.68);
        hdr = mix(hdr, reflected, reflectionMix);
        hdr += skyTint * glint * (0.04 + waterReflectivity * 0.16);
        graded = hdr * exposure;
        graded *= mix(vec3(0.95), vec3(0.82) + palette * 0.28, 0.18 + npuSpread * 0.20);
    }

    graded = computeVolumetricFog(uv, graded, viewPos, bakedLight, trajectoryDir, npuEnergy, npuSpread, shadowReceiver);
    graded *= 0.35;

    vec3 toneMapped = acesTonemap(graded);
    float vignette = 1.0 - smoothstep(0.30, 1.45, dot(uv * 2.0 - 1.0, uv * 2.0 - 1.0));
    toneMapped *= mix(0.88, 1.0, vignette);

    gl_FragData[0] = vec4(toSrgb(toneMapped), 1.0);
}
