#version 150 compatibility

uniform sampler2D colortex0;
uniform sampler2D colortex4;
uniform sampler2D depthtex0;
uniform sampler2D npuAssist;
uniform vec3 sunPosition;
uniform vec3 upPosition;

in vec2 texCoord;

// final is the best place to consume NPU-generated exposure / grading policy.
// Keep direct screen fetches and tone mapping on the GPU, but let the NPU steer
// glare direction, exposure, palette bias, or vignette strength through npuAssist.
float saturate1(float value) {
    return clamp(value, 0.0, 1.0);
}

vec2 saturate2(vec2 value) {
    return clamp(value, vec2(0.0), vec2(1.0));
}

vec3 saturate3(vec3 value) {
    return clamp(value, vec3(0.0), vec3(1.0));
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

void main() {
    vec2 uv = texCoord;
    vec3 hdr = texture2D(colortex0, uv).rgb;
    vec4 waterData = texture2D(colortex4, uv);
    vec4 assist = texture2D(npuAssist, saturate2(uv));
    float depth = texture2D(depthtex0, uv).r;

    // rg = flow direction, b = exposure / glow energy, a = blend weight.
    // If future contributors want more NPU usage, expand these channels into
    // low-resolution grading coefficients instead of trying to offload the full final pass.
    vec2 flow = assist.rg * 2.0 - 1.0;
    float energy = assist.b;
    float mask = assist.a;
    float distanceWeight = smoothstep(0.12, 0.98, depth);

    // Neighbor taps stay on the GPU because they read the live HDR buffer.
    vec3 glareSampleA = texture2D(colortex0, saturate2(uv + flow * (0.003 + energy * 0.006))).rgb;
    vec3 glareSampleB = texture2D(colortex0, saturate2(uv - flow * (0.002 + energy * 0.004))).rgb;
    float glowWeight = smoothstep(0.85, 2.4, luma(hdr)) * (0.08 + mask * 0.22);
    hdr += (glareSampleA + glareSampleB) * glowWeight;

    // Good NPU candidates here: exposure, palette mix, glare axis, vignette, bloom thresholds.
    // Bad NPU candidates here: direct HDR buffer reads or full-resolution tone mapping.
    float exposure = mix(0.95, 1.55, energy) * mix(0.92, 1.08, distanceWeight);
    vec3 graded = hdr * exposure;
    vec3 palette = normalize(max(vec3(0.28 + assist.r, 0.32 + assist.b, 0.24 + assist.g), vec3(0.0001)));
    graded *= mix(vec3(0.96), vec3(0.82) + palette * 0.26, 0.16 + mask * 0.18);

    float waterMask = waterData.a;
    if (waterMask > 0.01) {
        // Water reflection is staged as an NPU-guided policy:
        // the water pass writes flow + reflectivity into colortex4, then final samples
        // the live HDR scene using that policy to synthesize a lightweight reflection.
        vec2 waterFlow = waterData.rg * 2.0 - 1.0;
        float reflectivity = waterData.b;
        float sunHeight = saturate1(dot(normalize(sunPosition), normalize(upPosition)) * 0.5 + 0.5);
        vec2 reflectUv = vec2(
            uv.x + (waterFlow.x + flow.x) * (0.015 + reflectivity * 0.02),
            1.0 - uv.y + (waterFlow.y - flow.y) * (0.01 + reflectivity * 0.015)
        );
        vec3 reflected = texture2D(colortex0, saturate2(reflectUv)).rgb;
        vec3 skyTint = mix(vec3(0.22, 0.34, 0.50), vec3(0.72, 0.86, 1.05), sunHeight);
        float glint = pow(saturate1(1.0 - abs(waterFlow.y)), 6.0) * (0.12 + sunHeight * 0.25);
        float reflectionMix = clamp(reflectivity * (0.62 + mask * 0.24), 0.0, 0.92);
        hdr = mix(hdr, reflected * skyTint, reflectionMix);
        hdr += skyTint * glint * (0.35 + reflectivity * 0.55);
        graded = hdr * exposure;
        graded *= mix(vec3(0.96), vec3(0.82) + palette * 0.26, 0.16 + mask * 0.18);
    }

    vec3 toneMapped = acesTonemap(graded);
    float vignette = 1.0 - smoothstep(0.30, 1.45, dot(uv * 2.0 - 1.0, uv * 2.0 - 1.0));
    toneMapped *= mix(0.88, 1.0, vignette);

    gl_FragData[0] = vec4(toSrgb(toneMapped), 1.0);
}
