const int shadowMapResolution = 2048;
const float shadowDistance = 160.0;
const float shadowDistanceRenderMul = 1.0;
const float sunPathRotation = -35.0;

vec3 distortShadowClipPos(vec3 shadowClipPos) {
    float distortionFactor = length(shadowClipPos.xy);
    distortionFactor += 0.1;
    shadowClipPos.xy /= distortionFactor;
    shadowClipPos.z *= 0.5;
    return shadowClipPos;
}
