#version 330

uniform sampler2D InSampler;
uniform sampler2D NpuAssistSampler;

in vec2 texCoord;

layout(std140) uniform AssistConfig {
    float WarpAmount;
    float TintMix;
    float TintLift;
};

out vec4 fragColor;

void main() {
    vec4 assist = texture(NpuAssistSampler, texCoord);
    vec2 warpedUv = clamp(texCoord + ((assist.rg * 2.0) - 1.0) * WarpAmount, 0.0, 1.0);
    vec4 source = texture(InSampler, warpedUv);

    float tintFactor = 0.82 + assist.b * TintLift;
    vec3 tinted = source.rgb * tintFactor;
    vec3 outColor = mix(source.rgb, tinted, assist.a * TintMix);

    fragColor = vec4(clamp(outColor, 0.0, 1.0), 1.0);
}
