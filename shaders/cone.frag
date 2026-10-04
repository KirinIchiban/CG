#version 450

layout(location = 0) in vec3 fragColor;
layout(location = 0) out vec4 outColor;

layout(push_constant) uniform PushConstants {
    vec4 tintColor; 
} pc;

void main() {
    outColor = vec4(fragColor * pc.tintColor.rgb, 1.0f);
}