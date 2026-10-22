#version 450

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec3 inColor;

layout(location = 0) out vec3 fragPosition;
layout(location = 1) out vec3 fragNormal;
layout(location = 2) out vec3 fragColor;


layout(std140, set = 0, binding = 0) uniform SceneUniforms {
    mat4 view_projection;
    vec3 camera_position;             float _pad0;
    vec3 directional_light_direction; float _pad1;
    vec3 directional_light_color;     float _pad2;
    vec3 ambient_light;               float _pad3;
    uint point_light_count;
    uint spot_light_count;
    uint _pad4;
    uint _pad5;
} scene_uniforms;

layout(std140, set = 1, binding = 0) uniform ModelUniforms {
    mat4 model;
    vec3 albedo;    float _pad0;
    vec3 specular;  float _pad1;
    float shininess;
} model_uniforms;

void main() {
    vec4 world_pos = model_uniforms.model * vec4(inPosition, 1.0);
    fragPosition = world_pos.xyz;

    fragNormal = transpose(inverse(mat3(model_uniforms.model))) * inNormal;

    fragColor = inColor;
    gl_Position = scene_uniforms.view_projection * world_pos;
}
