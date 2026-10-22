#version 450

layout(location = 0) in vec3 fragPosition;
layout(location = 1) in vec3 fragNormal;
layout(location = 2) in vec3 fragColor;
layout(location = 0) out vec4 outColor;


struct PointLight {
    vec3 position;  float intensity;
    vec3 color;     float _pad;
};

struct SpotLight {
    vec3 position;  float inner_angle;
    vec3 color;     float outer_angle;
    vec3 direction; float intensity;
};

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

layout(std430, set = 0, binding = 1) readonly buffer PointLightsBuffer { PointLight point_lights[]; };
layout(std430, set = 0, binding = 2) readonly buffer SpotLightsBuffer  { SpotLight  spot_lights[];  };

layout(std140, set = 1, binding = 0) uniform ModelUniforms {
    mat4 model;
    vec3 albedo;    float _pad0;
    vec3 specular;  float _pad1;
    float shininess;
} model_uniforms;

layout(push_constant) uniform PushConstants {
    vec4 tintColor;
    float is_unlit;
    float _pad[3];
} pc;

// Блинн–Фонг для одного источника.
// L — единичный вектор от точки к источнику, radiance — цвет * интенсивность * затухание.
// Диффузная часть умножается на albedo (цвет материала * радужный цвет вершины),
// бликовая — на specular материала (цвет блика НЕ окрашивается albedo).
vec3 blinnPhong(vec3 N, vec3 V, vec3 L, vec3 radiance, vec3 base_color) {
    float ndotl = max(dot(N, L), 0.0);
    vec3 H = normalize(L + V);
    float spec = (ndotl > 0.0)
        ? pow(max(dot(N, H), 0.0), max(model_uniforms.shininess, 1.0))
        : 0.0;
    return (ndotl * base_color + spec * model_uniforms.specular) * radiance;
}

void main() {
    // Визуализация источников: постоянный цвет, без освещения
    if (pc.is_unlit > 0.5) {
        outColor = pc.tintColor;
        return;
    }

    vec3 N = normalize(fragNormal);
    vec3 V = normalize(scene_uniforms.camera_position - fragPosition);

    // Радужный цвет вершин * цвет материала
    vec3 base_color = fragColor * model_uniforms.albedo;

    // Ambient
    vec3 total = scene_uniforms.ambient_light * base_color;

    // Направленный источник (direction — куда светит, поэтому L = -direction)
    vec3 Ld = normalize(-scene_uniforms.directional_light_direction);
    total += blinnPhong(N, V, Ld, scene_uniforms.directional_light_color, base_color);

    // Точечные источники: затухание по закону обратных квадратов
    for (uint i = 0u; i < scene_uniforms.point_light_count; ++i) {
        vec3 to_light = point_lights[i].position - fragPosition;
        float d2 = max(dot(to_light, to_light), 1e-4);
        vec3 L = to_light * inversesqrt(d2);
        float att = point_lights[i].intensity / d2;
        total += blinnPhong(N, V, L, point_lights[i].color * att, base_color);
    }

    // Прожекторы: обратные квадраты + мягкий край между inner и outer углами
    for (uint i = 0u; i < scene_uniforms.spot_light_count; ++i) {
        vec3 to_light = spot_lights[i].position - fragPosition;
        float d2 = max(dot(to_light, to_light), 1e-4);
        vec3 L = to_light * inversesqrt(d2);

        float cos_theta = dot(L, -normalize(spot_lights[i].direction));
        float cos_outer = cos(spot_lights[i].outer_angle);
        float cos_inner = cos(spot_lights[i].inner_angle);
        float eps = max(cos_inner - cos_outer, 1e-4);
        float cone = clamp((cos_theta - cos_outer) / eps, 0.0, 1.0);

        float att = spot_lights[i].intensity / d2;
        total += blinnPhong(N, V, L, spot_lights[i].color * (att * cone), base_color);
    }

    total = total / (1.0 + total);
    outColor = vec4(total, 1.0);
}
