#version 450

layout(location = 0) in vec3 fragPosition;
layout(location = 1) in vec3 fragNormal;
layout(location = 2) in vec3 fragColor;
layout(location = 3) in vec2 fragUV;
layout(location = 0) out vec4 outColor;

struct PointLight { vec3 position; float intensity; vec3 color; float _pad; };
struct SpotLight  { vec3 position; float inner_angle; vec3 color; float outer_angle;
                    vec3 direction; float intensity; };

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

// <<< ЛАБ.3: set 1 = материал, три кортежа изображение-сэмплер
layout(set = 1, binding = 0) uniform sampler2D albedo_tex;
layout(set = 1, binding = 1) uniform sampler2D specular_tex;   // маска блеска
layout(set = 1, binding = 2) uniform sampler2D emissive_tex;   // маска свечения

layout(std140, set = 2, binding = 0) uniform ModelUniforms {
    mat4 model;
    vec3 albedo;    float shininess;
    vec3 specular;  float unlit;
    vec4 tint;                  // .a = подмес цвета вершин
    vec2 uv_scale;
    vec2 uv_offset;
    vec4 params;                // .x = emissive strength, .y = specular strength
} model_uniforms;

// <<< ЛАБ.3: push-константы ТОЛЬКО про эффект
layout(push_constant) uniform PushConstants {
    float time;
    uint  effect_mode;   // 0 off, 1 ripple, 2 bulge, 3 swirl
    float strength;
    float speed;
    vec2  center;
    float frequency;
    float radius;
} pc;

// Расходящаяся рябь по воде: радиальное смещение UV затухающей синусоидой
vec2 ripple(vec2 uv, float t) {
    vec2 d = uv - pc.center;
    float r = length(d);
    vec2 dir = (r > 1e-5) ? d / r : vec2(0.0);
    float wave = sin(r * pc.frequency - t * pc.speed) * exp(-r * 3.0);
    return uv + dir * wave * pc.strength;
}

// Маска из тиктока: локальное увеличение (bulge) вокруг точки
vec2 bulge(vec2 uv, vec2 c, float radius, float amount) {
    vec2 d = uv - c;
    float r = length(d);
    // в центре коэффициент < 1 -> сэмплируем ближе к центру -> картинка увеличивается
    float k = 1.0 - amount * (1.0 - smoothstep(0.0, radius, r));
    return c + d * k;
}

vec2 swirl(vec2 uv, float t) {
    vec2 d = uv - pc.center;
    float r = length(d);
    float a = pc.strength * 40.0 * exp(-r * 2.5) * sin(t * pc.speed * 0.4);
    float s = sin(a), c = cos(a);
    return pc.center + mat2(c, -s, s, c) * d;
}

// Блинн–Фонг. spec_color теперь = материал * ТЕКСТУРНАЯ МАСКА
vec3 blinnPhong(vec3 N, vec3 V, vec3 L, vec3 radiance, vec3 base_color, vec3 spec_color) {
    float ndotl = max(dot(N, L), 0.0);
    vec3 H = normalize(L + V);
    float spec = (ndotl > 0.0)
        ? pow(max(dot(N, H), 0.0), max(model_uniforms.shininess, 1.0))
        : 0.0;
    return (ndotl * base_color + spec * spec_color) * radiance;
}

void main() {
    // Визуализация источников: постоянный цвет, без освещения и текстур
    if (model_uniforms.unlit > 0.5) {
        outColor = vec4(model_uniforms.tint.rgb, 1.0);
        return;
    }

    vec2 uv = fragUV;
    float t = pc.time;

    vec3 albedo_sample;
    vec3 spec_sample;
    vec3 emis_sample;

    if (pc.effect_mode == 1u) {
        vec2 uvR = ripple(uv, t);
        vec2 uvG = ripple(uv, t + 0.12);
        vec2 uvB = ripple(uv, t + 0.24);
        albedo_sample = vec3(
            texture(albedo_tex, uvR).r,
            texture(albedo_tex, uvG).g,
            texture(albedo_tex, uvB).b);

        vec2 uvM = vec2(1.0) - uv * 1.7 + vec2(0.05 * sin(t * 0.7), 0.05 * cos(t * 0.9));
        vec3 caustic = texture(albedo_tex, ripple(uvM, t * 1.3)).rgb;
        float k = 0.25 + 0.15 * sin(t * pc.speed);
        albedo_sample = mix(albedo_sample, albedo_sample * caustic * 2.0, k);

        float phase = t * pc.speed + length(uv - pc.center) * pc.frequency * 0.3;
        vec3 mod_col = 0.78 + 0.32 * cos(vec3(0.0, 2.094, 4.188) + phase);
        albedo_sample *= mod_col;

        spec_sample = texture(specular_tex, uvG).rgb;
        emis_sample = texture(emissive_tex, uvR * 0.8 + vec2(0.0, t * 0.03)).rgb;

    } else if (pc.effect_mode == 2u) {
        //"ТИКТОК-МАСКА": увеличенные глаза и губы, пульсация от времени
        float pulse = 0.5 + 0.5 * sin(t * pc.speed);
        float amt = clamp(pc.strength * 12.0 * pulse, 0.0, 0.85);

        vec2 uvD = uv;
        uvD = bulge(uvD, pc.center + vec2(-0.12, -0.06), pc.radius,        amt);        // левый глаз
        uvD = bulge(uvD, pc.center + vec2( 0.12, -0.06), pc.radius,        amt);        // правый глаз
        uvD = bulge(uvD, pc.center + vec2( 0.00,  0.16), pc.radius * 1.35, amt * 0.8);  // губы

        albedo_sample = texture(albedo_tex, uvD).rgb;
        // вторая выборка того же изображения другими координатами -> мягкий "glow"
        vec3 halo = texture(albedo_tex, mix(uvD, pc.center, -0.04)).rgb;
        albedo_sample = mix(albedo_sample, halo, 0.22);
        albedo_sample *= 1.0 + 0.15 * sin(t * pc.speed * 2.0);

        spec_sample = texture(specular_tex, uvD).rgb;
        emis_sample = texture(emissive_tex, uvD).rgb;

    } else if (pc.effect_mode == 3u) {
        // ---- ВИХРЬ ----
        vec2 uvS = swirl(uv, t);
        vec3 a = texture(albedo_tex, uvS).rgb;
        vec3 b = texture(albedo_tex, swirl(uv * 1.3, t * 1.7)).rgb;
        albedo_sample = mix(a, b, 0.35 + 0.2 * sin(t * pc.speed));
        albedo_sample *= 0.85 + 0.25 * sin(t * pc.speed + uvS.x * 8.0);
        spec_sample = texture(specular_tex, uvS).rgb;
        emis_sample = texture(emissive_tex, uvS).rgb;

    } else {
        // ---- обычное сэмплирование ----
        albedo_sample = texture(albedo_tex,   uv).rgb;
        spec_sample   = texture(specular_tex, uv).rgb;
        emis_sample   = texture(emissive_tex, uv).rgb;
    }

    // Альбедо = текстура * цвет материала * tint (+ опциональный подмес цвета вершин)
    vec3 base_color = albedo_sample
                    * model_uniforms.albedo
                    * model_uniforms.tint.rgb
                    * mix(vec3(1.0), fragColor, model_uniforms.tint.a);

    // Specular-ТЕКСТУРА диктует, какая часть геометрии блестит
    vec3 spec_color = model_uniforms.specular * spec_sample * model_uniforms.params.y;

    vec3 N = normalize(fragNormal);
    vec3 V = normalize(scene_uniforms.camera_position - fragPosition);

    vec3 total = scene_uniforms.ambient_light * base_color;

    vec3 Ld = normalize(-scene_uniforms.directional_light_direction);
    total += blinnPhong(N, V, Ld, scene_uniforms.directional_light_color, base_color, spec_color);

    for (uint i = 0u; i < scene_uniforms.point_light_count; ++i) {
        vec3 to_light = point_lights[i].position - fragPosition;
        float d2 = max(dot(to_light, to_light), 1e-4);
        vec3 L = to_light * inversesqrt(d2);
        float att = point_lights[i].intensity / d2;
        total += blinnPhong(N, V, L, point_lights[i].color * att, base_color, spec_color);
    }

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
        total += blinnPhong(N, V, L, spot_lights[i].color * (att * cone), base_color, spec_color);
    }

    total = total / (1.0 + total);

    total += emis_sample * model_uniforms.params.x;

    outColor = vec4(total, 1.0);
}