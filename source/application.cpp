#include "geometry.hpp"
#include "graphics_internal.hpp"
#include "vk_buffer.hpp"
#include "vk_texture.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#define GLM_ENABLE_EXPERIMENTAL
#include <vulkan/vulkan.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/constants.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <imgui.h>

#ifndef TEXTURE_DIR
#define TEXTURE_DIR "assets/textures/"
#endif

namespace application {
namespace {

namespace fs = std::filesystem;

// GPU-структуры
struct alignas(16) SceneUniforms {
    glm::mat4 view_projection;       // 0
    glm::vec3 camera_position;       // 64
    float     _pad0;
    glm::vec3 directional_light_dir; // 80
    float     _pad1;
    glm::vec3 directional_light_col; // 96
    float     _pad2;
    glm::vec3 ambient_light;         // 112
    float     _pad3;
    uint32_t  point_light_count;     // 128
    uint32_t  spot_light_count;      // 132
    uint32_t  _pad4[2];
};
static_assert(sizeof(SceneUniforms) == 144, "SceneUniforms must be 144 bytes");

struct alignas(16) ModelUniforms {
    glm::mat4 model;        // 0
    glm::vec3 albedo;       // 64
    float     shininess;    // 76
    glm::vec3 specular;     // 80
    float     unlit;        // 92
    glm::vec4 tint;         // 96
    glm::vec2 uv_scale;     // 112
    glm::vec2 uv_offset;    // 120
    glm::vec4 params;       // 128  .x = emissive strength, .y = specular strength
};
static_assert(offsetof(ModelUniforms, albedo)    ==  64, "std140 mismatch");
static_assert(offsetof(ModelUniforms, shininess) ==  76, "std140 mismatch");
static_assert(offsetof(ModelUniforms, specular)  ==  80, "std140 mismatch");
static_assert(offsetof(ModelUniforms, unlit)     ==  92, "std140 mismatch");
static_assert(offsetof(ModelUniforms, tint)      ==  96, "std140 mismatch");
static_assert(offsetof(ModelUniforms, uv_scale)  == 112, "std140 mismatch");
static_assert(offsetof(ModelUniforms, uv_offset) == 120, "std140 mismatch");
static_assert(offsetof(ModelUniforms, params)    == 128, "std140 mismatch");
static_assert(sizeof(ModelUniforms) == 144, "ModelUniforms must be 144 bytes");

struct alignas(16) PointLight {
    glm::vec3 position; float intensity;
    glm::vec3 color;    float _pad;
};
static_assert(sizeof(PointLight) == 32, "std430 mismatch");

struct alignas(16) SpotLight {
    glm::vec3 position;  float inner_angle;
    glm::vec3 color;     float outer_angle;
    glm::vec3 direction; float intensity;
};
static_assert(sizeof(SpotLight) == 48, "std430 mismatch");

struct PushConstants {
    float    time;         // 0
    uint32_t effect_mode;  // 4   0=off 1=ripple 2=bulge 3=swirl
    float    strength;     // 8
    float    speed;        // 12
    glm::vec2 center;      // 16
    float    frequency;    // 24
    float    radius;       // 28
};
static_assert(sizeof(PushConstants) == 32, "PushConstants must be 32 bytes");


struct MeshData {
    std::vector<Vertex>   vertices;
    std::vector<uint32_t> indices;
};

struct Mesh {
    vk_buffer::Buffer vb{}, ib{};
    uint32_t index_count = 0;
};

struct Material {
    std::string        name;
    vk_texture::Texture albedo{};
    int                spec_index     = 0;
    int                emissive_index = 0;
    VkDescriptorSet    set            = VK_NULL_HANDLE;
    bool               dirty          = true;
};

struct NamedTexture {
    std::string        name;
    vk_texture::Texture tex{};
};

struct SceneObject {
    glm::vec3 position{0.0f, 1.0f, 0.0f};
    glm::vec3 rotation{0.0f};
    glm::vec3 scale{1.0f};
    glm::vec3 tint{1.0f};
    float     vertex_color_mix = 0.0f;
    bool      trajectory = true;
    bool      playing    = true;
    float     phase = 0.0f, speed = 0.8f, radius = 2.0f, bob_amplitude = 0.4f;

    int       material_index = 0;        
    glm::vec2 uv_scale{1.0f, 1.0f};      
    glm::vec2 uv_offset{0.0f};           
    float     emissive_strength = 1.0f;  
    float     specular_strength = 1.0f;  
    int       effect_mode = 0;          
};

struct StaticObject {
    Mesh      mesh{};
    glm::vec3 position{0.0f};
    glm::vec3 rotation{0.0f};
    glm::vec3 scale{1.0f};
    glm::vec3 tint{1.0f};
    glm::vec2 uv_scale{1.0f};
    glm::vec2 uv_offset{0.0f};
    float     shininess = 16.0f;
    float     emissive_strength = 0.0f;
    float     specular_strength = 0.3f;
    int       material_index = 0;
    int       effect_mode = 0;
    bool      visible = true;
    const char* label = "Object";
};

struct Camera {
    float pos[3]{0.0f, 3.0f, 6.0f};
    float yaw = -90.0f, pitch = -15.0f, fov = 45.0f;
};

struct EffectParams {
    float strength  = 0.035f;
    float speed     = 3.0f;
    float frequency = 45.0f;
    float radius    = 0.13f;
    glm::vec2 center{0.5f, 0.5f};
};

// Константы и глобальные данные
constexpr uint32_t kObjectCount = 2;
constexpr size_t   kMaxLights   = 20;
constexpr uint32_t kMaxModels   = 128;      // ёмкость динамического UBO
constexpr float    kConeRadius  = 1.0f;
constexpr float    kConeHeight  = 2.0f;
constexpr float    kSpotVizLength = 2.0f;

std::array<SceneObject, kObjectCount> objects;
Camera        camera;
EffectParams  effect;
StaticObject  floor_obj, wall_obj;

glm::vec3 directional_light_dir = glm::normalize(glm::vec3(-0.5f, -1.0f, -0.5f));
glm::vec3 directional_light_col = glm::vec3(1.0f);
glm::vec3 ambient_light         = glm::vec3(0.15f);
float     light_master          = 1.0f;   // 0 = темнота, видно только emissive
std::vector<PointLight> point_lights;
std::vector<SpotLight>  spot_lights;

glm::vec3 material_albedo   = glm::vec3(1.0f);
glm::vec3 material_specular = glm::vec3(1.0f);
float     material_shininess = 32.0f;

Mesh cone_mesh, viz_cube_mesh, viz_cone_mesh;

vk_buffer::Buffer scene_uniform_buffer;
vk_buffer::Buffer point_lights_buffer;
vk_buffer::Buffer spot_lights_buffer;
vk_buffer::Buffer models_buffer;            // ОДИН динамический UBO
VkDeviceSize      uniforms_sizeof = 0;    

VkDescriptorSet       scene_descriptor_set = VK_NULL_HANDLE;
VkDescriptorSet       model_descriptor_set = VK_NULL_HANDLE;   // один на всех
VkDescriptorSetLayout scene_layout    = VK_NULL_HANDLE;
VkDescriptorSetLayout material_layout = VK_NULL_HANDLE;
VkDescriptorSetLayout model_layout    = VK_NULL_HANDLE;
VkDescriptorPool      descriptor_pool = VK_NULL_HANDLE;
VkPipelineLayout      pipeline_layout = VK_NULL_HANDLE;
VkPipeline            pipeline = VK_NULL_HANDLE, line_pipeline = VK_NULL_HANDLE;

std::vector<Material>     materials;          
std::vector<NamedTexture> spec_textures;
std::vector<NamedTexture> emissive_textures;
vk_texture::Texture black_tex{};              // служебная 1x1

int    projection_type = 0;
double previous_time = -1.0;
float  current_time  = 0.0f;
bool   show_light_visualization = true;

// Утилиты
VkShaderModule loadShaderModule(const char* filename) {
    const std::string path = std::string(SHADER_DIR) + filename;
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) { std::cerr << "Shader not found: " << path << "\n"; return VK_NULL_HANDLE; }
    const size_t size = size_t(file.tellg());
    std::vector<char> code(size);
    file.seekg(0);
    file.read(code.data(), std::streamsize(size));
    file.close();
    const VkShaderModuleCreateInfo ci = {
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = code.size(),
        .pCode = reinterpret_cast<const uint32_t*>(code.data()),
    };
    VkShaderModule m = VK_NULL_HANDLE;
    vkCreateShaderModule(graphics::internal::context.device, &ci, nullptr, &m);
    return m;
}

glm::vec3 hsv2rgb(float h, float s, float v) {
    h -= std::floor(h);
    const glm::vec3 k(
        std::clamp(std::abs(h * 6.0f - 3.0f) - 1.0f, 0.0f, 1.0f),
        std::clamp(2.0f - std::abs(h * 6.0f - 2.0f), 0.0f, 1.0f),
        std::clamp(2.0f - std::abs(h * 6.0f - 4.0f), 0.0f, 1.0f));
    return v * glm::mix(glm::vec3(1.0f), k, s);
}

// Шум Перлина для emissive-текстуры
struct Perlin {
    std::array<int, 512> p{};
    explicit Perlin(uint32_t seed = 1337) {
        std::array<int, 256> perm{};
        std::iota(perm.begin(), perm.end(), 0);
        uint32_t s = seed;
        for (int i = 255; i > 0; --i) {
            s = s * 1664525u + 1013904223u;
            std::swap(perm[i], perm[s % (i + 1)]);
        }
        for (int i = 0; i < 512; ++i) p[i] = perm[i & 255];
    }
    static float fade(float t) { return t * t * t * (t * (t * 6 - 15) + 10); }
    static float grad(int h, float x, float y) {
        switch (h & 7) {
            case 0: return  x + y; case 1: return -x + y;
            case 2: return  x - y; case 3: return -x - y;
            case 4: return  x;     case 5: return -x;
            case 6: return  y;     default: return -y;
        }
    }
    float noise(float x, float y) const {
        const int X = int(std::floor(x)) & 255, Y = int(std::floor(y)) & 255;
        x -= std::floor(x); y -= std::floor(y);
        const float u = fade(x), v = fade(y);
        const int A = p[X] + Y, B = p[X + 1] + Y;
        return glm::mix(glm::mix(grad(p[A],     x,     y), grad(p[B],     x - 1, y),     u),
                        glm::mix(grad(p[A + 1], x, y - 1), grad(p[B + 1], x - 1, y - 1), u), v);
    }
    float fbm(float x, float y, int octaves = 5) const {
        float sum = 0.0f, amp = 0.5f, freq = 1.0f;
        for (int i = 0; i < octaves; ++i) { sum += amp * noise(x * freq, y * freq); freq *= 2.0f; amp *= 0.5f; }
        return sum;
    }
};

// Генерация геометрии
MeshData generateCone(float radius, float height, uint32_t segments) {
    MeshData mesh;
    if (segments < 3) return mesh;

    const float half_h = height * 0.5f;
    const float two_pi = 2.0f * glm::pi<float>();
    const float dtheta = two_pi / float(segments);
    const float slope  = std::sqrt(height * height + radius * radius);
    const glm::vec3 apex(0.0f, half_h, 0.0f);
    const glm::vec3 white(1.0f);

    auto side_normal = [&](float th) {
        return glm::vec3(height * std::cos(th) / slope, radius / slope, height * std::sin(th) / slope);
    };
    auto rim_pos = [&](float th) {
        return glm::vec3(radius * std::cos(th), -half_h, radius * std::sin(th));
    };

    const uint32_t rim_begin = 0;
    for (uint32_t i = 0; i <= segments; ++i) {
        const float th = dtheta * float(i);
        const float u  = float(i) / float(segments);
        mesh.vertices.push_back({ rim_pos(th), side_normal(th), white, {u, 1.0f} });
    }
    const uint32_t apex_begin = rim_begin + segments + 1;
    for (uint32_t i = 0; i < segments; ++i) {
        const float th_mid = dtheta * (float(i) + 0.5f);
        mesh.vertices.push_back({ apex, side_normal(th_mid), white,
                                  { (float(i) + 0.5f) / float(segments), 0.0f } });
    }
    const uint32_t base_center = apex_begin + segments;
    const uint32_t base_ring   = base_center + 1;
    mesh.vertices.push_back({ {0.0f, -half_h, 0.0f}, {0,-1,0}, white, {0.5f, 0.5f} });
    for (uint32_t i = 0; i <= segments; ++i) {
        const float th = dtheta * float(i);
        mesh.vertices.push_back({ rim_pos(th), {0,-1,0}, white,
                                  { 0.5f + 0.5f * std::cos(th), 0.5f + 0.5f * std::sin(th) } });
    }

    for (uint32_t i = 0; i < segments; ++i) {
        mesh.indices.push_back(apex_begin + i);
        mesh.indices.push_back(rim_begin + i + 1);
        mesh.indices.push_back(rim_begin + i);

        mesh.indices.push_back(base_center);
        mesh.indices.push_back(base_ring + i);
        mesh.indices.push_back(base_ring + i + 1);
    }
    return mesh;
}

MeshData generateFloor(float size) {
    MeshData mesh;
    const float h = size * 0.5f;
    const glm::vec3 n(0, 1, 0), c(1.0f);
    mesh.vertices = {
        { {-h,0,-h}, n, c, {0,0} },
        { { h,0,-h}, n, c, {1,0} },
        { { h,0, h}, n, c, {1,1} },
        { {-h,0, h}, n, c, {0,1} },
    };
    mesh.indices = {0,1,2, 0,2,3};
    return mesh;
}

MeshData generateWall(float w, float hgt) {
    MeshData mesh;
    const float hw = w * 0.5f, hh = hgt * 0.5f;
    const glm::vec3 n(0, 0, 1), c(1.0f);
    mesh.vertices = {
        { {-hw, hh, 0}, n, c, {0,0} },
        { { hw, hh, 0}, n, c, {1,0} },
        { { hw,-hh, 0}, n, c, {1,1} },
        { {-hw,-hh, 0}, n, c, {0,1} },
    };
    mesh.indices = {0,1,2, 0,2,3};
    return mesh;
}

MeshData generateCube(float size) {
    MeshData mesh;
    const float h = size * 0.5f;
    const glm::vec3 c(1.0f);
    const glm::vec2 uv[4] = { {0,0}, {1,0}, {1,1}, {0,1} };
    auto face = [&](glm::vec3 a, glm::vec3 b, glm::vec3 d, glm::vec3 e, glm::vec3 n) {
        const uint32_t base = uint32_t(mesh.vertices.size());
        mesh.vertices.push_back({a, n, c, uv[0]});
        mesh.vertices.push_back({b, n, c, uv[1]});
        mesh.vertices.push_back({d, n, c, uv[2]});
        mesh.vertices.push_back({e, n, c, uv[3]});
        mesh.indices.insert(mesh.indices.end(), {base, base+1, base+2, base, base+2, base+3});
    };
    face({-h,-h,-h},{ h,-h,-h},{ h, h,-h},{-h, h,-h},{0,0,-1});
    face({-h,-h, h},{ h,-h, h},{ h, h, h},{-h, h, h},{0,0, 1});
    face({-h,-h,-h},{ h,-h,-h},{ h,-h, h},{-h,-h, h},{0,-1,0});
    face({-h, h,-h},{ h, h,-h},{ h, h, h},{-h, h, h},{0, 1,0});
    face({-h,-h,-h},{-h,-h, h},{-h, h, h},{-h, h,-h},{-1,0,0});
    face({ h,-h,-h},{ h,-h, h},{ h, h, h},{ h, h,-h},{ 1,0,0});
    return mesh;
}

MeshData generateLightConeWire(uint32_t segments) {
    MeshData mesh;
    const glm::vec3 white(1.0f), up(0, 1, 0);
    mesh.vertices.push_back({ {0,0,0}, up, white, {0.5f, 0.0f} });
    for (uint32_t i = 0; i < segments; ++i) {
        const float th = 2.0f * glm::pi<float>() * float(i) / float(segments);
        mesh.vertices.push_back({ {std::cos(th), -1.0f, std::sin(th)}, up, white,
                                  { float(i)/segments, 1.0f } });
    }
    const uint32_t axis_end = segments + 1;
    mesh.vertices.push_back({ {0,-1,0}, up, white, {0.5f, 1.0f} });
    for (uint32_t i = 0; i < segments; ++i) {
        mesh.indices.push_back(1 + i);
        mesh.indices.push_back(1 + (i + 1) % segments);
    }
    const uint32_t step = std::max(1u, segments / 8u);
    for (uint32_t i = 0; i < segments; i += step) {
        mesh.indices.push_back(0);
        mesh.indices.push_back(1 + i);
    }
    mesh.indices.push_back(0);
    mesh.indices.push_back(axis_end);
    return mesh;
}

//вершины и индексы попадают в VRAM через staging
bool uploadMesh(const MeshData& data, Mesh& out) {
    if (data.vertices.empty() || data.indices.empty()) return false;
    const VkDeviceSize v_size = sizeof(Vertex)   * data.vertices.size();
    const VkDeviceSize i_size = sizeof(uint32_t) * data.indices.size();
    if (!vk_buffer::upload_device_local(data.vertices.data(), v_size,
                                        VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, out.vb)) return false;
    if (!vk_buffer::upload_device_local(data.indices.data(), i_size,
                                        VK_BUFFER_USAGE_INDEX_BUFFER_BIT, out.ib)) return false;
    out.index_count = static_cast<uint32_t>(data.indices.size());
    return true;
}

//процедурные текстуры
std::vector<uint8_t> makeCheckerboard(uint32_t size = 512, uint32_t cells = 8) {
    std::vector<uint8_t> px(size * size * 4);
    const uint32_t cs = size / cells;
    for (uint32_t y = 0; y < size; ++y)
        for (uint32_t x = 0; x < size; ++x) {
            const bool dark = ((x / cs) + (y / cs)) % 2 == 0;
            uint8_t* p = &px[(y * size + x) * 4];
            const uint8_t v = dark ? 40 : 220;
            p[0] = v; p[1] = v; p[2] = dark ? 60 : 200; p[3] = 255;
        }
    return px;
}

std::vector<uint8_t> makeUVGrid(uint32_t size = 512) {
    std::vector<uint8_t> px(size * size * 4);
    for (uint32_t y = 0; y < size; ++y)
        for (uint32_t x = 0; x < size; ++x) {
            uint8_t* p = &px[(y * size + x) * 4];
            const bool line = (x % 64 < 2) || (y % 64 < 2);
            p[0] = line ? 255 : uint8_t(255.0f * x / size);
            p[1] = line ? 255 : uint8_t(255.0f * y / size);
            p[2] = line ? 255 : 128;
            p[3] = 255;
        }
    return px;
}

// Emissive из шума Перлина: редкие яркие «руны», остальное почти чёрное
std::vector<uint8_t> makePerlinEmissive(uint32_t size = 512) {
    Perlin perlin(20260511);
    std::vector<uint8_t> px(size * size * 4);
    for (uint32_t y = 0; y < size; ++y)
        for (uint32_t x = 0; x < size; ++x) {
            const float fx = float(x) / size * 6.0f;
            const float fy = float(y) / size * 6.0f;
            float n = perlin.fbm(fx, fy, 5) * 0.5f + 0.5f;
            n = std::clamp(n, 0.0f, 1.0f);
            const float glow = std::pow(n, 4.0f) * 2.2f;
            const float g = std::clamp(glow, 0.0f, 1.0f);
            uint8_t* p = &px[(y * size + x) * 4];
            p[0] = uint8_t(255 * std::clamp(g * 0.35f, 0.0f, 1.0f));
            p[1] = uint8_t(255 * std::clamp(g * 0.85f, 0.0f, 1.0f));
            p[2] = uint8_t(255 * g);
            p[3] = 255;
        }
    return px;
}

// Specular-маска из яркости альбедо
std::vector<uint8_t> makeSpecularFromLuminance(const std::string& path, uint32_t& w, uint32_t& h) {
    // генерируем маску процедурно по UV (полосы + пятна)
    (void)path;
    w = h = 256;
    Perlin perlin(777);
    std::vector<uint8_t> px(w * h * 4);
    for (uint32_t y = 0; y < h; ++y)
        for (uint32_t x = 0; x < w; ++x) {
            float n = perlin.fbm(float(x) / w * 8.0f, float(y) / h * 8.0f, 4) * 0.5f + 0.5f;
            const float m = std::clamp(std::pow(std::clamp(n, 0.0f, 1.0f), 2.0f) * 1.6f, 0.0f, 1.0f);
            uint8_t* p = &px[(y * w + x) * 4];
            p[0] = p[1] = p[2] = uint8_t(255 * m);
            p[3] = 255;
        }
    return px;
}

bool isImageFile(const fs::path& p) {
    static const std::vector<std::string> ext = {".png",".jpg",".jpeg",".bmp",".tga",".gif"};
    std::string e = p.extension().string();
    std::transform(e.begin(), e.end(), e.begin(), ::tolower);
    return std::find(ext.begin(), ext.end(), e) != ext.end();
}

bool createTextures() {
    //служебная чёрная текстура 1x1
    const uint8_t black[4] = {0, 0, 0, 255};
    vk_texture::CreateInfo flat;
    flat.srgb = false; flat.generate_mips = false;
    flat.address = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    if (!vk_texture::create_from_pixels(black, 1, 1, black_tex, flat)) return false;

    spec_textures.push_back({"Black (no shine)", {}});
    emissive_textures.push_back({"Black (no glow)", {}});

    //Perlin emissive
    {
        auto px = makePerlinEmissive(512);
        NamedTexture nt{"Perlin Glow", {}};
        vk_texture::CreateInfo ci; ci.srgb = true;
        if (!vk_texture::create_from_pixels(px.data(), 512, 512, nt.tex, ci)) return false;
        emissive_textures.push_back(std::move(nt));           // [1]
    }
    //процедурная specular-маска
    {
        uint32_t w, h;
        auto px = makeSpecularFromLuminance("", w, h);
        NamedTexture nt{"Perlin Specular", {}};
        vk_texture::CreateInfo ci; ci.srgb = false;           // маска — не цвет
        if (!vk_texture::create_from_pixels(px.data(), w, h, nt.tex, ci)) return false;
        spec_textures.push_back(std::move(nt));               // [1]
    }

    //сканируем папку
    std::vector<fs::path> albedo_files, spec_files, emis_files;
    std::error_code ec;
    if (fs::exists(TEXTURE_DIR, ec)) {
        for (const auto& entry : fs::directory_iterator(TEXTURE_DIR, ec)) {
            if (!entry.is_regular_file() || !isImageFile(entry.path())) continue;
            const std::string stem = entry.path().stem().string();
            std::string low = stem;
            std::transform(low.begin(), low.end(), low.begin(), ::tolower);
            if (low.size() >= 5 && low.substr(low.size() - 5) == "_spec")      spec_files.push_back(entry.path());
            else if (low.size() >= 5 && low.substr(low.size() - 5) == "_emis") emis_files.push_back(entry.path());
            else                                                              albedo_files.push_back(entry.path());
        }
    }
    std::sort(albedo_files.begin(), albedo_files.end());

    for (const auto& p : spec_files) {
        NamedTexture nt{p.stem().string(), {}};
        vk_texture::CreateInfo ci; ci.srgb = false;
        if (vk_texture::load_from_file(p.string(), nt.tex, ci)) spec_textures.push_back(std::move(nt));
    }
    for (const auto& p : emis_files) {
        NamedTexture nt{p.stem().string(), {}};
        vk_texture::CreateInfo ci; ci.srgb = true;
        if (vk_texture::load_from_file(p.string(), nt.tex, ci)) emissive_textures.push_back(std::move(nt));
    }

    //альбедо из файлов
    for (const auto& p : albedo_files) {
        Material m;
        m.name = p.stem().string();
        vk_texture::CreateInfo ci; ci.srgb = true;
        if (!vk_texture::load_from_file(p.string(), m.albedo, ci)) continue;
        // если есть одноимённые маски — подхватываем
        for (size_t i = 0; i < spec_textures.size(); ++i)
            if (spec_textures[i].name == m.name + "_spec") m.spec_index = int(i);
        for (size_t i = 0; i < emissive_textures.size(); ++i)
            if (emissive_textures[i].name == m.name + "_emis") m.emissive_index = int(i);
        materials.push_back(std::move(m));
    }

    //процедурные альбедо
    {
        auto px = makeCheckerboard();
        Material m; m.name = "Checkerboard";
        vk_texture::CreateInfo ci; ci.srgb = true;
        if (!vk_texture::create_from_pixels(px.data(), 512, 512, m.albedo, ci)) return false;
        materials.push_back(std::move(m));
    }
    {
        auto px = makeUVGrid();
        Material m; m.name = "UV Grid";
        vk_texture::CreateInfo ci; ci.srgb = true;
        if (!vk_texture::create_from_pixels(px.data(), 512, 512, m.albedo, ci)) return false;
        materials.push_back(std::move(m));
    }

    if (materials.empty()) { std::cerr << "No textures at all\n"; return false; }

    // Раздаём материалы объектам по умолчанию
    objects[0].material_index = 0;
    objects[0].effect_mode    = 1;                 // рябь на первом конусе
    objects[1].material_index = std::min(1, int(materials.size()) - 1);
    floor_obj.material_index  = std::min(3, int(materials.size()) - 1);
    floor_obj.uv_scale        = glm::vec2(8.0f);
    wall_obj.material_index   = std::min(2, int(materials.size()) - 1);
    wall_obj.uv_scale         = glm::vec2(4.0f, 2.0f);
    return true;
}

const vk_texture::Texture& specTex(int i) {
    return (i <= 0 || i >= int(spec_textures.size())) ? black_tex : spec_textures[i].tex;
}
const vk_texture::Texture& emisTex(int i) {
    return (i <= 0 || i >= int(emissive_textures.size())) ? black_tex : emissive_textures[i].tex;
}

// Буферы и дескрипторы
bool createBuffersAndDescriptors() {
    auto& ctx = graphics::internal::context;

    if (!createTextures()) return false;

    // --- геометрия -> VRAM ---
    if (!uploadMesh(generateCone(kConeRadius, kConeHeight, 64), cone_mesh)) return false;
    if (!uploadMesh(generateCube(0.3f), viz_cube_mesh)) return false;
    if (!uploadMesh(generateLightConeWire(32), viz_cone_mesh)) return false;

    floor_obj.mesh = Mesh{};
    if (!uploadMesh(generateFloor(20.0f), floor_obj.mesh)) return false;
    floor_obj.label = "Floor";
    floor_obj.position = glm::vec3(0.0f);

    if (!uploadMesh(generateWall(20.0f, 10.0f), wall_obj.mesh)) return false;
    wall_obj.label = "Back Wall";
    wall_obj.position = glm::vec3(0.0f, 5.0f, -10.0f);

    if (!vk_buffer::create(sizeof(SceneUniforms), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, scene_uniform_buffer)) return false;
    if (!vk_buffer::create(sizeof(PointLight) * kMaxLights, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, point_lights_buffer)) return false;
    if (!vk_buffer::create(sizeof(SpotLight)  * kMaxLights, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, spot_lights_buffer)) return false;

    //выравненный шаг для динамического UBO
    VkPhysicalDeviceProperties props{};
    vkGetPhysicalDeviceProperties(ctx.physical_device, &props);
    const VkDeviceSize align = props.limits.minUniformBufferOffsetAlignment;
    uniforms_sizeof = ((sizeof(ModelUniforms) + align - 1) / align) * align;
    std::cout << "minUniformBufferOffsetAlignment = " << align
              << ", uniforms_sizeof = " << uniforms_sizeof << "\n";

    if (!vk_buffer::create(uniforms_sizeof * kMaxModels,
                           VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, models_buffer)) return false;

    //Layouts: set 0 сцена, set 1 материал, set 2 модель
    const VkDescriptorSetLayoutBinding scene_bindings[] = {
        {0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
        {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
        {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
    };
    const VkDescriptorSetLayoutCreateInfo scene_li = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = 3, .pBindings = scene_bindings,
    };
    if (vkCreateDescriptorSetLayout(ctx.device, &scene_li, nullptr, &scene_layout) != VK_SUCCESS) return false;

    //COMBINED_IMAGE_SAMPLER — albedo / specular / emissive
    const VkDescriptorSetLayoutBinding material_bindings[] = {
        {0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
        {1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
        {2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
    };
    const VkDescriptorSetLayoutCreateInfo material_li = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = 3, .pBindings = material_bindings,
    };
    if (vkCreateDescriptorSetLayout(ctx.device, &material_li, nullptr, &material_layout) != VK_SUCCESS) return false;

    // UNIFORM_BUFFER_DYNAMIC вместо UNIFORM_BUFFER
    const VkDescriptorSetLayoutBinding model_binding = {
        0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1,
        VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, nullptr
    };
    const VkDescriptorSetLayoutCreateInfo model_li = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = 1, .pBindings = &model_binding,
    };
    if (vkCreateDescriptorSetLayout(ctx.device, &model_li, nullptr, &model_layout) != VK_SUCCESS) return false;

    //Пул
    const uint32_t mat_count = uint32_t(materials.size());
    const VkDescriptorPoolSize pool_sizes[] = {
        {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,         1},
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,         2},
        {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1},
        {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 3 * mat_count},
    };
    const VkDescriptorPoolCreateInfo pool_info = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .maxSets = 1 + mat_count + 1,
        .poolSizeCount = 4, .pPoolSizes = pool_sizes,
    };
    if (vkCreateDescriptorPool(ctx.device, &pool_info, nullptr, &descriptor_pool) != VK_SUCCESS) return false;

    //Аллокация: 1 сцена + N материалов + 1 модель
    std::vector<VkDescriptorSetLayout> layouts;
    layouts.push_back(scene_layout);
    layouts.insert(layouts.end(), mat_count, material_layout);
    layouts.push_back(model_layout);

    std::vector<VkDescriptorSet> sets(layouts.size());
    const VkDescriptorSetAllocateInfo alloc = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = descriptor_pool,
        .descriptorSetCount = uint32_t(layouts.size()),
        .pSetLayouts = layouts.data(),
    };
    if (vkAllocateDescriptorSets(ctx.device, &alloc, sets.data()) != VK_SUCCESS) return false;

    scene_descriptor_set = sets[0];
    for (uint32_t i = 0; i < mat_count; ++i) materials[i].set = sets[1 + i];
    model_descriptor_set = sets[1 + mat_count];

    //Привязка set 0
    const VkDescriptorBufferInfo scene_ubo = {scene_uniform_buffer.handle, 0, sizeof(SceneUniforms)};
    const VkDescriptorBufferInfo pt_ssbo   = {point_lights_buffer.handle,  0, VK_WHOLE_SIZE};
    const VkDescriptorBufferInfo sp_ssbo   = {spot_lights_buffer.handle,   0, VK_WHOLE_SIZE};
    const VkWriteDescriptorSet scene_writes[] = {
        {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = scene_descriptor_set,
         .dstBinding = 0, .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
         .pBufferInfo = &scene_ubo},
        {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = scene_descriptor_set,
         .dstBinding = 1, .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
         .pBufferInfo = &pt_ssbo},
        {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = scene_descriptor_set,
         .dstBinding = 2, .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
         .pBufferInfo = &sp_ssbo},
    };
    vkUpdateDescriptorSets(ctx.device, 3, scene_writes, 0, nullptr);

    //привязка динамического UBO. offset = 0, range = ОДИН элемент.
    const VkDescriptorBufferInfo dyn_info = {models_buffer.handle, 0, sizeof(ModelUniforms)};
    const VkWriteDescriptorSet dyn_write = {
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstSet = model_descriptor_set, .dstBinding = 0, .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
        .pBufferInfo = &dyn_info,
    };
    vkUpdateDescriptorSets(ctx.device, 1, &dyn_write, 0, nullptr);

    //материалы заполняются в refreshMaterialSets()
    point_lights.push_back({glm::vec3(2.0f, 3.0f, 2.0f), 15.0f, glm::vec3(1.0f, 0.5f, 0.2f), 0.0f});
    spot_lights.push_back({glm::vec3(-2.0f, 4.0f, 0.0f), glm::radians(12.5f),
                           glm::vec3(0.2f, 0.5f, 1.0f), glm::radians(17.5f),
                           glm::vec3(0.0f, -1.0f, 0.0f), 15.0f});
    return true;
}

//перезапись наборов материалов, когда пользователь меняет маски
void refreshMaterialSets() {
    auto& ctx = graphics::internal::context;
    for (auto& m : materials) {
        if (!m.dirty) continue;
        const vk_texture::Texture& a = m.albedo;
        const vk_texture::Texture& s = specTex(m.spec_index);
        const vk_texture::Texture& e = emisTex(m.emissive_index);

        const VkDescriptorImageInfo infos[] = {
            {a.sampler, a.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL},
            {s.sampler, s.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL},
            {e.sampler, e.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL},
        };
        const VkWriteDescriptorSet writes[] = {
            {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = m.set, .dstBinding = 0,
             .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
             .pImageInfo = &infos[0]},
            {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = m.set, .dstBinding = 1,
             .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
             .pImageInfo = &infos[1]},
            {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = m.set, .dstBinding = 2,
             .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
             .pImageInfo = &infos[2]},
        };
        vkUpdateDescriptorSets(ctx.device, 3, writes, 0, nullptr);
        m.dirty = false;
    }
}

bool createPipeline() {
    auto& ctx = graphics::internal::context;
    VkShaderModule vs = loadShaderModule("cone.vert.spv");
    VkShaderModule fs = loadShaderModule("cone.frag.spv");
    if (!vs || !fs) return false;

    const VkPipelineShaderStageCreateInfo stages[] = {
        {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
         .stage = VK_SHADER_STAGE_VERTEX_BIT, .module = vs, .pName = "main"},
        {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
         .stage = VK_SHADER_STAGE_FRAGMENT_BIT, .module = fs, .pName = "main"},
    };
    auto binding_desc = Vertex::getBindingDescription();
    auto attr_desc    = Vertex::getAttributeDescriptions();

    const VkPipelineVertexInputStateCreateInfo vertex_input = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
        .vertexBindingDescriptionCount = 1, .pVertexBindingDescriptions = binding_desc.data(),
        .vertexAttributeDescriptionCount = uint32_t(attr_desc.size()),
        .pVertexAttributeDescriptions = attr_desc.data(),
    };
    const VkPipelineInputAssemblyStateCreateInfo ia_tris = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
        .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
    };
    const VkPipelineInputAssemblyStateCreateInfo ia_lines = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
        .topology = VK_PRIMITIVE_TOPOLOGY_LINE_LIST,
    };
    const VkPipelineViewportStateCreateInfo viewport = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
        .viewportCount = 1, .scissorCount = 1,
    };
    const VkPipelineRasterizationStateCreateInfo raster = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
        .polygonMode = VK_POLYGON_MODE_FILL,
        .cullMode = VK_CULL_MODE_NONE,
        .frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE,
        .lineWidth = 1.0f,
    };
    const VkPipelineMultisampleStateCreateInfo ms = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
        .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
    };
    const VkPipelineDepthStencilStateCreateInfo ds = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
        .depthTestEnable = VK_TRUE, .depthWriteEnable = VK_TRUE,
        .depthCompareOp = VK_COMPARE_OP_LESS,
    };
    const VkPipelineColorBlendAttachmentState blend_att = {
        .colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                          VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT,
    };
    const VkPipelineColorBlendStateCreateInfo blend = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
        .attachmentCount = 1, .pAttachments = &blend_att,
    };
    const VkDynamicState dyn_states[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    const VkPipelineDynamicStateCreateInfo dyn = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
        .dynamicStateCount = 2, .pDynamicStates = dyn_states,
    };
    const VkPushConstantRange pc_range = {
        .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT, .offset = 0, .size = sizeof(PushConstants),
    };
    //: три набора
    const VkDescriptorSetLayout set_layouts[] = {scene_layout, material_layout, model_layout};
    const VkPipelineLayoutCreateInfo layout_info = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 3, .pSetLayouts = set_layouts,
        .pushConstantRangeCount = 1, .pPushConstantRanges = &pc_range,
    };
    if (vkCreatePipelineLayout(ctx.device, &layout_info, nullptr, &pipeline_layout) != VK_SUCCESS) return false;

    auto build = [&](const VkPipelineInputAssemblyStateCreateInfo& ia, VkPipeline& out) {
        const VkGraphicsPipelineCreateInfo pi = {
            .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
            .stageCount = 2, .pStages = stages,
            .pVertexInputState = &vertex_input, .pInputAssemblyState = &ia,
            .pViewportState = &viewport, .pRasterizationState = &raster,
            .pMultisampleState = &ms, .pDepthStencilState = &ds,
            .pColorBlendState = &blend, .pDynamicState = &dyn,
            .layout = pipeline_layout, .renderPass = ctx.render_pass, .subpass = 0,
        };
        return vkCreateGraphicsPipelines(ctx.device, VK_NULL_HANDLE, 1, &pi, nullptr, &out) == VK_SUCCESS;
    };
    if (!build(ia_tris, pipeline))  return false;
    if (!build(ia_lines, line_pipeline)) return false;

    vkDestroyShaderModule(ctx.device, vs, nullptr);
    vkDestroyShaderModule(ctx.device, fs, nullptr);
    return true;
}

// ImGui
bool comboTexture(const char* label, int& index, const std::vector<NamedTexture>& list) {
    const char* preview = (index >= 0 && index < int(list.size())) ? list[index].name.c_str() : "?";
    bool changed = false;
    if (ImGui::BeginCombo(label, preview)) {
        for (int i = 0; i < int(list.size()); ++i) {
            const bool sel = (i == index);
            if (ImGui::Selectable(list[i].name.c_str(), sel)) { index = i; changed = true; }
            if (sel) ImGui::SetItemDefaultFocus();
        }
        ImGui::EndCombo();
    }
    return changed;
}

bool comboMaterial(const char* label, int& index) {
    const char* preview = (index >= 0 && index < int(materials.size())) ? materials[index].name.c_str() : "?";
    bool changed = false;
    if (ImGui::BeginCombo(label, preview)) {
        for (int i = 0; i < int(materials.size()); ++i) {
            const bool sel = (i == index);
            if (ImGui::Selectable(materials[i].name.c_str(), sel)) { index = i; changed = true; }
            if (sel) ImGui::SetItemDefaultFocus();
        }
        ImGui::EndCombo();
    }
    return changed;
}

const char* kEffectNames[] = {"Off", "Water Ripple", "TikTok Bulge", "Swirl"};

void effectUI() {
    if (ImGui::CollapsingHeader("Effect (push constants)", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::SliderFloat("Strength",  &effect.strength,  0.0f, 0.2f);
        ImGui::SliderFloat("Speed",     &effect.speed,     0.0f, 10.0f);
        ImGui::SliderFloat("Frequency", &effect.frequency, 1.0f, 120.0f);
        ImGui::SliderFloat("Radius",    &effect.radius,    0.02f, 0.5f);
        ImGui::SliderFloat2("Center",   &effect.center.x,  0.0f, 1.0f);
        ImGui::TextDisabled("Time = %.2f s", current_time);
    }
}

void materialUI(const char* id, int& mat_index, int& effect_mode,
                glm::vec2& uv_scale, glm::vec2& uv_offset,
                float& emis_strength, float& spec_strength) {
    ImGui::PushID(id);
    comboMaterial("Albedo texture", mat_index);
    if (mat_index >= 0 && mat_index < int(materials.size())) {
        Material& m = materials[mat_index];
        if (comboTexture("Specular map", m.spec_index, spec_textures))     m.dirty = true;
        if (comboTexture("Emissive map", m.emissive_index, emissive_textures)) m.dirty = true;
    }
    ImGui::Combo("Effect", &effect_mode, kEffectNames, 4);
    ImGui::SliderFloat2("UV scale",  &uv_scale.x,  0.1f, 16.0f);
    ImGui::SliderFloat2("UV offset", &uv_offset.x, -2.0f, 2.0f);
    ImGui::SliderFloat("Emissive strength", &emis_strength, 0.0f, 5.0f);
    ImGui::SliderFloat("Specular strength", &spec_strength, 0.0f, 3.0f);
    ImGui::PopID();
}

} // namespace

bool initialize() {
    if (!createBuffersAndDescriptors()) return false;
    if (!createPipeline()) return false;
    return true;
}

void shutdown() {
    auto& ctx = graphics::internal::context;
    vkQueueWaitIdle(ctx.graphics_queue);

    vkDestroyPipeline(ctx.device, pipeline, nullptr);
    vkDestroyPipeline(ctx.device, line_pipeline, nullptr);
    vkDestroyPipelineLayout(ctx.device, pipeline_layout, nullptr);
    vkDestroyDescriptorSetLayout(ctx.device, scene_layout, nullptr);
    vkDestroyDescriptorSetLayout(ctx.device, material_layout, nullptr);
    vkDestroyDescriptorSetLayout(ctx.device, model_layout, nullptr);
    vkDestroyDescriptorPool(ctx.device, descriptor_pool, nullptr);

    auto destroyMesh = [](Mesh& m) { vk_buffer::destroy(m.vb); vk_buffer::destroy(m.ib); };
    destroyMesh(cone_mesh); destroyMesh(viz_cube_mesh); destroyMesh(viz_cone_mesh);
    destroyMesh(floor_obj.mesh); destroyMesh(wall_obj.mesh);

    vk_buffer::destroy(scene_uniform_buffer);
    vk_buffer::destroy(point_lights_buffer);
    vk_buffer::destroy(spot_lights_buffer);
    vk_buffer::destroy(models_buffer);

    for (auto& m : materials) vk_texture::destroy(m.albedo);
    for (auto& t : spec_textures)     if (t.tex.image) vk_texture::destroy(t.tex);
    for (auto& t : emissive_textures) if (t.tex.image) vk_texture::destroy(t.tex);
    vk_texture::destroy(black_tex);
}

void update(double time) {
    const double dt = (previous_time >= 0.0) ? (time - previous_time) : 0.0;
    previous_time = time;
    current_time = static_cast<float>(time);

    ImGui::Begin("Lab 3: Texture Mapping");

    if (ImGui::CollapsingHeader("Camera", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::SliderFloat3("Position", camera.pos, -15.0f, 15.0f);
        ImGui::SliderFloat("Yaw",   &camera.yaw,   -180.0f, 180.0f);
        ImGui::SliderFloat("Pitch", &camera.pitch, -89.0f, 89.0f);
        ImGui::RadioButton("Perspective",  &projection_type, 0); ImGui::SameLine();
        ImGui::RadioButton("Orthographic", &projection_type, 1);
        if (projection_type == 0) ImGui::SliderFloat("FOV", &camera.fov, 10.0f, 120.0f);
    }

    if (ImGui::CollapsingHeader("Lighting", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::SliderFloat("Master (0 = darkness)", &light_master, 0.0f, 1.0f);
        ImGui::SliderFloat3("Direction", &directional_light_dir.x, -1.0f, 1.0f);
        if (glm::length(directional_light_dir) < 1e-4f) directional_light_dir = glm::vec3(0, -1, 0);
        directional_light_dir = glm::normalize(directional_light_dir);
        ImGui::ColorEdit3("Color",   &directional_light_col.x);
        ImGui::ColorEdit3("Ambient", &ambient_light.x);
        if (ImGui::Button("Lights OFF (show emissive)")) light_master = 0.0f;
        ImGui::SameLine();
        if (ImGui::Button("Lights ON")) light_master = 1.0f;
    }

    if (ImGui::CollapsingHeader("Material (global)")) {
        ImGui::ColorEdit3("Albedo tint", &material_albedo.x);
        ImGui::ColorEdit3("Specular color", &material_specular.x);
        ImGui::SliderFloat("Shininess", &material_shininess, 1.0f, 256.0f);
    }

    effectUI();

    if (ImGui::CollapsingHeader("Point Lights")) {
        ImGui::BeginDisabled(point_lights.size() >= kMaxLights);
        if (ImGui::Button("Add Point Light"))
            point_lights.push_back({glm::vec3(0,3,0), 15.0f, glm::vec3(1), 0.0f});
        ImGui::EndDisabled();
        for (size_t i = 0; i < point_lights.size(); ++i) {
            ImGui::PushID(int(i));
            if (ImGui::TreeNode(("Point " + std::to_string(i)).c_str())) {
                ImGui::SliderFloat3("Position", &point_lights[i].position.x, -8.0f, 8.0f);
                ImGui::SliderFloat("Intensity", &point_lights[i].intensity, 0.0f, 50.0f);
                ImGui::ColorEdit3("Color", &point_lights[i].color.x);
                if (ImGui::Button("Remove")) {
                    point_lights.erase(point_lights.begin() + i);
                    ImGui::TreePop(); ImGui::PopID(); break;
                }
                ImGui::TreePop();
            }
            ImGui::PopID();
        }
    }

    if (ImGui::CollapsingHeader("Spot Lights")) {
        ImGui::BeginDisabled(spot_lights.size() >= kMaxLights);
        if (ImGui::Button("Add Spot Light"))
            spot_lights.push_back({glm::vec3(0,3,0), glm::radians(12.5f), glm::vec3(1),
                                   glm::radians(17.5f), glm::vec3(0,-1,0), 15.0f});
        ImGui::EndDisabled();
        for (size_t i = 0; i < spot_lights.size(); ++i) {
            ImGui::PushID(int(i));
            if (ImGui::TreeNode(("Spot " + std::to_string(i)).c_str())) {
                auto& s = spot_lights[i];
                ImGui::SliderFloat3("Position", &s.position.x, -8.0f, 8.0f);
                ImGui::SliderFloat3("Direction", &s.direction.x, -1.0f, 1.0f);
                if (glm::length(s.direction) < 1e-4f) s.direction = glm::vec3(0,-1,0);
                s.direction = glm::normalize(s.direction);
                ImGui::SliderFloat("Inner Angle", &s.inner_angle, 0.0f, glm::radians(45.0f));
                ImGui::SliderFloat("Outer Angle", &s.outer_angle, 0.0f, glm::radians(60.0f));
                s.inner_angle = std::min(s.inner_angle, s.outer_angle);
                ImGui::SliderFloat("Intensity", &s.intensity, 0.0f, 50.0f);
                ImGui::ColorEdit3("Color", &s.color.x);
                if (ImGui::Button("Remove")) {
                    spot_lights.erase(spot_lights.begin() + i);
                    ImGui::TreePop(); ImGui::PopID(); break;
                }
                ImGui::TreePop();
            }
            ImGui::PopID();
        }
    }

    ImGui::Checkbox("Show Light Visualization", &show_light_visualization);

    for (size_t i = 0; i < objects.size(); ++i) {
        ImGui::PushID(int(i) + 1000);
        if (ImGui::CollapsingHeader(("Object " + std::to_string(i + 1)).c_str(),
                                    ImGuiTreeNodeFlags_DefaultOpen)) {
            auto& o = objects[i];
            materialUI("obj", o.material_index, o.effect_mode, o.uv_scale, o.uv_offset,
                       o.emissive_strength, o.specular_strength);
            ImGui::Separator();
            ImGui::Checkbox("Animate", &o.trajectory);
            if (o.trajectory) {
                if (ImGui::Button(o.playing ? "Pause" : "Play")) o.playing = !o.playing;
                ImGui::SliderFloat("Speed", &o.speed, 0.1f, 3.0f);
                ImGui::SliderFloat("Radius", &o.radius, 1.0f, 4.0f);
                ImGui::SliderFloat("Bob height", &o.bob_amplitude, 0.0f, 1.0f);
            } else {
                ImGui::SliderFloat3("Position", &o.position.x, -5.0f, 5.0f);
            }
            ImGui::SliderFloat3("Rotation", &o.rotation.x, -180.0f, 180.0f);
            ImGui::SliderFloat3("Scale", &o.scale.x, 0.2f, 3.0f);
            ImGui::ColorEdit3("Tint", &o.tint.x);
            ImGui::SliderFloat("Vertex color mix", &o.vertex_color_mix, 0.0f, 1.0f);
        }
        ImGui::PopID();
    }

    for (StaticObject* so : {&floor_obj, &wall_obj}) {
        ImGui::PushID(so->label);
        if (ImGui::CollapsingHeader(so->label)) {
            ImGui::Checkbox("Visible", &so->visible);
            materialUI("static", so->material_index, so->effect_mode, so->uv_scale, so->uv_offset,
                       so->emissive_strength, so->specular_strength);
            ImGui::SliderFloat("Shininess", &so->shininess, 1.0f, 256.0f);
            ImGui::ColorEdit3("Tint", &so->tint.x);
        }
        ImGui::PopID();
    }

    ImGui::End();

    for (auto& o : objects)
        if (o.trajectory && o.playing) o.phase += float(dt) * o.speed * 0.6f;
}

// Рендер
void render(const graphics::internal::FrameData& fd) {
    auto& ctx = graphics::internal::context;

    refreshMaterialSets();   //fence предыдущего кадра уже дождались в prepare()

    const float aspect = (ctx.swapchain_extent.height > 0)
        ? float(ctx.swapchain_extent.width) / float(ctx.swapchain_extent.height) : 1.0f;

    const glm::vec3 cam_pos = glm::make_vec3(camera.pos);
    glm::vec3 front;
    front.x = std::cos(glm::radians(camera.yaw)) * std::cos(glm::radians(camera.pitch));
    front.y = std::sin(glm::radians(camera.pitch));
    front.z = std::sin(glm::radians(camera.yaw)) * std::cos(glm::radians(camera.pitch));
    front = glm::normalize(front);

    const glm::mat4 view = glm::lookAt(cam_pos, cam_pos + front, glm::vec3(0, 1, 0));
    glm::mat4 proj;
    if (projection_type == 0) {
        proj = glm::perspective(glm::radians(camera.fov), aspect, 0.1f, 100.0f);
    } else {
        constexpr float e = 5.0f;
        proj = glm::ortho(-e * aspect, e * aspect, -e, e, -100.0f, 100.0f);
    }
    proj[1][1] *= -1.0f;

    const size_t point_count = std::min(point_lights.size(), kMaxLights);
    const size_t spot_count  = std::min(spot_lights.size(),  kMaxLights);

    SceneUniforms su{};
    su.view_projection         = proj * view;
    su.camera_position         = cam_pos;
    su.directional_light_dir   = directional_light_dir;
    su.directional_light_col   = directional_light_col * light_master;
    su.ambient_light           = ambient_light * light_master;
    su.point_light_count       = uint32_t(point_count);
    su.spot_light_count        = uint32_t(spot_count);
    std::memcpy(scene_uniform_buffer.mapped, &su, sizeof(SceneUniforms));

    {
        std::vector<PointLight> pl(point_lights.begin(), point_lights.begin() + point_count);
        std::vector<SpotLight> sl(spot_lights.begin(),  spot_lights.begin()  + spot_count);
        for (auto& l : pl) l.intensity *= light_master;
        for (auto& l : sl) l.intensity *= light_master;
        if (!pl.empty()) std::memcpy(point_lights_buffer.mapped, pl.data(), pl.size() * sizeof(PointLight));
        if (!sl.empty()) std::memcpy(spot_lights_buffer.mapped,  sl.data(), sl.size() * sizeof(SpotLight));
    }

    uint32_t model_slot = 0;
    auto push_model = [&](const ModelUniforms& m) -> uint32_t {
        const uint32_t idx = model_slot++;
        if (idx >= kMaxModels) return 0;
        auto* dst = static_cast<std::byte*>(models_buffer.mapped) + idx * uniforms_sizeof;
        std::memcpy(dst, &m, sizeof(ModelUniforms));
        return idx;
    };

    auto make_pc = [&](int mode) {
        PushConstants pc{};
        pc.time        = current_time;
        pc.effect_mode = uint32_t(mode);
        pc.strength    = effect.strength;
        pc.speed       = effect.speed;
        pc.center      = effect.center;
        pc.frequency   = effect.frequency;
        pc.radius      = effect.radius;
        return pc;
    };

    const VkCommandBufferBeginInfo begin = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    vkBeginCommandBuffer(fd.command_buffer, &begin);

    const VkClearValue clears[] = {{.color = {{0.03f, 0.03f, 0.05f, 1.0f}}}, {.depthStencil = {1.0f, 0}}};
    const VkRenderPassBeginInfo rp = {
        .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
        .renderPass = ctx.render_pass, .framebuffer = fd.framebuffer,
        .renderArea = {.extent = ctx.swapchain_extent},
        .clearValueCount = 2, .pClearValues = clears,
    };
    vkCmdBeginRenderPass(fd.command_buffer, &rp, VK_SUBPASS_CONTENTS_INLINE);
    vkCmdBindPipeline(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);

    const VkViewport vp = {0.0f, 0.0f, float(ctx.swapchain_extent.width),
                           float(ctx.swapchain_extent.height), 0.0f, 1.0f};
    vkCmdSetViewport(fd.command_buffer, 0, 1, &vp);
    const VkRect2D sc = {{0,0}, ctx.swapchain_extent};
    vkCmdSetScissor(fd.command_buffer, 0, 1, &sc);

    vkCmdBindDescriptorSets(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                            pipeline_layout, 0, 1, &scene_descriptor_set, 0, nullptr);

    const VkDeviceSize zero_offsets[] = {0};

    //привязка на draw: set1 = материал, set2 = модель + динамический сдвиг
    auto draw = [&](const Mesh& mesh, uint32_t slot, int mat_index, int effect_mode) {
        const uint32_t dyn_offset = uint32_t(slot * uniforms_sizeof);
        const int mi = std::clamp(mat_index, 0, int(materials.size()) - 1);
        const VkDescriptorSet sets[] = { materials[mi].set, model_descriptor_set };
        vkCmdBindDescriptorSets(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                pipeline_layout, 1, 2, sets,
                                1, &dyn_offset);            // сдвиг
        const PushConstants pc = make_pc(effect_mode);
        vkCmdPushConstants(fd.command_buffer, pipeline_layout, VK_SHADER_STAGE_FRAGMENT_BIT,
                           0, sizeof(PushConstants), &pc);
        const VkBuffer vb[] = {mesh.vb.handle};
        vkCmdBindVertexBuffers(fd.command_buffer, 0, 1, vb, zero_offsets);
        vkCmdBindIndexBuffer(fd.command_buffer, mesh.ib.handle, 0, VK_INDEX_TYPE_UINT32);
        vkCmdDrawIndexed(fd.command_buffer, mesh.index_count, 1, 0, 0, 0);
    };

    //статические объекты: пол и стена
    for (StaticObject* so : {&floor_obj, &wall_obj}) {
        if (!so->visible) continue;
        ModelUniforms mu{};
        glm::mat4 m = glm::translate(glm::mat4(1.0f), so->position);
        m = glm::rotate(m, glm::radians(so->rotation.y), glm::vec3(0,1,0));
        mu.model      = glm::scale(m, so->scale);
        mu.albedo     = material_albedo;
        mu.specular   = material_specular;
        mu.shininess  = so->shininess;
        mu.unlit      = 0.0f;
        mu.tint       = glm::vec4(so->tint, 0.0f);
        mu.uv_scale   = so->uv_scale;
        mu.uv_offset  = so->uv_offset;
        mu.params     = glm::vec4(so->emissive_strength, so->specular_strength, 0, 0);
        draw(so->mesh, push_model(mu), so->material_index, so->effect_mode);
    }

    //анимированные конусы
    for (size_t i = 0; i < objects.size(); ++i) {
        const auto& o = objects[i];
        ModelUniforms mu{};
        glm::vec3 pos = o.position;
        if (o.trajectory) {
            const float t = o.phase + float(i) * 2.094f;
            const float denom = 1.0f + std::sin(t) * std::sin(t);
            const float half_h = 0.5f * kConeHeight * o.scale.y;
            const float lift = o.bob_amplitude * 0.5f * (1.0f + std::sin(2.0f * t));
            pos = { o.radius * std::cos(t) / denom, half_h + lift,
                    o.radius * std::sin(t) * std::cos(t) / denom };
        }
        glm::mat4 m = glm::translate(glm::mat4(1.0f), pos);
        m = glm::rotate(m, glm::radians(o.rotation.x), glm::vec3(1,0,0));
        m = glm::rotate(m, glm::radians(o.rotation.y), glm::vec3(0,1,0));
        m = glm::rotate(m, glm::radians(o.rotation.z), glm::vec3(0,0,1));
        mu.model     = glm::scale(m, o.scale);
        mu.albedo    = material_albedo;
        mu.specular  = material_specular;
        mu.shininess = material_shininess;
        mu.unlit     = 0.0f;
        mu.tint      = glm::vec4(o.tint, o.vertex_color_mix);
        mu.uv_scale  = o.uv_scale;
        mu.uv_offset = o.uv_offset;
        mu.params    = glm::vec4(o.emissive_strength, o.specular_strength, 0, 0);
        draw(cone_mesh, push_model(mu), o.material_index, o.effect_mode);
    }

    //визуализация источников: unlit, текстура игнорируется в шейдере
    if (show_light_visualization) {
        auto draw_viz = [&](const glm::mat4& model, const glm::vec3& color, const Mesh& mesh) {
            ModelUniforms mu{};
            mu.model     = model;
            mu.unlit     = 1.0f;
            mu.tint      = glm::vec4(color, 0.0f);
            mu.uv_scale  = glm::vec2(1.0f);
            draw(mesh, push_model(mu), 0, 0);
        };
        for (size_t i = 0; i < point_count; ++i)
            draw_viz(glm::translate(glm::mat4(1.0f), point_lights[i].position),
                     point_lights[i].color, viz_cube_mesh);
        for (size_t i = 0; i < spot_count; ++i)
            draw_viz(glm::translate(glm::mat4(1.0f), spot_lights[i].position),
                     spot_lights[i].color, viz_cube_mesh);

        vkCmdBindPipeline(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, line_pipeline);
        for (size_t i = 0; i < spot_count; ++i) {
            const auto& l = spot_lights[i];
            const glm::vec3 dir = glm::normalize(l.direction);
            const glm::vec3 ay = -dir;
            const glm::vec3 helper = std::abs(ay.y) > 0.99f ? glm::vec3(1,0,0) : glm::vec3(0,1,0);
            const glm::vec3 ax = glm::normalize(glm::cross(helper, ay));
            const glm::vec3 az = glm::cross(ax, ay);
            const glm::mat4 rot(glm::vec4(ax,0), glm::vec4(ay,0), glm::vec4(az,0), glm::vec4(0,0,0,1));
            const float r = kSpotVizLength * std::tan(std::min(l.outer_angle, glm::radians(80.0f)));
            const glm::mat4 model = glm::translate(glm::mat4(1.0f), l.position) * rot
                                  * glm::scale(glm::mat4(1.0f), glm::vec3(r, kSpotVizLength, r));
            draw_viz(model, glm::vec3(1.0f), viz_cone_mesh);
        }
    }

    vkCmdEndRenderPass(fd.command_buffer);
    vkEndCommandBuffer(fd.command_buffer);
}

} // namespace application