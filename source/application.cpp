#include "geometry.hpp"
#include "graphics_internal.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <array>

#define GLM_ENABLE_EXPERIMENTAL
#include <vulkan/vulkan.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/constants.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/quaternion.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <imgui.h>

#include "vk_buffer.hpp"

namespace application {

namespace {

// ---------------------------------------------------------------------------
// Структуры, лежащие в GPU-буферах. Их layout обязан совпадать с блоками
// в cone.vert / cone.frag (std140 для uniform, std430 для storage).
// ---------------------------------------------------------------------------
struct alignas(16) SceneUniforms {
    glm::mat4 view_projection;       // offset 0
    glm::vec3 camera_position;       // offset 64
    float _pad0;
    glm::vec3 directional_light_dir; // offset 80
    float _pad1;
    glm::vec3 directional_light_col; // offset 96
    float _pad2;
    glm::vec3 ambient_light;         // offset 112
    float _pad3;
    uint32_t point_light_count;      // offset 128
    uint32_t spot_light_count;       // offset 132
    uint32_t _pad4[2];
};
static_assert(offsetof(SceneUniforms, camera_position) == 64, "std140 mismatch");
static_assert(offsetof(SceneUniforms, directional_light_dir) == 80, "std140 mismatch");
static_assert(offsetof(SceneUniforms, ambient_light) == 112, "std140 mismatch");
static_assert(offsetof(SceneUniforms, point_light_count) == 128, "std140 mismatch");
static_assert(sizeof(SceneUniforms) == 144, "SceneUniforms must be 144 bytes");

struct alignas(16) ModelUniforms {
    glm::mat4 model;                 // offset 0
    glm::vec3 albedo;                // offset 64
    float _pad0;
    glm::vec3 specular;              // offset 80
    float _pad1;
    float shininess;                 // offset 96
    float _pad2[3];
};
static_assert(offsetof(ModelUniforms, albedo) == 64, "std140 mismatch");
static_assert(offsetof(ModelUniforms, specular) == 80, "std140 mismatch");
static_assert(offsetof(ModelUniforms, shininess) == 96, "std140 mismatch");
static_assert(sizeof(ModelUniforms) % 16 == 0, "ModelUniforms size must be multiple of 16");

struct alignas(16) PointLight {
    glm::vec3 position;
    float intensity;
    glm::vec3 color;
    float _pad;
};
static_assert(sizeof(PointLight) == 32 && offsetof(PointLight, color) == 16, "std430 mismatch");

struct alignas(16) SpotLight {
    glm::vec3 position;
    float inner_angle;
    glm::vec3 color;
    float outer_angle;
    glm::vec3 direction;
    float intensity;
};
static_assert(sizeof(SpotLight) == 48 && offsetof(SpotLight, direction) == 32, "std430 mismatch");

struct SceneObject {
    float position[3]{0.0f, 1.0f, 0.0f};
    float rotation[3]{0.0f, 0.0f, 0.0f};
    float scale[3]{1.0f, 1.0f, 1.0f};
    float tint[3]{1.0f, 1.0f, 1.0f};
    bool trajectory = true;
    bool playing = true;
    float phase = 0.0f;
    float speed = 0.8f;
    float radius = 2.0f;
    float bob_amplitude = 0.4f;

    vk_buffer::Buffer model_uniform_buffer;
    VkDescriptorSet model_descriptor_set = VK_NULL_HANDLE;
};

struct Camera {
    float pos[3]{0.0f, 3.0f, 6.0f};
    float yaw = -90.0f;
    float pitch = -15.0f;
    float fov = 45.0f;
};

struct PushConstants {
    glm::vec4 tintColor;
    float is_unlit;
    float _pad[3];
};

struct MeshData {
    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;
};

struct FloorMeshData {
    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;
};

struct FloorObject {
    vk_buffer::Buffer vertex_buffer;
    vk_buffer::Buffer index_buffer;
    vk_buffer::Buffer uniform_buffer;
    VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
    uint32_t index_count = 0;
    float size = 20.0f;
    float tint[3] = {0.5f, 0.5f, 0.5f};
    bool show_floor = true;
};

// Один "слот" модели для визуализации источника. Каждая фигура визуализации получает СВОЙ uniform-буфер и дескриптор
struct VizSlot {
    vk_buffer::Buffer buffer;
    VkDescriptorSet set = VK_NULL_HANDLE;
};

constexpr size_t kObjectCount = 2;
constexpr size_t kMaxLights = 20;
constexpr size_t kVizSlotCount = kMaxLights * 3;  // куб точечного + куб и конус прожектора
constexpr float kConeRadius = 1.0f;
constexpr float kConeHeight = 2.0f;
constexpr float kSpotVizLength = 2.0f;

std::array<SceneObject, kObjectCount> objects;

Camera camera;
glm::vec3 directional_light_dir = glm::normalize(glm::vec3(-0.5f, -1.0f, -0.5f));
glm::vec3 directional_light_col = glm::vec3(1.0f);
glm::vec3 ambient_light = glm::vec3(0.15f);
std::vector<PointLight> point_lights;
std::vector<SpotLight> spot_lights;

glm::vec3 material_albedo = glm::vec3(1.0f, 1.0f, 1.0f);
glm::vec3 material_specular = glm::vec3(1.0f, 1.0f, 1.0f);
float material_shininess = 32.0f;

vk_buffer::Buffer vertex_buffer;
vk_buffer::Buffer index_buffer;
uint32_t current_index_count = 0;

vk_buffer::Buffer scene_uniform_buffer;
vk_buffer::Buffer point_lights_buffer;
vk_buffer::Buffer spot_lights_buffer;

VkDescriptorSet scene_descriptor_set = VK_NULL_HANDLE;

VkDescriptorSetLayout scene_layout = VK_NULL_HANDLE;
VkDescriptorSetLayout model_layout = VK_NULL_HANDLE;
VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
VkPipeline pipeline = VK_NULL_HANDLE;
VkPipeline line_pipeline = VK_NULL_HANDLE;

int projection_type = 0; // 0 - Perspective, 1 - Orthographicbool 
double previous_time = -1.0;

FloorObject floor_obj;
bool show_light_visualization = true;

std::array<VizSlot, kVizSlotCount> viz_slots;

MeshData viz_cube_mesh;
vk_buffer::Buffer viz_cube_vb, viz_cube_ib;
uint32_t viz_cube_index_count = 0;

MeshData viz_cone_mesh;
vk_buffer::Buffer viz_cone_vb, viz_cone_ib;
uint32_t viz_cone_index_count = 0;

VkShaderModule loadShaderModule(const char* filename) {
    const std::string path = std::string(SHADER_DIR) + filename;
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) return VK_NULL_HANDLE;
    const size_t size = size_t(file.tellg());
    std::vector<char> code(size);
    file.seekg(0);
    file.read(code.data(), std::streamsize(size));
    file.close();

    const VkShaderModuleCreateInfo create_info = {
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = code.size(),
        .pCode = reinterpret_cast<const uint32_t*>(code.data()),
    };
    VkShaderModule module = VK_NULL_HANDLE;
    vkCreateShaderModule(graphics::internal::context.device, &create_info, nullptr, &module);
    return module;
}

FloorMeshData generateFloor(float size) {
    FloorMeshData mesh;
    const float half_size = size * 0.5f;
    mesh.vertices.push_back({ .position = {-half_size, 0.0f, -half_size}, .normal = {0.0f, 1.0f, 0.0f}, .color = {1.0f, 1.0f, 1.0f} });
    mesh.vertices.push_back({ .position = { half_size, 0.0f, -half_size}, .normal = {0.0f, 1.0f, 0.0f}, .color = {1.0f, 1.0f, 1.0f} });
    mesh.vertices.push_back({ .position = { half_size, 0.0f,  half_size}, .normal = {0.0f, 1.0f, 0.0f}, .color = {1.0f, 1.0f, 1.0f} });
    mesh.vertices.push_back({ .position = {-half_size, 0.0f,  half_size}, .normal = {0.0f, 1.0f, 0.0f}, .color = {1.0f, 1.0f, 1.0f} });
    mesh.indices = {0, 1, 2, 0, 2, 3};
    return mesh;
}

glm::vec3 hsv2rgb(float h, float s, float v) {
    h -= std::floor(h);
    const glm::vec3 k(
        std::clamp(std::abs(h * 6.0f - 3.0f) - 1.0f, 0.0f, 1.0f),
        std::clamp(2.0f - std::abs(h * 6.0f - 2.0f), 0.0f, 1.0f),
        std::clamp(2.0f - std::abs(h * 6.0f - 4.0f), 0.0f, 1.0f));
    return v * glm::mix(glm::vec3(1.0f), k, s);
}

MeshData generateCone(float radius, float height, uint32_t segments) {
    MeshData mesh;
    if (segments < 3) return mesh;

    const float half_h = height * 0.5f;
    const float two_pi = 2.0f * glm::pi<float>();
    const float dtheta = two_pi / float(segments);
    const float slope = std::sqrt(height * height + radius * radius);
    const glm::vec3 apex(0.0f, half_h, 0.0f);

    auto side_normal = [&](float theta) {
        return glm::vec3(height * std::cos(theta) / slope, radius / slope, height * std::sin(theta) / slope);
    };
    auto rim_pos = [&](float theta) {
        return glm::vec3(radius * std::cos(theta), -half_h, radius * std::sin(theta));
    };

    constexpr float kRainbowSat = 0.7f;   // насыщенность: 0 = серый, 1 = чистый цвет
    constexpr float kRainbowVal = 0.6f;   // яркость
    for (uint32_t i = 0; i < segments; ++i) {
        const float theta = dtheta * float(i);
        mesh.vertices.push_back({ rim_pos(theta), side_normal(theta), hsv2rgb(theta / two_pi, kRainbowSat, kRainbowVal) });
    }
    // [segments, 2*segments): вершина конуса, по одной на сегмент
    for (uint32_t i = 0; i < segments; ++i) {
        const float theta_mid = dtheta * (float(i) + 0.5f);
        const glm::vec3 c = glm::mix(hsv2rgb(theta_mid / two_pi, kRainbowSat, kRainbowVal), glm::vec3(kRainbowVal), 0.5f);
        mesh.vertices.push_back({ apex, side_normal(theta_mid), c });
    }
    // Основание: центр + собственное кольцо с нормалью вниз
    const uint32_t base_center = 2 * segments;
    const uint32_t base_ring = base_center + 1;
    mesh.vertices.push_back({ glm::vec3(0.0f, -half_h, 0.0f), glm::vec3(0.0f, -1.0f, 0.0f), glm::vec3(0.8f) });
    for (uint32_t i = 0; i < segments; ++i) {
        const float theta = dtheta * float(i);
        mesh.vertices.push_back({ rim_pos(theta), glm::vec3(0.0f, -1.0f, 0.0f), hsv2rgb(theta / two_pi, kRainbowSat, kRainbowVal * 0.9f) });
    }

    for (uint32_t i = 0; i < segments; ++i) {
        const uint32_t next = (i + 1) % segments;
        // боковая грань (порядок обхода — наружу)
        mesh.indices.push_back(segments + i);
        mesh.indices.push_back(next);
        mesh.indices.push_back(i);
        // основание (смотрит вниз)
        mesh.indices.push_back(base_center);
        mesh.indices.push_back(base_ring + i);
        mesh.indices.push_back(base_ring + next);
    }
    return mesh;
}

MeshData generateCube(float size) {
    MeshData mesh;
    float h = size * 0.5f;
    mesh.vertices = {
        {{-h,-h,-h}, {0,0,-1}, {1,1,1}}, {{h,-h,-h}, {0,0,-1}, {1,1,1}}, {{h,h,-h}, {0,0,-1}, {1,1,1}}, {{-h,h,-h}, {0,0,-1}, {1,1,1}},
        {{-h,-h,h}, {0,0,1}, {1,1,1}}, {{h,-h,h}, {0,0,1}, {1,1,1}}, {{h,h,h}, {0,0,1}, {1,1,1}}, {{-h,h,h}, {0,0,1}, {1,1,1}},
        {{-h,-h,-h}, {0,-1,0}, {1,1,1}}, {{h,-h,-h}, {0,-1,0}, {1,1,1}}, {{h,-h,h}, {0,-1,0}, {1,1,1}}, {{-h,-h,h}, {0,-1,0}, {1,1,1}},
        {{-h,h,-h}, {0,1,0}, {1,1,1}}, {{h,h,-h}, {0,1,0}, {1,1,1}}, {{h,h,h}, {0,1,0}, {1,1,1}}, {{-h,h,h}, {0,1,0}, {1,1,1}},
        {{-h,-h,-h}, {-1,0,0}, {1,1,1}}, {{-h,-h,h}, {-1,0,0}, {1,1,1}}, {{-h,h,h}, {-1,0,0}, {1,1,1}}, {{-h,h,-h}, {-1,0,0}, {1,1,1}},
        {{h,-h,-h}, {1,0,0}, {1,1,1}}, {{h,-h,h}, {1,0,0}, {1,1,1}}, {{h,h,h}, {1,0,0}, {1,1,1}}, {{h,h,-h}, {1,0,0}, {1,1,1}}
    };
    mesh.indices = {
        0,1,2, 0,2,3, 4,6,5, 4,7,6, 8,9,10, 8,10,11, 12,13,14, 12,14,15, 16,17,18, 16,18,19, 20,22,21, 20,23,22
    };
    return mesh;
}


MeshData generateLightConeWire(uint32_t segments) {
    MeshData mesh;
    const glm::vec3 white(1.0f);
    const glm::vec3 up(0.0f, 1.0f, 0.0f);
    mesh.vertices.push_back({ glm::vec3(0.0f), up, white });                        
    for (uint32_t i = 0; i < segments; ++i) {
        const float theta = 2.0f * glm::pi<float>() * float(i) / float(segments);
        mesh.vertices.push_back({ glm::vec3(std::cos(theta), -1.0f, std::sin(theta)), up, white });
    }
    const uint32_t axis_end = segments + 1;
    mesh.vertices.push_back({ glm::vec3(0.0f, -1.0f, 0.0f), up, white });

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

bool uploadMesh(const MeshData& mesh, vk_buffer::Buffer& vb, vk_buffer::Buffer& ib) {
    const VkDeviceSize v_size = sizeof(mesh.vertices[0]) * mesh.vertices.size();
    const VkDeviceSize i_size = sizeof(mesh.indices[0]) * mesh.indices.size();
    if (!vk_buffer::create(v_size, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, vb)) return false;
    std::memcpy(vb.mapped, mesh.vertices.data(), v_size);
    if (!vk_buffer::create(i_size, VK_BUFFER_USAGE_INDEX_BUFFER_BIT, ib)) return false;
    std::memcpy(ib.mapped, mesh.indices.data(), i_size);
    return true;
}

bool createBuffersAndDescriptors() {
    auto& context = graphics::internal::context;

    MeshData cone = generateCone(kConeRadius, kConeHeight, 36);
    current_index_count = static_cast<uint32_t>(cone.indices.size());
    if (!uploadMesh(cone, vertex_buffer, index_buffer)) return false;

    if (!vk_buffer::create(sizeof(SceneUniforms), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, scene_uniform_buffer)) return false;

    if (!vk_buffer::create(sizeof(PointLight) * kMaxLights, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, point_lights_buffer)) return false;
    if (!vk_buffer::create(sizeof(SpotLight) * kMaxLights, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, spot_lights_buffer)) return false;

    for (auto& obj : objects) {
        if (!vk_buffer::create(sizeof(ModelUniforms), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, obj.model_uniform_buffer)) return false;
    }

    const VkDescriptorSetLayoutBinding scene_bindings[] = {
        {0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT},
        {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT},
        {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT},
    };
    const VkDescriptorSetLayoutCreateInfo scene_layout_info = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = 3,
        .pBindings = scene_bindings,
    };
    if (vkCreateDescriptorSetLayout(context.device, &scene_layout_info, nullptr, &scene_layout) != VK_SUCCESS) return false;

    const VkDescriptorSetLayoutBinding model_binding = {
        0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT
    };
    const VkDescriptorSetLayoutCreateInfo model_layout_info = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = 1,
        .pBindings = &model_binding,
    };
    if (vkCreateDescriptorSetLayout(context.device, &model_layout_info, nullptr, &model_layout) != VK_SUCCESS) return false;

    // Наборы: 1 сцена + объекты + пол + слоты визуализации
    const uint32_t model_set_count = static_cast<uint32_t>(objects.size() + 1 + viz_slots.size());
    const uint32_t total_sets = 1 + model_set_count;
    const VkDescriptorPoolSize pool_sizes[] = {
        {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, total_sets},
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2},
    };
    const VkDescriptorPoolCreateInfo pool_info = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .maxSets = total_sets,
        .poolSizeCount = 2,
        .pPoolSizes = pool_sizes,
    };
    if (vkCreateDescriptorPool(context.device, &pool_info, nullptr, &descriptor_pool) != VK_SUCCESS) return false;

    const VkDescriptorSetAllocateInfo scene_alloc = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = descriptor_pool,
        .descriptorSetCount = 1,
        .pSetLayouts = &scene_layout,
    };
    if (vkAllocateDescriptorSets(context.device, &scene_alloc, &scene_descriptor_set) != VK_SUCCESS) return false;

    std::vector<VkDescriptorSetLayout> model_layouts(model_set_count, model_layout);
    const VkDescriptorSetAllocateInfo model_alloc = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = descriptor_pool,
        .descriptorSetCount = model_set_count,
        .pSetLayouts = model_layouts.data(),
    };
    std::vector<VkDescriptorSet> model_sets(model_set_count);
    if (vkAllocateDescriptorSets(context.device, &model_alloc, model_sets.data()) != VK_SUCCESS) return false;

    for (size_t i = 0; i < objects.size(); ++i) {
        objects[i].model_descriptor_set = model_sets[i];
    }
    floor_obj.descriptor_set = model_sets[objects.size()];
    for (size_t i = 0; i < viz_slots.size(); ++i) {
        viz_slots[i].set = model_sets[objects.size() + 1 + i];
    }

    const VkDescriptorBufferInfo scene_ubo_info = {scene_uniform_buffer.handle, 0, sizeof(SceneUniforms)};
    const VkDescriptorBufferInfo pt_ssbo_info = {point_lights_buffer.handle, 0, VK_WHOLE_SIZE};
    const VkDescriptorBufferInfo sp_ssbo_info = {spot_lights_buffer.handle, 0, VK_WHOLE_SIZE};

    const VkWriteDescriptorSet scene_writes[] = {
        {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, scene_descriptor_set, 0, 0, 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, nullptr, &scene_ubo_info},
        {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, scene_descriptor_set, 1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &pt_ssbo_info},
        {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, scene_descriptor_set, 2, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &sp_ssbo_info},
    };
    vkUpdateDescriptorSets(context.device, 3, scene_writes, 0, nullptr);

    for (size_t i = 0; i < objects.size(); ++i) {
        const VkDescriptorBufferInfo model_ubo_info = {objects[i].model_uniform_buffer.handle, 0, sizeof(ModelUniforms)};
        const VkWriteDescriptorSet model_write = {
            VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, objects[i].model_descriptor_set, 0, 0, 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, nullptr, &model_ubo_info
        };
        vkUpdateDescriptorSets(context.device, 1, &model_write, 0, nullptr);
    }

    point_lights.push_back({glm::vec3(2.0f, 3.0f, 2.0f), 15.0f, glm::vec3(1.0f, 0.5f, 0.2f), 0.0f});
    spot_lights.push_back({glm::vec3(-2.0f, 4.0f, 0.0f), glm::radians(12.5f), glm::vec3(0.2f, 0.5f, 1.0f), glm::radians(17.5f), glm::vec3(0.0f, -1.0f, 0.0f), 15.0f});

    FloorMeshData floor_mesh = generateFloor(floor_obj.size);
    floor_obj.index_count = static_cast<uint32_t>(floor_mesh.indices.size());

    const VkDeviceSize f_v_size = sizeof(floor_mesh.vertices[0]) * floor_mesh.vertices.size();
    if (!vk_buffer::create(f_v_size, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, floor_obj.vertex_buffer)) return false;
    std::memcpy(floor_obj.vertex_buffer.mapped, floor_mesh.vertices.data(), f_v_size);

    const VkDeviceSize f_i_size = sizeof(floor_mesh.indices[0]) * floor_mesh.indices.size();
    if (!vk_buffer::create(f_i_size, VK_BUFFER_USAGE_INDEX_BUFFER_BIT, floor_obj.index_buffer)) return false;
    std::memcpy(floor_obj.index_buffer.mapped, floor_mesh.indices.data(), f_i_size);

    if (!vk_buffer::create(sizeof(ModelUniforms), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, floor_obj.uniform_buffer)) return false;

    const VkDescriptorBufferInfo floor_buf_info = { floor_obj.uniform_buffer.handle, 0, sizeof(ModelUniforms) };
    const VkWriteDescriptorSet floor_write = {
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstSet = floor_obj.descriptor_set,
        .dstBinding = 0,
        .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
        .pBufferInfo = &floor_buf_info,
    };
    vkUpdateDescriptorSets(context.device, 1, &floor_write, 0, nullptr);

    // Слоты визуализации: у каждого свой uniform-буфер
    for (auto& slot : viz_slots) {
        if (!vk_buffer::create(sizeof(ModelUniforms), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, slot.buffer)) return false;
        const VkDescriptorBufferInfo info = { slot.buffer.handle, 0, sizeof(ModelUniforms) };
        const VkWriteDescriptorSet write = {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = slot.set,
            .dstBinding = 0,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            .pBufferInfo = &info,
        };
        vkUpdateDescriptorSets(context.device, 1, &write, 0, nullptr);
    }

    viz_cube_mesh = generateCube(0.3f);
    viz_cube_index_count = static_cast<uint32_t>(viz_cube_mesh.indices.size());
    if (!uploadMesh(viz_cube_mesh, viz_cube_vb, viz_cube_ib)) return false;

    viz_cone_mesh = generateLightConeWire(32);
    viz_cone_index_count = static_cast<uint32_t>(viz_cone_mesh.indices.size());
    if (!uploadMesh(viz_cone_mesh, viz_cone_vb, viz_cone_ib)) return false;

    return true;
}

bool createPipeline() {
    auto& context = graphics::internal::context;
    VkShaderModule vs = loadShaderModule("cone.vert.spv");
    VkShaderModule fs = loadShaderModule("cone.frag.spv");
    if (!vs || !fs) return false;

    const VkPipelineShaderStageCreateInfo stages[] = {
        {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_VERTEX_BIT, vs, "main"},
        {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_FRAGMENT_BIT, fs, "main"},
    };

    auto binding_desc = Vertex::getBindingDescription();
    auto attr_desc = Vertex::getAttributeDescriptions();

    const VkPipelineVertexInputStateCreateInfo vertex_input = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
        .vertexBindingDescriptionCount = 1,
        .pVertexBindingDescriptions = binding_desc.data(),
        .vertexAttributeDescriptionCount = static_cast<uint32_t>(attr_desc.size()),
        .pVertexAttributeDescriptions = attr_desc.data(),
    };

    const VkPipelineInputAssemblyStateCreateInfo input_assembly_tris = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
        .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
    };
    const VkPipelineInputAssemblyStateCreateInfo input_assembly_lines = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
        .topology = VK_PRIMITIVE_TOPOLOGY_LINE_LIST,
    };

    const VkPipelineViewportStateCreateInfo viewport_state = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
        .viewportCount = 1, .scissorCount = 1
    };

    const VkPipelineRasterizationStateCreateInfo rasterization = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
        .polygonMode = VK_POLYGON_MODE_FILL,
        .cullMode = VK_CULL_MODE_NONE,
        .frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE,
        .lineWidth = 1.0f,
    };

    const VkPipelineMultisampleStateCreateInfo multisample = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
        .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT
    };

    const VkPipelineDepthStencilStateCreateInfo depth_stencil = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
        .depthTestEnable = VK_TRUE,
        .depthWriteEnable = VK_TRUE,
        .depthCompareOp = VK_COMPARE_OP_LESS,
    };

    const VkPipelineColorBlendAttachmentState color_blend_attachment = {
        .colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT,
    };

    const VkPipelineColorBlendStateCreateInfo color_blend = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
        .attachmentCount = 1,
        .pAttachments = &color_blend_attachment,
    };

    const VkDynamicState dynamic_states[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    const VkPipelineDynamicStateCreateInfo dynamic_state = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
        .dynamicStateCount = 2,
        .pDynamicStates = dynamic_states,
    };

    const VkPushConstantRange push_constant_range = {
        .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
        .offset = 0,
        .size = sizeof(PushConstants),
    };

    const VkDescriptorSetLayout set_layouts[] = {scene_layout, model_layout};
    const VkPipelineLayoutCreateInfo layout_info = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 2,
        .pSetLayouts = set_layouts,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &push_constant_range,
    };

    if (vkCreatePipelineLayout(context.device, &layout_info, nullptr, &pipeline_layout) != VK_SUCCESS) return false;

    auto build = [&](const VkPipelineInputAssemblyStateCreateInfo& input_assembly, VkPipeline& out) {
        const VkGraphicsPipelineCreateInfo pipeline_info = {
            .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
            .stageCount = 2,
            .pStages = stages,
            .pVertexInputState = &vertex_input,
            .pInputAssemblyState = &input_assembly,
            .pViewportState = &viewport_state,
            .pRasterizationState = &rasterization,
            .pMultisampleState = &multisample,
            .pDepthStencilState = &depth_stencil,
            .pColorBlendState = &color_blend,
            .pDynamicState = &dynamic_state,
            .layout = pipeline_layout,
            .renderPass = context.render_pass,
            .subpass = 0,
        };
        return vkCreateGraphicsPipelines(context.device, VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &out) == VK_SUCCESS;
    };

    if (!build(input_assembly_tris, pipeline)) return false;
    if (!build(input_assembly_lines, line_pipeline)) return false;

    vkDestroyShaderModule(context.device, vs, nullptr);
    vkDestroyShaderModule(context.device, fs, nullptr);
    return true;
}

} // namespace

bool initialize() {
	if (!createBuffersAndDescriptors()) return false;
	if (!createPipeline()) return false;
	return true;
}

void shutdown() {
    auto& context = graphics::internal::context;
    vkQueueWaitIdle(context.graphics_queue);

    vkDestroyPipeline(context.device, pipeline, nullptr);
    vkDestroyPipeline(context.device, line_pipeline, nullptr);
    vkDestroyPipelineLayout(context.device, pipeline_layout, nullptr);
    vkDestroyDescriptorSetLayout(context.device, scene_layout, nullptr);
    vkDestroyDescriptorSetLayout(context.device, model_layout, nullptr);
    vkDestroyDescriptorPool(context.device, descriptor_pool, nullptr);

    vk_buffer::destroy(vertex_buffer);
    vk_buffer::destroy(index_buffer);
    vk_buffer::destroy(scene_uniform_buffer);
    vk_buffer::destroy(point_lights_buffer);
    vk_buffer::destroy(spot_lights_buffer);

    for (auto& obj : objects) {
        vk_buffer::destroy(obj.model_uniform_buffer);
    }

    vk_buffer::destroy(floor_obj.vertex_buffer);
    vk_buffer::destroy(floor_obj.index_buffer);
    vk_buffer::destroy(floor_obj.uniform_buffer);

    for (auto& slot : viz_slots) {
        vk_buffer::destroy(slot.buffer);
    }

    vk_buffer::destroy(viz_cube_vb);
    vk_buffer::destroy(viz_cube_ib);
    vk_buffer::destroy(viz_cone_vb);
    vk_buffer::destroy(viz_cone_ib);
}

void update(double time) {
    const double dt = (previous_time >= 0.0) ? (time - previous_time) : 0.0;
    previous_time = time;

    ImGui::Begin("Lab 2: Camera & Lighting");

    if (ImGui::CollapsingHeader("Camera", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::SliderFloat3("Position", camera.pos, -10.0f, 10.0f);
        ImGui::SliderFloat("Yaw", &camera.yaw, -180.0f, 180.0f);
        ImGui::SliderFloat("Pitch", &camera.pitch, -89.0f, 89.0f);
        ImGui::SeparatorText("Projection");
        ImGui::RadioButton("Perspective", &projection_type, 0);
        ImGui::SameLine();
        ImGui::RadioButton("Orthographic", &projection_type, 1);
        if (projection_type == 0) {
            ImGui::SliderFloat("FOV", &camera.fov, 10.0f, 120.0f);
        }
    }

    if (ImGui::CollapsingHeader("Directional Light", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::SliderFloat3("Direction", &directional_light_dir.x, -1.0f, 1.0f);
        // защита от нулевого вектора (normalize(0) даёт NaN и чёрный экран)
        if (glm::length(directional_light_dir) < 1e-4f) directional_light_dir = glm::vec3(0.0f, -1.0f, 0.0f);
        directional_light_dir = glm::normalize(directional_light_dir);
        ImGui::ColorEdit3("Color", &directional_light_col.x);
        ImGui::ColorEdit3("Ambient", &ambient_light.x);
    }

    if (ImGui::CollapsingHeader("Material", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::ColorEdit3("Albedo", &material_albedo.x);
        ImGui::ColorEdit3("Specular", &material_specular.x);
        ImGui::SliderFloat("Shininess", &material_shininess, 1.0f, 256.0f);
    }

    if (ImGui::CollapsingHeader("Point Lights")) {
        ImGui::BeginDisabled(point_lights.size() >= kMaxLights);
        if (ImGui::Button("Add Point Light")) {
            point_lights.push_back({glm::vec3(0.0f, 3.0f, 0.0f), 15.0f, glm::vec3(1.0f, 1.0f, 1.0f), 0.0f});
        }
        ImGui::EndDisabled();
        for (size_t i = 0; i < point_lights.size(); i++) {
            ImGui::PushID(static_cast<int>(i));
            if (ImGui::TreeNode(("Point Light " + std::to_string(i)).c_str())) {
                ImGui::SliderFloat3("Position", &point_lights[i].position.x, -8.0f, 8.0f);
                ImGui::SliderFloat("Intensity", &point_lights[i].intensity, 0.0f, 50.0f);
                ImGui::ColorEdit3("Color", &point_lights[i].color.x);
                if (ImGui::Button("Remove")) {
                    point_lights.erase(point_lights.begin() + i);
                    ImGui::TreePop();
                    ImGui::PopID();
                    break;
                }
                ImGui::TreePop();
            }
            ImGui::PopID();
        }
    }

    if (ImGui::CollapsingHeader("Spot Lights")) {
        ImGui::BeginDisabled(spot_lights.size() >= kMaxLights);
        if (ImGui::Button("Add Spot Light")) {
            spot_lights.push_back({glm::vec3(0.0f, 3.0f, 0.0f), glm::radians(12.5f), glm::vec3(1.0f, 1.0f, 1.0f), glm::radians(17.5f), glm::vec3(0.0f, -1.0f, 0.0f), 15.0f});
        }
        ImGui::EndDisabled();
        for (size_t i = 0; i < spot_lights.size(); i++) {
            ImGui::PushID(static_cast<int>(i));
            if (ImGui::TreeNode(("Spot Light " + std::to_string(i)).c_str())) {
                auto& s = spot_lights[i];
                ImGui::SliderFloat3("Position", &s.position.x, -8.0f, 8.0f);
                ImGui::SliderFloat3("Direction", &s.direction.x, -1.0f, 1.0f);
                if (glm::length(s.direction) < 1e-4f) s.direction = glm::vec3(0.0f, -1.0f, 0.0f);
                s.direction = glm::normalize(s.direction);
                ImGui::SliderFloat("Inner Angle (rad)", &s.inner_angle, 0.0f, glm::radians(45.0f));
                ImGui::SliderFloat("Outer Angle (rad)", &s.outer_angle, 0.0f, glm::radians(60.0f));
                // внутренний угол не должен превышать внешний, иначе край не будет мягким
                s.inner_angle = std::min(s.inner_angle, s.outer_angle);
                ImGui::SliderFloat("Intensity", &s.intensity, 0.0f, 50.0f);
                ImGui::ColorEdit3("Color", &s.color.x);
                if (ImGui::Button("Remove")) {
                    spot_lights.erase(spot_lights.begin() + i);
                    ImGui::TreePop();
                    ImGui::PopID();
                    break;
                }
                ImGui::TreePop();
            }
            ImGui::PopID();
        }
    }

    ImGui::Checkbox("Show Light Visualization", &show_light_visualization);

    for (size_t i = 0; i < objects.size(); ++i) {
        ImGui::PushID(static_cast<int>(i));
        if (ImGui::CollapsingHeader(("Object " + std::to_string(i + 1)).c_str())) {
            ImGui::Checkbox("Animate", &objects[i].trajectory);
            if (objects[i].trajectory) {
                if (ImGui::Button(objects[i].playing ? "Pause" : "Play")) objects[i].playing = !objects[i].playing;
                ImGui::SliderFloat("Speed", &objects[i].speed, 0.1f, 3.0f);
                ImGui::SliderFloat("Radius", &objects[i].radius, 1.0f, 4.0f);
                ImGui::SliderFloat("Bob height", &objects[i].bob_amplitude, 0.0f, 1.0f);
            } else {
                ImGui::SliderFloat3("Position", objects[i].position, -5.0f, 5.0f);
            }
            ImGui::SliderFloat3("Rotation", objects[i].rotation, -180.0f, 180.0f);
            ImGui::SliderFloat3("Scale", objects[i].scale, 0.2f, 3.0f);
            ImGui::ColorEdit3("Tint", objects[i].tint);
        }
        ImGui::PopID();
    }

    ImGui::End();

    for (size_t i = 0; i < objects.size(); ++i) {
        auto& obj = objects[i];
        if (obj.trajectory && obj.playing) {
            obj.phase += static_cast<float>(dt) * obj.speed * 0.6f;
        }
    }
}

void render(const graphics::internal::FrameData& fd) {
    auto& context = graphics::internal::context;
    float aspect = (context.swapchain_extent.height > 0)
        ? static_cast<float>(context.swapchain_extent.width) / static_cast<float>(context.swapchain_extent.height)
        : 1.0f;

    glm::vec3 cam_pos = glm::make_vec3(camera.pos);
    glm::vec3 front;
    front.x = cos(glm::radians(camera.yaw)) * cos(glm::radians(camera.pitch));
    front.y = sin(glm::radians(camera.pitch));
    front.z = sin(glm::radians(camera.yaw)) * cos(glm::radians(camera.pitch));
    front = glm::normalize(front);

    glm::mat4 view = glm::lookAt(cam_pos, cam_pos + front, glm::vec3(0.0f, 1.0f, 0.0f));
    glm::mat4 proj;
    if (projection_type == 0) {
        proj = glm::perspective(glm::radians(camera.fov), aspect, 0.1f, 100.0f);
    } else {
        constexpr float extent = 5.0f;
        proj = glm::ortho(-extent * aspect, extent * aspect, -extent, extent, -100.0f, 100.0f);
    }
    proj[1][1] *= -1.0f;

    const size_t point_count = std::min(point_lights.size(), kMaxLights);
    const size_t spot_count = std::min(spot_lights.size(), kMaxLights);

    SceneUniforms scene_ubo{};
    scene_ubo.view_projection = proj * view;
    scene_ubo.camera_position = cam_pos;
    scene_ubo.directional_light_dir = directional_light_dir;
    scene_ubo.directional_light_col = directional_light_col;
    scene_ubo.ambient_light = ambient_light;
    scene_ubo.point_light_count = static_cast<uint32_t>(point_count);
    scene_ubo.spot_light_count = static_cast<uint32_t>(spot_count);
    std::memcpy(scene_uniform_buffer.mapped, &scene_ubo, sizeof(SceneUniforms));

    if (point_count > 0) {
        std::memcpy(point_lights_buffer.mapped, point_lights.data(), point_count * sizeof(PointLight));
    }
    if (spot_count > 0) {
        std::memcpy(spot_lights_buffer.mapped, spot_lights.data(), spot_count * sizeof(SpotLight));
    }

    const VkCommandBufferBeginInfo begin_info = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    vkBeginCommandBuffer(fd.command_buffer, &begin_info);
    const VkClearValue clear_values[] = {{.color = {{0.05f, 0.05f, 0.08f, 1.0f}}}, {.depthStencil = {1.0f, 0}}};
    const VkRenderPassBeginInfo render_pass_begin = {
        .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
        .renderPass = context.render_pass, .framebuffer = fd.framebuffer,
        .renderArea = {.extent = context.swapchain_extent}, .clearValueCount = 2, .pClearValues = clear_values,
    };

    vkCmdBeginRenderPass(fd.command_buffer, &render_pass_begin, VK_SUBPASS_CONTENTS_INLINE);
    vkCmdBindPipeline(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
    const VkViewport viewport = { 0.0f, 0.0f, float(context.swapchain_extent.width), float(context.swapchain_extent.height), 0.0f, 1.0f };
    vkCmdSetViewport(fd.command_buffer, 0, 1, &viewport);

    const VkRect2D scissor = { .offset = {0, 0}, .extent = context.swapchain_extent };
    vkCmdSetScissor(fd.command_buffer, 0, 1, &scissor);
    const VkDeviceSize offsets[] = {0};

    vkCmdBindDescriptorSets(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout, 0, 1, &scene_descriptor_set, 0, nullptr);

    // Пол
    if (floor_obj.show_floor) {
        ModelUniforms floor_ubo{};
        floor_ubo.model = glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 0.0f, 0.0f));
        floor_ubo.albedo = glm::make_vec3(floor_obj.tint);
        floor_ubo.specular = glm::vec3(0.2f, 0.2f, 0.2f);
        floor_ubo.shininess = 16.0f;
        std::memcpy(floor_obj.uniform_buffer.mapped, &floor_ubo, sizeof(ModelUniforms));

        const VkBuffer floor_vb[] = {floor_obj.vertex_buffer.handle};
        vkCmdBindVertexBuffers(fd.command_buffer, 0, 1, floor_vb, offsets);
        vkCmdBindIndexBuffer(fd.command_buffer, floor_obj.index_buffer.handle, 0, VK_INDEX_TYPE_UINT32);
        vkCmdBindDescriptorSets(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout, 1, 1, &floor_obj.descriptor_set, 0, nullptr);

        PushConstants pc{glm::vec4(1.0f, 1.0f, 1.0f, 1.0f), 0.0f, {0.0f, 0.0f, 0.0f}};
        vkCmdPushConstants(fd.command_buffer, pipeline_layout, VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(PushConstants), &pc);
        vkCmdDrawIndexed(fd.command_buffer, floor_obj.index_count, 1, 0, 0, 0);
    }

    const VkBuffer main_vb[] = {vertex_buffer.handle};
    vkCmdBindVertexBuffers(fd.command_buffer, 0, 1, main_vb, offsets);
    vkCmdBindIndexBuffer(fd.command_buffer, index_buffer.handle, 0, VK_INDEX_TYPE_UINT32);

    for (size_t i = 0; i < objects.size(); ++i) {
        auto& obj = objects[i];
        ModelUniforms model_ubo{};
        // albedo материала (общий) * цвет объекта; радужный цвет вершин умножается в шейдере
        model_ubo.albedo = material_albedo * glm::make_vec3(obj.tint);
        model_ubo.specular = material_specular;
        model_ubo.shininess = material_shininess;

        const glm::vec3 scale = glm::make_vec3(obj.scale);
        glm::vec3 pos = glm::make_vec3(obj.position);

        if (obj.trajectory) {
            const float t = obj.phase + static_cast<float>(i) * 2.094f;
            const float a = obj.radius;
            const float denom = 1.0f + std::sin(t) * std::sin(t);
            const float half_height = 0.5f * kConeHeight * scale.y;
            const float lift = obj.bob_amplitude * 0.5f * (1.0f + std::sin(2.0f * t));
            pos = {a * std::cos(t) / denom, half_height + lift, a * std::sin(t) * std::cos(t) / denom};
        }

        glm::mat4 model = glm::translate(glm::mat4(1.0f), pos);
        model = glm::rotate(model, glm::radians(obj.rotation[0]), glm::vec3(1.0f, 0.0f, 0.0f));
        model = glm::rotate(model, glm::radians(obj.rotation[1]), glm::vec3(0.0f, 1.0f, 0.0f));
        model = glm::rotate(model, glm::radians(obj.rotation[2]), glm::vec3(0.0f, 0.0f, 1.0f));
        model_ubo.model = glm::scale(model, scale);

        std::memcpy(obj.model_uniform_buffer.mapped, &model_ubo, sizeof(ModelUniforms));
        vkCmdBindDescriptorSets(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout, 1, 1, &obj.model_descriptor_set, 0, nullptr);

        PushConstants pc{glm::vec4(obj.tint[0], obj.tint[1], obj.tint[2], 1.0f), 0.0f, {0.0f, 0.0f, 0.0f}};
        vkCmdPushConstants(fd.command_buffer, pipeline_layout, VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(PushConstants), &pc);
        vkCmdDrawIndexed(fd.command_buffer, current_index_count, 1, 0, 0, 0);
    }

    // Визуализация источников (постоянный цвет, без Блинн–Фонга)
    if (show_light_visualization) {
        size_t slot = 0;
        auto draw_viz = [&](const glm::mat4& model, const glm::vec4& color, uint32_t index_count) {
            if (slot >= viz_slots.size()) return;
            ModelUniforms ubo{};
            ubo.model = model;
            std::memcpy(viz_slots[slot].buffer.mapped, &ubo, sizeof(ModelUniforms));
            vkCmdBindDescriptorSets(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout, 1, 1, &viz_slots[slot].set, 0, nullptr);
            PushConstants pc{color, 1.0f, {0.0f, 0.0f, 0.0f}};
            vkCmdPushConstants(fd.command_buffer, pipeline_layout, VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(PushConstants), &pc);
            vkCmdDrawIndexed(fd.command_buffer, index_count, 1, 0, 0, 0);
            ++slot;
        };

        // Кубы цвета источника: точечные и прожекторные
        const VkBuffer cube_vb_arr[] = {viz_cube_vb.handle};
        vkCmdBindVertexBuffers(fd.command_buffer, 0, 1, cube_vb_arr, offsets);
        vkCmdBindIndexBuffer(fd.command_buffer, viz_cube_ib.handle, 0, VK_INDEX_TYPE_UINT32);

        for (size_t i = 0; i < point_count; ++i) {
            const auto& light = point_lights[i];
            draw_viz(glm::translate(glm::mat4(1.0f), light.position), glm::vec4(light.color, 1.0f), viz_cube_index_count);
        }
        for (size_t i = 0; i < spot_count; ++i) {
            const auto& light = spot_lights[i];
            draw_viz(glm::translate(glm::mat4(1.0f), light.position), glm::vec4(light.color, 1.0f), viz_cube_index_count);
        }

        vkCmdBindPipeline(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, line_pipeline);
        const VkBuffer cone_vb_arr[] = {viz_cone_vb.handle};
        vkCmdBindVertexBuffers(fd.command_buffer, 0, 1, cone_vb_arr, offsets);
        vkCmdBindIndexBuffer(fd.command_buffer, viz_cone_ib.handle, 0, VK_INDEX_TYPE_UINT32);

        for (size_t i = 0; i < spot_count; ++i) {
            const auto& light = spot_lights[i];
            const glm::vec3 dir = glm::normalize(light.direction);
            const glm::vec3 axis_y = -dir;
            const glm::vec3 helper = std::abs(axis_y.y) > 0.99f ? glm::vec3(1.0f, 0.0f, 0.0f) : glm::vec3(0.0f, 1.0f, 0.0f);
            const glm::vec3 axis_x = glm::normalize(glm::cross(helper, axis_y));
            const glm::vec3 axis_z = glm::cross(axis_x, axis_y);
            const glm::mat4 rot(glm::vec4(axis_x, 0.0f), glm::vec4(axis_y, 0.0f), glm::vec4(axis_z, 0.0f), glm::vec4(0.0f, 0.0f, 0.0f, 1.0f));

            const float base_radius = kSpotVizLength * std::tan(std::min(light.outer_angle, glm::radians(80.0f)));
            const glm::mat4 model = glm::translate(glm::mat4(1.0f), light.position) * rot
                                  * glm::scale(glm::mat4(1.0f), glm::vec3(base_radius, kSpotVizLength, base_radius));
            draw_viz(model, glm::vec4(1.0f, 1.0f, 1.0f, 1.0f), viz_cone_index_count);
        }
    }

    vkCmdEndRenderPass(fd.command_buffer);
    vkEndCommandBuffer(fd.command_buffer);
}

} // namespace application