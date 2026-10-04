#pragma once
#include <vector>
#include <glm/glm.hpp>
#include <vulkan/vulkan.h>

// Структура вершины, совпадающая с layout= в вершинном шейдере
struct Vertex {
    glm::vec3 position;
    glm::vec3 color;

    static std::vector<VkVertexInputBindingDescription> getBindingDescription() {
        std::vector<VkVertexInputBindingDescription> bindingDescriptions(1);
        bindingDescriptions[0].binding = 0;
        bindingDescriptions[0].stride = sizeof(Vertex);
        bindingDescriptions[0].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
        return bindingDescriptions;
    }

    static std::vector<VkVertexInputAttributeDescription> getAttributeDescriptions() {
        std::vector<VkVertexInputAttributeDescription> attributeDescriptions(2);
        
        //позиция location = 0
        attributeDescriptions[0].binding = 0;
        attributeDescriptions[0].location = 0;
        attributeDescriptions[0].format = VK_FORMAT_R32G32B32_SFLOAT;
        attributeDescriptions[0].offset = offsetof(Vertex, position);

        //цвет location = 1
        attributeDescriptions[1].binding = 0;
        attributeDescriptions[1].location = 1;
        attributeDescriptions[1].format = VK_FORMAT_R32G32B32_SFLOAT;
        attributeDescriptions[1].offset = offsetof(Vertex, color);

        return attributeDescriptions;
    }
};

inline void generateCone(std::vector<Vertex>& vertices, std::vector<uint32_t>& indices, int segments = 32) {
    vertices.clear();
    indices.clear();

    vertices.push_back({ {0.0f, 0.5f, 0.0f}, {1.0f, 1.0f, 1.0f} });
    vertices.push_back({ {0.0f, -0.5f, 0.0f}, {0.2f, 0.2f, 0.2f} });
    int baseCenterIndex = 1;
    float radius = 0.5f;
    float halfHeight = 0.5f;

    for (int i = 0; i < segments; ++i) {
        float theta = 2.0f * 3.1415926535f * static_cast<float>(i) / static_cast<float>(segments);
        float x = radius * cos(theta);
        float z = radius * sin(theta);
        
        glm::vec3 col = {
            sin(theta) * 0.5f + 0.5f,
            cos(theta) * 0.5f + 0.5f,
            0.8f
        };
        vertices.push_back({ {x, -halfHeight, z}, col });
    }

    for (int i = 0; i < segments; ++i) {
        int current = 2 + i;
        int next = 2 + (i + 1) % segments;

        indices.push_back(0);
        indices.push_back(next);
        indices.push_back(current);
        indices.push_back(baseCenterIndex);
        indices.push_back(current);
        indices.push_back(next);
    }
}