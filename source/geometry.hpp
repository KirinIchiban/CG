#pragma once
#include <vector>
#include <glm/glm.hpp>
#include <vulkan/vulkan.h>

// Структура вершины, совпадающая с layout в вершинном шейдере
struct Vertex {
    glm::vec3 position;
    glm::vec3 normal;  // Добавлено
    glm::vec3 color;

    static std::vector<VkVertexInputBindingDescription> getBindingDescription() {
        std::vector<VkVertexInputBindingDescription> bindingDescriptions(1);
        bindingDescriptions[0].binding = 0;
        bindingDescriptions[0].stride = sizeof(Vertex);
        bindingDescriptions[0].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
        return bindingDescriptions;
    }

    static std::vector<VkVertexInputAttributeDescription> getAttributeDescriptions() {
        // ВАЖНО: Теперь атрибутов 3, а не 2!
        std::vector<VkVertexInputAttributeDescription> attributeDescriptions(3);
        
        // позиция location = 0
        attributeDescriptions[0].binding = 0;
        attributeDescriptions[0].location = 0;
        attributeDescriptions[0].format = VK_FORMAT_R32G32B32_SFLOAT;
        attributeDescriptions[0].offset = offsetof(Vertex, position);

        // нормаль location = 1
        attributeDescriptions[1].binding = 0;
        attributeDescriptions[1].location = 1;
        attributeDescriptions[1].format = VK_FORMAT_R32G32B32_SFLOAT;
        attributeDescriptions[1].offset = offsetof(Vertex, normal);

        // цвет location = 2
        attributeDescriptions[2].binding = 0;
        attributeDescriptions[2].location = 2;
        attributeDescriptions[2].format = VK_FORMAT_R32G32B32_SFLOAT;
        attributeDescriptions[2].offset = offsetof(Vertex, color);

        return attributeDescriptions;
    }
};