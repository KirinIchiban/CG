#pragma once
#include <vector>
#include <glm/glm.hpp>
#include <vulkan/vulkan.h>

// Структура вершины, совпадающая с layout'ами в вершинном шейдере
struct Vertex {
    glm::vec3 position;
    glm::vec3 color;
};

// Функция для процедурной генерации конуса
inline void generateCone(std::vector<Vertex>& vertices, std::vector<uint32_t>& indices, int segments = 32) {}