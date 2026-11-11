#pragma once
#include <string>
#include <cstdint>
#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>

namespace vk_texture {

struct Texture {
    VkImage       image      = VK_NULL_HANDLE;
    VmaAllocation allocation = VK_NULL_HANDLE;
    VkImageView   view       = VK_NULL_HANDLE;
    VkSampler     sampler    = VK_NULL_HANDLE;
    uint32_t      width      = 0;
    uint32_t      height     = 0;
    uint32_t      mip_levels = 1;
};

struct CreateInfo {
    bool srgb            = true;   // альбедо/emissive = true, маски = false
    bool generate_mips   = true;
    VkSamplerAddressMode address = VK_SAMPLER_ADDRESS_MODE_REPEAT;
};

bool load_from_file(const std::string& path, Texture& out, const CreateInfo& info = {});
bool create_from_pixels(const uint8_t* rgba, uint32_t w, uint32_t h,
                        Texture& out, const CreateInfo& info = {});
void destroy(Texture& t);

} // namespace vk_texture