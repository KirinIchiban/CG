#include "vk_texture.hpp"
#include "graphics_internal.hpp"
#include "vk_buffer.hpp"

#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_PNG
#define STBI_ONLY_JPEG
#define STBI_ONLY_BMP
#define STBI_ONLY_TGA
#define STBI_ONLY_GIF
#include <stb_image.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <functional>
#include <iostream>
#include <vector>

namespace vk_texture {
namespace {

auto& ctx() { return graphics::internal::context; }

bool one_time_submit(const std::function<void(VkCommandBuffer)>& record) {
    const VkCommandPoolCreateInfo pool_info = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT,
        .queueFamilyIndex = ctx().graphics_queue_index,
    };
    VkCommandPool pool = VK_NULL_HANDLE;
    if (vkCreateCommandPool(ctx().device, &pool_info, nullptr, &pool) != VK_SUCCESS) return false;

    const VkCommandBufferAllocateInfo alloc = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = pool, .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 1,
    };
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    vkAllocateCommandBuffers(ctx().device, &alloc, &cmd);

    const VkCommandBufferBeginInfo begin = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    };
    vkBeginCommandBuffer(cmd, &begin);
    record(cmd);
    vkEndCommandBuffer(cmd);

    const VkSubmitInfo submit = {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &cmd,
    };
    const bool ok = vkQueueSubmit(ctx().graphics_queue, 1, &submit, VK_NULL_HANDLE) == VK_SUCCESS;
    vkQueueWaitIdle(ctx().graphics_queue);
    vkDestroyCommandPool(ctx().device, pool, nullptr);
    return ok;
}

void transition(VkCommandBuffer cmd, VkImage image, uint32_t base_mip, uint32_t mip_count,
                VkImageLayout old_layout, VkImageLayout new_layout,
                VkAccessFlags src_access, VkAccessFlags dst_access,
                VkPipelineStageFlags src_stage, VkPipelineStageFlags dst_stage) {
    const VkImageMemoryBarrier barrier = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        .srcAccessMask = src_access,
        .dstAccessMask = dst_access,
        .oldLayout = old_layout,
        .newLayout = new_layout,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = image,
        .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, base_mip, mip_count, 0, 1 },
    };
    vkCmdPipelineBarrier(cmd, src_stage, dst_stage, 0, 0, nullptr, 0, nullptr, 1, &barrier);
}

// Мип-цепочка через последовательный vkCmdBlitImage с LINEAR-фильтром
void generate_mips(VkCommandBuffer cmd, VkImage image, uint32_t w, uint32_t h, uint32_t levels) {
    int32_t mw = static_cast<int32_t>(w);
    int32_t mh = static_cast<int32_t>(h);

    for (uint32_t i = 1; i < levels; ++i) {
        transition(cmd, image, i - 1, 1,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);

        const int32_t nw = std::max(1, mw / 2);
        const int32_t nh = std::max(1, mh / 2);

        const VkImageBlit blit = {
            .srcSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, i - 1, 0, 1 },
            .srcOffsets = { {0,0,0}, {mw, mh, 1} },
            .dstSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, i, 0, 1 },
            .dstOffsets = { {0,0,0}, {nw, nh, 1} },
        };
        vkCmdBlitImage(cmd,
            image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            1, &blit, VK_FILTER_LINEAR);

        // mip i-1 готов -> сразу отдаём шейдеру
        transition(cmd, image, i - 1, 1,
            VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            VK_ACCESS_TRANSFER_READ_BIT, VK_ACCESS_SHADER_READ_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);

        mw = nw; mh = nh;
    }
    // последний уровень
    transition(cmd, image, levels - 1, 1,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);
}

bool build(const uint8_t* rgba, uint32_t w, uint32_t h, Texture& out, const CreateInfo& info) {
    const VkDeviceSize size = VkDeviceSize(w) * h * 4;

    vk_buffer::Buffer staging{};
    if (!vk_buffer::create(size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, staging)) return false;
    std::memcpy(staging.mapped, rgba, static_cast<size_t>(size));

    uint32_t levels = 1;
    if (info.generate_mips) {
        levels = static_cast<uint32_t>(std::floor(std::log2(std::max(w, h)))) + 1;
    }

    const VkFormat format = info.srgb ? VK_FORMAT_R8G8B8A8_SRGB : VK_FORMAT_R8G8B8A8_UNORM;

    const VkImageCreateInfo image_info = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
        .imageType = VK_IMAGE_TYPE_2D,
        .format = format,
        .extent = { w, h, 1 },
        .mipLevels = levels,
        .arrayLayers = 1,
        .samples = VK_SAMPLE_COUNT_1_BIT,
        .tiling = VK_IMAGE_TILING_OPTIMAL,          // проприетарная раскладка GPU
        .usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                 VK_IMAGE_USAGE_TRANSFER_SRC_BIT |  // нужен для blit при генерации мипов
                 VK_IMAGE_USAGE_SAMPLED_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
    };
    const VmaAllocationCreateInfo alloc_info = {
        .flags = 0,
        .usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,   // текстура живёт в VRAM
    };
    if (vmaCreateImage(ctx().allocator, &image_info, &alloc_info,
                       &out.image, &out.allocation, nullptr) != VK_SUCCESS) {
        vk_buffer::destroy(staging);
        std::cerr << "Failed to create VkImage\n";
        return false;
    }
    out.width = w; out.height = h; out.mip_levels = levels;

    const VkImage image = out.image;
    const VkBuffer src  = staging.handle;
    const bool ok = one_time_submit([&](VkCommandBuffer cmd) {
        transition(cmd, image, 0, levels,
            VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            0, VK_ACCESS_TRANSFER_WRITE_BIT,
            VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);

        const VkBufferImageCopy region = {
            .bufferOffset = 0, .bufferRowLength = 0, .bufferImageHeight = 0,
            .imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 },
            .imageOffset = {0,0,0},
            .imageExtent = { w, h, 1 },
        };
        vkCmdCopyBufferToImage(cmd, src, image,
                               VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

        if (levels > 1) {
            generate_mips(cmd, image, w, h, levels);
        } else {
            transition(cmd, image, 0, 1,
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);
        }
    });
    vk_buffer::destroy(staging);
    if (!ok) { destroy(out); return false; }

    const VkImageViewCreateInfo view_info = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
        .image = out.image,
        .viewType = VK_IMAGE_VIEW_TYPE_2D,
        .format = format,
        .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, levels, 0, 1 },
    };
    if (vkCreateImageView(ctx().device, &view_info, nullptr, &out.view) != VK_SUCCESS) {
        destroy(out); return false;
    }

    VkPhysicalDeviceProperties props{};
    vkGetPhysicalDeviceProperties(ctx().physical_device, &props);

    const VkSamplerCreateInfo sampler_info = {
        .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
        .magFilter = VK_FILTER_LINEAR,                 // билинейная при увеличении
        .minFilter = VK_FILTER_LINEAR,
        .mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR,   // + мипы = трилинейная
        .addressModeU = info.address,
        .addressModeV = info.address,
        .addressModeW = info.address,
        .mipLodBias = 0.0f,
        .anisotropyEnable = VK_TRUE,                   // против "мыла" под острым углом
        .maxAnisotropy = std::min(16.0f, props.limits.maxSamplerAnisotropy),
        .compareEnable = VK_FALSE,
        .minLod = 0.0f,
        .maxLod = static_cast<float>(levels),
        .borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK,
        .unnormalizedCoordinates = VK_FALSE,
    };
    if (vkCreateSampler(ctx().device, &sampler_info, nullptr, &out.sampler) != VK_SUCCESS) {
        destroy(out); return false;
    }
    return true;
}

} // namespace

bool load_from_file(const std::string& path, Texture& out, const CreateInfo& info) {
    int w = 0, h = 0, channels = 0;
    stbi_uc* pixels = stbi_load(path.c_str(), &w, &h, &channels, STBI_rgb_alpha);
    if (!pixels) {
        std::cerr << "stbi_load failed: " << path << " (" << stbi_failure_reason() << ")\n";
        return false;
    }
    const bool ok = build(pixels, static_cast<uint32_t>(w), static_cast<uint32_t>(h), out, info);
    stbi_image_free(pixels);
    if (ok) std::cout << "Loaded texture: " << path << " (" << w << "x" << h << ")\n";
    return ok;
}

bool create_from_pixels(const uint8_t* rgba, uint32_t w, uint32_t h,
                        Texture& out, const CreateInfo& info) {
    return build(rgba, w, h, out, info);
}

void destroy(Texture& t) {
    auto& c = ctx();
    if (t.sampler) vkDestroySampler(c.device, t.sampler, nullptr);
    if (t.view)    vkDestroyImageView(c.device, t.view, nullptr);
    if (t.image)   vmaDestroyImage(c.allocator, t.image, t.allocation);
    t = {};
}

} // namespace vk_texture