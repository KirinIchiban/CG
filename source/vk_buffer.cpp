#include "vk_buffer.hpp"
#include "graphics_internal.hpp"
#include <cstring>
#include <functional>
#include <iostream>

namespace vk_buffer {
namespace {

// Одноразовый command buffer: записали -> отправили -> дождались -> освободили.
bool one_time_submit(const std::function<void(VkCommandBuffer)>& record) {
    auto& ctx = graphics::internal::context;

    const VkCommandPoolCreateInfo pool_info = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT,
        .queueFamilyIndex = ctx.graphics_queue_index,
    };
    VkCommandPool pool = VK_NULL_HANDLE;
    if (vkCreateCommandPool(ctx.device, &pool_info, nullptr, &pool) != VK_SUCCESS) return false;

    const VkCommandBufferAllocateInfo alloc = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = pool,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1,
    };
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    if (vkAllocateCommandBuffers(ctx.device, &alloc, &cmd) != VK_SUCCESS) {
        vkDestroyCommandPool(ctx.device, pool, nullptr);
        return false;
    }

    const VkCommandBufferBeginInfo begin = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    };
    vkBeginCommandBuffer(cmd, &begin);
    record(cmd);
    vkEndCommandBuffer(cmd);

    const VkSubmitInfo submit = {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .commandBufferCount = 1,
        .pCommandBuffers = &cmd,
    };
    const bool ok = vkQueueSubmit(ctx.graphics_queue, 1, &submit, VK_NULL_HANDLE) == VK_SUCCESS;
    vkQueueWaitIdle(ctx.graphics_queue);   // staging нельзя удалять до конца копирования

    vkDestroyCommandPool(ctx.device, pool, nullptr);
    return ok;
}

} // namespace

bool create(VkDeviceSize size, VkBufferUsageFlags usage, Buffer& out_buffer) {
    auto& context = graphics::internal::context;

    const VkBufferCreateInfo create_info = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = size,
        .usage = usage,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
    };
    const VmaAllocationCreateInfo allocation_info = {
        .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                 VMA_ALLOCATION_CREATE_MAPPED_BIT,
        .usage = VMA_MEMORY_USAGE_AUTO,
    };
    VmaAllocationInfo result_info{};
    if (vmaCreateBuffer(context.allocator, &create_info, &allocation_info,
                        &out_buffer.handle, &out_buffer.allocation, &result_info) != VK_SUCCESS) {
        std::cerr << "Failed to create Vulkan buffer via VMA\n";
        return false;
    }
    out_buffer.mapped = result_info.pMappedData;
    return out_buffer.mapped != nullptr;
}

bool create_device_local(VkDeviceSize size, VkBufferUsageFlags usage, Buffer& out_buffer) {
    auto& context = graphics::internal::context;

    const VkBufferCreateInfo create_info = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = size,
        .usage = usage | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
    };
    const VmaAllocationCreateInfo allocation_info = {
        .flags = 0,
        .usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
    };
    if (vmaCreateBuffer(context.allocator, &create_info, &allocation_info,
                        &out_buffer.handle, &out_buffer.allocation, nullptr) != VK_SUCCESS) {
        std::cerr << "Failed to create device-local Vulkan buffer\n";
        return false;
    }
    out_buffer.mapped = nullptr;
    return true;
}

bool upload_device_local(const void* data, VkDeviceSize size,
                         VkBufferUsageFlags usage, Buffer& out_buffer) {
    Buffer staging{};
    if (!create(size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, staging)) return false;
    std::memcpy(staging.mapped, data, static_cast<size_t>(size));

    if (!create_device_local(size, usage, out_buffer)) {
        destroy(staging);
        return false;
    }

    const VkBuffer handle_src = staging.handle;
    const VkBuffer handle_dst = out_buffer.handle;
    const bool ok = one_time_submit([&](VkCommandBuffer cmd) {
        const VkBufferCopy region = { .srcOffset = 0, .dstOffset = 0, .size = size };
        vkCmdCopyBuffer(cmd, handle_src, handle_dst, 1, &region);

        const VkBufferMemoryBarrier barrier = {
            .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
            .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT | VK_ACCESS_INDEX_READ_BIT,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .buffer = handle_dst,
            .offset = 0,
            .size = size,
        };
        vkCmdPipelineBarrier(cmd,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_VERTEX_INPUT_BIT,
            0, 0, nullptr, 1, &barrier, 0, nullptr);
    });

    destroy(staging);
    return ok;
}

void destroy(Buffer& buffer) {
    if (buffer.handle != VK_NULL_HANDLE) {
        vmaDestroyBuffer(graphics::internal::context.allocator, buffer.handle, buffer.allocation);
        buffer = {};
    }
}

} // namespace vk_buffer