#pragma once

#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>

namespace vk_buffer {

struct Buffer {
	VkBuffer handle = VK_NULL_HANDLE;
	VmaAllocation allocation = VK_NULL_HANDLE;
	void* mapped = nullptr;
};

bool create(VkDeviceSize size, VkBufferUsageFlags usage, Buffer& out_buffer);
void destroy(Buffer& buffer);

} // namespace vk_buffer