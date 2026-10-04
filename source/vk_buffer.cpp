#include "vk_buffer.hpp"
#include "graphics_internal.hpp"
#include <iostream>

namespace vk_buffer {

bool create(VkDeviceSize size, VkBufferUsageFlags usage, Buffer& out_buffer) {
	auto& context = graphics::internal::context;
	const VkBufferCreateInfo create_info = {
		.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
		.size = size,
		.usage = usage,
		.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
	};
	const VmaAllocationCreateInfo allocation_info = {
		.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT,
		.usage = VMA_MEMORY_USAGE_AUTO,
	};
	VmaAllocationInfo result_info{};
	if (vmaCreateBuffer(context.allocator, &create_info, &allocation_info, &out_buffer.handle, &out_buffer.allocation, &result_info) != VK_SUCCESS) {
		std::cerr << "Failed to create Vulkan buffer via VMA\n";
		return false;
	}
	out_buffer.mapped = result_info.pMappedData;
	return out_buffer.mapped != nullptr;
}

void destroy(Buffer& buffer) {
	if (buffer.handle != VK_NULL_HANDLE) {
		vmaDestroyBuffer(graphics::internal::context.allocator, buffer.handle, buffer.allocation);
		buffer = {};
	}
}

} // namespace vk_buffer