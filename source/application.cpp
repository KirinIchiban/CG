#include "geometry.hpp"
#include "graphics_internal.hpp"

#include <cmath>
#include <cstddef>
#include <cstring>
#include <fstream>
#include <iostream>
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

struct alignas(16) UBO {
	glm::mat4 model;
	glm::mat4 view;
	glm::mat4 proj;
};
static_assert(sizeof(UBO) == 192);
static bool show_imgui_demo = false;


struct SceneObject {
	float position[3]{0.0f, 0.0f, 0.0f};
	float rotation[3]{0.0f, 0.0f, 0.0f};
	float scale[3]{1.0f, 1.0f, 1.0f};
	float tint[3]{1.0f, 1.0f, 1.0f};
	
	bool trajectory = true;
	bool playing = true;
	float phase = 0.0f;
	float speed = 0.8f;
	float radius = 2.0f;
	float bob_amplitude = 0.4f;
	float bank_gain = 1.5f;
	float max_bank = 30.0f;
	
	float roll_deg = 0.0f;
	glm::vec3 prev_vel = glm::vec3(0.0f);
	float spin_angle_y = 0.0f;

	vk_buffer::Buffer uniform_buffer;
	VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
};

constexpr size_t kObjectCount = 2;
std::array<SceneObject, kObjectCount> objects;

struct ConeMesh {
	std::vector<Vertex> vertices;
	std::vector<uint32_t> indices;
};

ConeMesh generateCone(float radius, float height, uint32_t segments) {
	ConeMesh mesh;
	if (segments < 3) return mesh;
	mesh.vertices.reserve(segments + 2);
	mesh.indices.reserve(segments * 6);
	const float half_h = height * 0.5f;

	// верхняя вершина
	mesh.vertices.push_back({ .position = {0.0f, half_h, 0.0f}, .color = {1.0f, 1.0f, 1.0f} });
	// центр основания
	mesh.vertices.push_back({ .position = {0.0f, -half_h, 0.0f}, .color = {0.2f, 0.2f, 0.2f} });

	// точки основания с цветом от локальных координат
	for (uint32_t i = 0; i < segments; ++i) {
		const float theta = 2.0f * glm::pi<float>() * float(i) / float(segments);
		const float x = radius * std::cos(theta);
		const float z = radius * std::sin(theta);
		const glm::vec3 local_color = glm::vec3(
			std::abs(x) / radius,
			(std::cos(theta) + 1.0f) * 0.5f,
			std::abs(z) / radius
		);
		mesh.vertices.push_back({ .position = {x, -half_h, z}, .color = local_color });
	}

	for (uint32_t i = 0; i < segments; ++i) {
		mesh.indices.push_back(0);
		mesh.indices.push_back(2 + i);
		mesh.indices.push_back(2 + (i + 1) % segments);
	}
	for (uint32_t i = 0; i < segments; ++i) {
		mesh.indices.push_back(1);
		mesh.indices.push_back(2 + (i + 1) % segments);
		mesh.indices.push_back(2 + i);
	}
	return mesh;
}

vk_buffer::Buffer vertex_buffer;
vk_buffer::Buffer index_buffer;
uint32_t current_index_count = 0;

VkDescriptorSetLayout descriptor_set_layout = VK_NULL_HANDLE;
VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
VkPipeline pipeline = VK_NULL_HANDLE;

int projection_type = 0; // 0 - Perspective, 1 - Orthographicbool 
float perspective_fov = 45.0f;
double previous_time = -1.0;

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

bool createBuffersAndDescriptors() {
	auto& context = graphics::internal::context;

	ConeMesh cone = generateCone(1.0f, 2.0f, 36);
	current_index_count = static_cast<uint32_t>(cone.indices.size());

	std::cerr << "cone.indices.size() = "
          << cone.indices.size() << '\n';

	std::cerr << "current_index_count = "
			<< current_index_count << '\n';
			
	const VkDeviceSize v_size = sizeof(cone.vertices[0]) * cone.vertices.size();
	if (!vk_buffer::create(v_size, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, vertex_buffer)) return false;
	std::memcpy(vertex_buffer.mapped, cone.vertices.data(), v_size);

	const VkDeviceSize i_size = sizeof(cone.indices[0]) * cone.indices.size();
	if (!vk_buffer::create(i_size, VK_BUFFER_USAGE_INDEX_BUFFER_BIT, index_buffer)) return false;
	std::memcpy(index_buffer.mapped, cone.indices.data(), i_size);

	objects[0].scale[0] = 0.8f; objects[0].scale[1] = 0.8f; objects[0].scale[2] = 0.8f;
	objects[1].scale[0] = 0.8f; objects[1].scale[1] = 0.8f; objects[1].scale[2] = 0.8f;
	objects[0].position[0] = 0.0f;
	objects[1].position[0] = 2.5f;
	objects[0].phase = 0.0f;
	objects[1].phase = -1.2f;

	for (auto& obj : objects) {
		if (!vk_buffer::create(sizeof(UBO), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, obj.uniform_buffer)) {
			return false;
		}
	}

	const VkDescriptorSetLayoutBinding binding = {
		.binding = 0,
		.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
		.descriptorCount = 1,
		.stageFlags = VK_SHADER_STAGE_VERTEX_BIT,
	};
	const VkDescriptorSetLayoutCreateInfo layout_info = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
		.bindingCount = 1,
		.pBindings = &binding,
	};
	if (vkCreateDescriptorSetLayout(context.device, &layout_info, nullptr, &descriptor_set_layout) != VK_SUCCESS) {
		return false;
	}

	const VkDescriptorPoolSize pool_size = {
		.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
		.descriptorCount = static_cast<uint32_t>(objects.size()),
	};
	const VkDescriptorPoolCreateInfo pool_info = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
		.maxSets = static_cast<uint32_t>(objects.size()),
		.poolSizeCount = 1,
		.pPoolSizes = &pool_size,
	};
	if (vkCreateDescriptorPool(context.device, &pool_info, nullptr, &descriptor_pool) != VK_SUCCESS) {
		return false;
	}

	std::vector<VkDescriptorSetLayout> layouts(objects.size(), descriptor_set_layout);
	const VkDescriptorSetAllocateInfo alloc_info = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
		.descriptorPool = descriptor_pool,
		.descriptorSetCount = static_cast<uint32_t>(layouts.size()),
		.pSetLayouts = layouts.data(),
	};

	std::vector<VkDescriptorSet> allocated_sets(objects.size());
	if (vkAllocateDescriptorSets(context.device, &alloc_info, allocated_sets.data()) != VK_SUCCESS) {
		return false;
	}

	for (size_t i = 0; i < objects.size(); ++i) {
		objects[i].descriptor_set = allocated_sets[i];
		const VkDescriptorBufferInfo buffer_info = {
			.buffer = objects[i].uniform_buffer.handle,
			.offset = 0,
			.range = sizeof(UBO),
		};
		const VkWriteDescriptorSet write = {
			.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
			.dstSet = objects[i].descriptor_set,
			.dstBinding = 0,
			.descriptorCount = 1,
			.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
			.pBufferInfo = &buffer_info,
		};
		vkUpdateDescriptorSets(context.device, 1, &write, 0, nullptr);
	}

	return true;
}

bool createPipeline() {
	auto& context = graphics::internal::context;
	VkShaderModule vs = loadShaderModule("cone.vert.spv");
	VkShaderModule fs = loadShaderModule("cone.frag.spv");

	const VkPipelineShaderStageCreateInfo stages[] = {
		{ .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_VERTEX_BIT, .module = vs, .pName = "main" },
		{ .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_FRAGMENT_BIT, .module = fs, .pName = "main" },
	};

	const VkVertexInputBindingDescription binding = {
		.binding = 0,
		.stride = sizeof(Vertex),
		.inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
	};
	const VkVertexInputAttributeDescription attributes[] = {
		{ .location = 0, .binding = 0, .format = VK_FORMAT_R32G32B32_SFLOAT, .offset = offsetof(Vertex, position) },
		{ .location = 1, .binding = 0, .format = VK_FORMAT_R32G32B32_SFLOAT, .offset = offsetof(Vertex, color) },
	};

	const VkPipelineVertexInputStateCreateInfo vertex_input = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
		.vertexBindingDescriptionCount = 1,
		.pVertexBindingDescriptions = &binding,
		.vertexAttributeDescriptionCount = 2,
		.pVertexAttributeDescriptions = attributes,
	};

	const VkPipelineInputAssemblyStateCreateInfo input_assembly = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
		.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
	};

	const VkPipelineViewportStateCreateInfo viewport_state = { .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO, .viewportCount = 1, .scissorCount = 1 };
	const VkPipelineRasterizationStateCreateInfo rasterization = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
		.polygonMode = VK_POLYGON_MODE_FILL,
		.cullMode = VK_CULL_MODE_BACK_BIT,
		.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE,
		.lineWidth = 1.0f,
	};
	const VkPipelineMultisampleStateCreateInfo multisample = { .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO, .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT };
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

	const VkDynamicState dynamic_states[] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
	const VkPipelineDynamicStateCreateInfo dynamic_state = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
		.dynamicStateCount = 2,
		.pDynamicStates = dynamic_states,
	};

	const VkPushConstantRange push_constant_range = { .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT, .offset = 0, .size = sizeof(glm::vec4) };
	const VkPipelineLayoutCreateInfo layout_info = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
		.setLayoutCount = 1,
		.pSetLayouts = &descriptor_set_layout,
		.pushConstantRangeCount = 1,
		.pPushConstantRanges = &push_constant_range,
	};
	vkCreatePipelineLayout(context.device, &layout_info, nullptr, &pipeline_layout);

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
	vkCreateGraphicsPipelines(context.device, VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &pipeline);

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
	vkDestroyPipelineLayout(context.device, pipeline_layout, nullptr);
	vkDestroyDescriptorPool(context.device, descriptor_pool, nullptr);
	vkDestroyDescriptorSetLayout(context.device, descriptor_set_layout, nullptr);

	vk_buffer::destroy(vertex_buffer);
	vk_buffer::destroy(index_buffer);
	for (auto& obj : objects) {
		vk_buffer::destroy(obj.uniform_buffer);
	}
}

void update(double time) {
	const double dt = (previous_time >= 0.0) ? (time - previous_time) : 0.0;
	previous_time = time;

	ImGui::Begin("Lab Controls (Variant 9)");

	ImGui::SeparatorText("Projection (Task 1)");
	ImGui::RadioButton("Perspective", &projection_type, 0);
	ImGui::SameLine();
	ImGui::RadioButton("Orthographic", &projection_type, 1);

	if (projection_type == 0) {
		ImGui::SliderFloat("FOV", &perspective_fov, 10.0f, 120.0f);
	}

	for (size_t i = 0; i < objects.size(); ++i) {
		ImGui::PushID(static_cast<int>(i));
		char label[64];
		std::snprintf(label, sizeof(label), "Object %zu Settings", i + 1);
		if (ImGui::CollapsingHeader(label)) {
			ImGui::Checkbox("Animate Trajectory (Lemniscate)", &objects[i].trajectory);
			if (objects[i].trajectory) {
				if (ImGui::Button(objects[i].playing ? "Pause" : "Play")) {
					objects[i].playing = !objects[i].playing;
				}
				ImGui::SliderFloat("Speed", &objects[i].speed, 0.1f, 3.0f);
				ImGui::SliderFloat("Radius", &objects[i].radius, 1.0f, 4.0f);
				ImGui::SliderFloat("Bob Amplitude", &objects[i].bob_amplitude, 0.0f, 1.0f);
				ImGui::SliderFloat("Bank Gain", &objects[i].bank_gain, 0.0f, 1.0f);
			} else {
				ImGui::SliderFloat3("Position", objects[i].position, -3.0f, 3.0f);
				ImGui::SliderFloat3("Rotation", objects[i].rotation, -180.0f, 180.0f);
			}
			ImGui::SliderFloat3("Scale", objects[i].scale, 0.1f, 3.0f);
			ImGui::ColorEdit3("Color Tint", objects[i].tint);
		}
		ImGui::PopID();
	}

	ImGui::SeparatorText("ImGui Demo");
	ImGui::Checkbox("Show Demo Window", &show_imgui_demo);
	if (show_imgui_demo) {
		ImGui::ShowDemoWindow(&show_imgui_demo);
	}

	ImGui::End();

	for (size_t i = 0; i < objects.size(); ++i) {
		auto& obj = objects[i];
		if (obj.trajectory && obj.playing) {
			float t = obj.phase;
			float a = obj.radius;
			float denom = 1.0f + std::sin(t) * std::sin(t);

			if (std::abs(denom) < 1e-5f) denom = 1e-5f;

			glm::vec3 pos = {
				a * std::cos(t) / denom,
				obj.bob_amplitude * std::sin(t * 2.0f),
				a * std::sin(t) * std::cos(t) / denom
			};

			if (i == 1) {
				obj.spin_angle_y += static_cast<float>(dt) * 45.0f;
				if (obj.spin_angle_y > 360.0f) obj.spin_angle_y -= 360.0f;
			}

			float speed_mult = obj.speed * 0.6f; 
			obj.phase += static_cast<float>(dt) * speed_mult;
		}
	}
}

void render(const graphics::internal::FrameData& fd) {
	auto& context = graphics::internal::context;
	float aspect = (context.swapchain_extent.height > 0) 
		? static_cast<float>(context.swapchain_extent.width) / static_cast<float>(context.swapchain_extent.height) 
		: 1.0f;

	glm::mat4 view = glm::lookAt(glm::vec3(0.0f, 2.5f, 5.0f), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
	glm::mat4 proj;
	
	if (projection_type == 0) {
		proj = glm::perspective(glm::radians(perspective_fov), aspect, 0.1f, 100.0f);
	} else {
		constexpr float extent = 5.0f; 
		proj = glm::ortho(-extent * aspect, extent * aspect, -extent, extent, -100.0f, 100.0f);
	}
	proj[1][1] *= -1.0f;

	for (size_t i = 0; i < objects.size(); ++i) {
		auto& obj = objects[i];
		UBO ubo{};

		if (obj.trajectory) {
			float t = obj.phase;
			float a = obj.radius;
			float denom = 1.0f + std::sin(t) * std::sin(t);
			if (std::abs(denom) < 1e-5f) denom = 1e-5f;

			glm::vec3 pos = {
				a * std::cos(t) / denom,
				obj.bob_amplitude * std::sin(t * 2.0f),
				a * std::sin(t) * std::cos(t) / denom
			};

			glm::mat4 model = glm::translate(glm::mat4(1.0f), pos);

			if (i == 1) {
				model = glm::rotate(model, glm::radians(obj.spin_angle_y), glm::vec3(0.0f, 1.0f, 0.0f));
			} else {
				model = glm::rotate(model, glm::radians(std::sin(t) * 15.0f * obj.bank_gain), glm::vec3(0.0f, 0.0f, 1.0f));
			}

			model = glm::scale(model, glm::make_vec3(obj.scale));
			ubo.model = model;
		} else {
			glm::mat4 model = glm::translate(glm::mat4(1.0f), glm::make_vec3(obj.position));
			model = glm::rotate(model, glm::radians(obj.rotation[0]), glm::vec3(1.0f, 0.0f, 0.0f));
			model = glm::rotate(model, glm::radians(obj.rotation[1]), glm::vec3(0.0f, 1.0f, 0.0f));
			model = glm::rotate(model, glm::radians(obj.rotation[2]), glm::vec3(0.0f, 0.0f, 1.0f));
			model = glm::scale(model, glm::make_vec3(obj.scale));
			ubo.model = model;
		}

		ubo.view = view;
		ubo.proj = proj;

		std::memcpy(obj.uniform_buffer.mapped, &ubo, sizeof(UBO));
	}

	const VkCommandBufferBeginInfo begin_info = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
	vkBeginCommandBuffer(fd.command_buffer, &begin_info);

	const VkClearValue clear_values[] = {
		{ .color = { { 0.08f, 0.08f, 0.12f, 1.0f } } },
		{ .depthStencil = { 1.0f, 0 } },
	};

	const VkRenderPassBeginInfo render_pass_begin = {
		.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
		.renderPass = context.render_pass,
		.framebuffer = fd.framebuffer,
		.renderArea = { .extent = context.swapchain_extent },
		.clearValueCount = 2,
		.pClearValues = clear_values,
	};

	vkCmdBeginRenderPass(fd.command_buffer, &render_pass_begin, VK_SUBPASS_CONTENTS_INLINE);
	vkCmdBindPipeline(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);

	const VkViewport viewport = { 0.0f, 0.0f, float(context.swapchain_extent.width), float(context.swapchain_extent.height), 0.0f, 1.0f };
	vkCmdSetViewport(fd.command_buffer, 0, 1, &viewport);

	const VkRect2D scissor = { .extent = context.swapchain_extent };
	vkCmdSetScissor(fd.command_buffer, 0, 1, &scissor);

	const VkBuffer vertex_buffers[] = { vertex_buffer.handle };
	const VkDeviceSize offsets[] = { 0 };
	vkCmdBindVertexBuffers(fd.command_buffer, 0, 1, vertex_buffers, offsets);
	vkCmdBindIndexBuffer(fd.command_buffer, index_buffer.handle, 0, VK_INDEX_TYPE_UINT32);

    for (const auto& obj : objects) {
		vkCmdBindDescriptorSets(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout, 0, 1, &obj.descriptor_set, 0, nullptr);

		glm::vec4 push_color(obj.tint[0], obj.tint[1], obj.tint[2], 1.0f);
		vkCmdPushConstants(fd.command_buffer, pipeline_layout, VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(glm::vec4), &push_color);

		vkCmdDrawIndexed(fd.command_buffer, current_index_count, 1, 0, 0, 0);
	}

	vkCmdEndRenderPass(fd.command_buffer);
	vkEndCommandBuffer(fd.command_buffer);
}
} // namespace application