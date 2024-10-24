//
// Created by nick on 10/16/24.
//

#include "LTCN.h"

#include <cstdlib>
#include <iostream>
#include <ostream>
#include <vulkan/vulkan.h>

namespace LTCN {
    LTCN::LTCN() {


    }


    Layer::Layer(VU::Instance instance, Bounds inWin, Bounds outWin) : inputWindow(inputWindow), outputWindow(outputWindow) {
        weights = new VU::Buffer(0, 0, 0);
    }

    LTCN::~LTCN() {
    }

    void * LTCN::evaluate(void *input, float timestep) {
        return nullptr;
    }

    void LTCN::train(void *input, void *output, float strength, float timestep) {
    }

    Rect::Rect(int x, int y, int width, int height) : x(x), y(y), width(width), height(height) {}

    Rect::Rect(int width, int height) : x(0), y(0), width(width), height(height) {}

    NetworkTensor::NetworkTensor(VkInstance instance, unsigned int id, std::vector<size_t> shape) {
        buffer = new VU::Buffer(in);
    }

    NetworkTensor::~NetworkTensor() {
        delete buffer;
    }

    namespace VU {
        Buffer::Buffer(VkPhysicalDevice physicalDevice, VkDevice device, size_t size) : size(size), device(device), physicalDevice(physicalDevice) {
            VkBufferCreateInfo bufferInfo = {
                .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                .size = size,
                .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
            };
            if(vkCreateBuffer(device, &bufferInfo, nullptr, &buffer) != VK_SUCCESS) {
                std::cerr << "Failed to create buffer" << std::endl;
                exit(0);
            }
            VkMemoryRequirements memoryRequirements;
            vkGetBufferMemoryRequirements(device, buffer, &memoryRequirements);
            VkPhysicalDeviceMemoryProperties memoryProperties;
            vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memoryProperties);
            int memoryTypeIndex = -1;
            for(int i = 0; i < memoryProperties.memoryTypeCount; i++) {
                unsigned int wantedMemoryProperties = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
                if((memoryProperties.memoryTypes[i].propertyFlags & wantedMemoryProperties) == wantedMemoryProperties) {
                    memoryTypeIndex = i;
                    break;
                }
            }
            VkMemoryAllocateInfo allocateInfo = {
                .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                .allocationSize = memoryRequirements.size,
                .memoryTypeIndex = (uint32_t)memoryTypeIndex,
            };
            vkAllocateMemory(device, &allocateInfo, nullptr, &deviceMemory);
        }

        Buffer::Buffer(Instance * instance, size_t size) {
            Buffer(instance->getPhysicalDevice(), instance->getDevice(), size);
        }

        Buffer::~Buffer() {
            vkDestroyBuffer(device, buffer, nullptr);
            vkFreeMemory(device, deviceMemory, nullptr);
        }

        void * Buffer::map() {
            if(data.has_value()) {
                vkUnmapMemory(device, deviceMemory);
            }
            void * retval;
            vkMapMemory(device, deviceMemory, 0, VK_WHOLE_SIZE, 0, &retval);
            data = retval;
            return retval;
        }

        void * Buffer::map(size_t offset, size_t size) {
            if(data.has_value()) {
                vkUnmapMemory(device, deviceMemory);
            }
            void * retval;
            vkMapMemory(device, deviceMemory, offset, size, 0, &retval);
            data = retval;
            return retval;
        }

        void Buffer::unmap() {
            if(data.has_value()) {
                vkUnmapMemory(device, deviceMemory);
            }
        }

        void Buffer::flush() {
            if(!data.has_value()) {
                return;
            }
            VkMappedMemoryRange mappedMemoryRange = {
                .sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE,
                .memory = deviceMemory,
                .offset = 0,
                .size = VK_WHOLE_SIZE
            };
            vkFlushMappedMemoryRanges(device, 1, &mappedMemoryRange);
        }

        Instance::Instance() {
            VkApplicationInfo appInfo = {
                .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
                .pApplicationName = "LTCN",
                .applicationVersion = VK_MAKE_VERSION(1, 0, 0),
                .pEngineName = "No Engine",
                .engineVersion = VK_MAKE_VERSION(1, 0, 0),
                .apiVersion = VK_API_VERSION_1_1
            };
            const char * enabledLayers[] = {
                "VK_LAYER_KHRONOS_validation"
            };
            VkValidationFeatureEnableEXT enabledValidationFeatures[] = {
                VK_VALIDATION_FEATURE_ENABLE_SYNCHRONIZATION_VALIDATION_EXT,
                VK_VALIDATION_FEATURE_ENABLE_GPU_ASSISTED_EXT,
                VK_VALIDATION_FEATURE_ENABLE_BEST_PRACTICES_EXT
              };

            VkValidationFeaturesEXT validationFeatures = {
                .sType = VK_STRUCTURE_TYPE_VALIDATION_FEATURES_EXT,
                .enabledValidationFeatureCount = 3,
                .pEnabledValidationFeatures = enabledValidationFeatures
              };
            VkInstanceCreateInfo instanceInfo = {
                .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        #ifdef DEBUG
                .pNext = &validationFeatures,
        #endif
                .pApplicationInfo = &appInfo,
        #ifdef DEBUG
                .enabledLayerCount = 1,
        #else
                .enabledLayerCount = 0,
        #endif
                .ppEnabledLayerNames = enabledLayers,
                .enabledExtensionCount = 0,
            };
            if(vkCreateInstance(&instanceInfo, nullptr, &instance) != VK_SUCCESS) {
                std::cerr << "Failed to create instance" << std::endl;
                exit(0);
            }
        }

        Instance::~Instance() {
            vkDestroyInstance(instance, nullptr);
        }

        VkInstance Instance::getInstance() {
            return instance;
        }

        VkPhysicalDevice Instance::getPhysicalDevice() {
            return physicalDevice;
        }

        VkDevice Instance::getDevice() {
            return device;
        }

        Buffer Instance::createBuffer(size_t size) {
            return VU::Buffer(physicalDevice, device, size);
        }
    }
}