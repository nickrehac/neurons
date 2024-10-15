#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE

#define VERTEX_SHADER_PATH "vert.spv"
#define FRAGMENT_SHADER_PATH "frag.spv"
#define COMPUTE_SHADER_PATH "comp.spv"

#include <glm/vec4.hpp>
#include <glm/mat4x4.hpp>

#include <iostream>
#include <fstream>
#include <vector>
#include <limits>
#include <algorithm>
#include <optional>
#include <functional>
#include <deque>

class NeuronApplication {
  struct ComputePushConstants {
    float dTime;
    unsigned int mode;
  };

  struct ComputeUBO {
    glm::ivec2 resolution;
    glm::ivec2 networkDimensions;
    glm::vec4 zoom;
  };

  std::deque<std::function<void()>> destructQueue;

  VkInstance instance{};
  GLFWwindow* window;
  VkSurfaceKHR surface{};
  VkPhysicalDevice physicalDevice{};
  VkDevice device{};
  VkQueue graphicsQueue{};
  VkQueue computeQueue{};
  VkQueue presentQueue{};
  uint32_t queueFamilies[3]{};
  VkSwapchainKHR swapchain;
  VkExtent2D swapchainExtent{};
  VkFormat swapchainImageFormat;
  std::vector<VkImage> swapchainImages;
  std::vector<VkImageView> swapchainImageViews;
  std::vector<VkImage> computeImages;
  std::vector<VkDeviceMemory> computeImageMemories;
  std::vector<VkImageView> computeImageViews;
  VkRenderPass renderPass{};
  VkPipelineLayout pipelineLayout{};
  
  std::vector<VkSemaphore> imageAvailableSemaphores;
  int currentFrame;

  std::vector<VkSemaphore> computeSemaphores;
  std::vector<VkFence> drawFences;

  VkCommandPool commandPool{};
  std::vector<VkCommandBuffer> commandBuffers;

  VkPipeline computePipeline{};
  VkPipelineLayout computePipelineLayout{};
  VkDescriptorSetLayout computeDescriptorSetLayout{};

  VkDescriptorPool descriptorPool{};
  std::vector<VkDescriptorSet> computeDescriptorSets;

  unsigned int neuronSize[2] = {100, 100};

  VkBuffer neuronBuffer{};
  VkDeviceMemory neuronBufferMemory{};
  float * neuronBufferMap{};

  VkBuffer weightBuffer{};
  VkDeviceMemory weightBufferMemory{};
  float * weightBufferMap{};

  VkBuffer ubo{};
  VkDeviceMemory uboMemory{};
  ComputeUBO * uboMap{};

  VkRenderPass computeRenderPass{};

  bool framebufferResized = false;

  void createLogicalDevice() {
    //enabled extensions
    const std::vector<const char*> deviceExtensions = {
      VK_KHR_SWAPCHAIN_EXTENSION_NAME
    };

    //get queue family indices
    uint32_t graphicsFamilyIndex = 0;
    uint32_t computeFamilyIndex = 0;
    uint32_t presentFamilyIndex = 0;
    uint32_t queueFamilyCount = 0;
    uint32_t queueFamilyIndexCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, nullptr);
    std::vector<VkQueueFamilyProperties> queueFamilyProperties(queueFamilyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, queueFamilyProperties.data());
    for(int i = 0; i < queueFamilyCount; i++) {
      if(queueFamilyProperties[i].queueFlags & (VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT)) {
        graphicsFamilyIndex = i;
        queueFamilyIndexCount++;
        break;
      }
      if(queueFamilyProperties[i].queueFlags & VK_QUEUE_COMPUTE_BIT & ~VK_QUEUE_GRAPHICS_BIT) {
        computeFamilyIndex = i;
        queueFamilyIndexCount++;
      }
      VkBool32 presentSupport = false;
      vkGetPhysicalDeviceSurfaceSupportKHR(physicalDevice, i, surface, &presentSupport);
      if(presentSupport && queueFamilyProperties[i].queueFlags & ~VK_QUEUE_COMPUTE_BIT & ~VK_QUEUE_GRAPHICS_BIT) {
        presentFamilyIndex = i;
        queueFamilyIndexCount++;
      }
    }


    float priority = 1.0f;

    //graphics queue
    VkDeviceQueueCreateInfo graphicsQueueCreateInfo = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
      .queueFamilyIndex = graphicsFamilyIndex,
      .queueCount = 1,
      .pQueuePriorities = &priority
    };
    
    //compute queue
    VkDeviceQueueCreateInfo computeQueueCreateInfo = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
      .queueFamilyIndex = computeFamilyIndex,
      .queueCount = 1,
      .pQueuePriorities = &priority
    };

    //present queue
    VkDeviceQueueCreateInfo presentQueueCreateInfo = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
      .queueFamilyIndex = presentFamilyIndex,
      .queueCount = 1,
      .pQueuePriorities = &priority
    };

    VkDeviceQueueCreateInfo deviceQueues[3] = {graphicsQueueCreateInfo, computeQueueCreateInfo, presentQueueCreateInfo};

    //create device
    VkDeviceCreateInfo deviceCreateInfo = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
      .queueCreateInfoCount = 1,
      .pQueueCreateInfos = deviceQueues,
      .enabledExtensionCount = (unsigned int)deviceExtensions.size(),
      .ppEnabledExtensionNames = deviceExtensions.data()
    };

    VkResult res = vkCreateDevice(physicalDevice, &deviceCreateInfo, nullptr, &device);
    destructQueue.emplace_front([=]() {
      vkDestroyDevice(device, nullptr);
    });

    if(res != VK_SUCCESS) {
      std::cout << "failed to create logical device";
    }

    vkGetDeviceQueue(device, graphicsFamilyIndex, 0, &graphicsQueue);
    vkGetDeviceQueue(device, computeFamilyIndex, 0, &computeQueue);
    vkGetDeviceQueue(device, presentFamilyIndex, 0, &presentQueue);
    queueFamilies[0] = graphicsFamilyIndex;
    queueFamilies[1] = computeFamilyIndex;
    queueFamilies[2] = presentFamilyIndex;
  }

  void findAdequateHardware() {
    uint32_t deviceCount = 0;
    vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);
    if(deviceCount == 0) {
      std::cout<<"pain";
    }
    std::vector<VkPhysicalDevice> deviceList(deviceCount);
    vkEnumeratePhysicalDevices(instance, &deviceCount, deviceList.data());
    std::pair<VkPhysicalDeviceType, VkPhysicalDevice> bestDevice;
    for(int i = 0; i < deviceCount; i++) {
      //list device names?
      VkPhysicalDeviceProperties deviceInfo;
      vkGetPhysicalDeviceProperties(deviceList[i], &deviceInfo);
      std::cout << "Device " << deviceInfo.deviceID << ": " << deviceInfo.deviceName << "   type: " << deviceInfo.deviceType << "\n";

      VkSurfaceCapabilitiesKHR surfaceCapabilities;
      vkGetPhysicalDeviceSurfaceCapabilitiesKHR(deviceList[i], surface, &surfaceCapabilities);

      std::cout << "\tDirect Compute Draw? ";
      if(surfaceCapabilities.supportedUsageFlags & VK_IMAGE_USAGE_STORAGE_BIT) std::cout << "YES\n";
      else std::cout << "NO\n";
      
      if(deviceInfo.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
        bestDevice.first = deviceInfo.deviceType;
        bestDevice.second = deviceList[i];
        break;
      }
      if(deviceInfo.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU) {
        bestDevice.first = deviceInfo.deviceType;
        bestDevice.second = deviceList[i];
      }
      
      //use if certain features are required
      //VkPhysicalDeviceFeatures deviceFeatures;
      //vkGetPhysicalDeviceFeatures(deviceList[i], &deviceFeatures);
    }
    physicalDevice = bestDevice.second;
  }

  void recreateSwapchain() {
    vkDeviceWaitIdle(device);

    destroySwapchain();

    createSwapchain();

    std::vector<VkWriteDescriptorSet> descriptorSetWrites(computeImages.size());
    std::vector<VkDescriptorImageInfo> descriptorImageInfos(computeImages.size());
    for(int i = 0; i < descriptorSetWrites.size(); i++) {
      descriptorImageInfos[i] = {
        .imageView = computeImageViews[i],
        .imageLayout = VK_IMAGE_LAYOUT_GENERAL
      };
      descriptorSetWrites[i] = {
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstSet = computeDescriptorSets[i],
        .dstBinding = 1,
        .dstArrayElement = 0,
        .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
        .pImageInfo = &descriptorImageInfos[i]
      };
      uboMap->resolution.x = swapchainExtent.width;
      uboMap->resolution.y = swapchainExtent.height;
      VkMappedMemoryRange mappedMemoryRange = {
        .sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE,
        .memory = uboMemory,
        .offset = 0,
        .size = sizeof(ComputeUBO)
      };
      vkFlushMappedMemoryRanges(device, 1, &mappedMemoryRange);
    }

    vkUpdateDescriptorSets(device,
      descriptorSetWrites.size(),
      descriptorSetWrites.data(),
      0, nullptr);
  }

  void createSwapchain() {
    VkSurfaceCapabilitiesKHR capabilities;
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physicalDevice, surface, &capabilities);
    uint32_t formatCount;
    vkGetPhysicalDeviceSurfaceFormatsKHR(physicalDevice, surface, &formatCount, nullptr);
    std::vector<VkSurfaceFormatKHR> formats(formatCount);
    vkGetPhysicalDeviceSurfaceFormatsKHR(physicalDevice, surface, &formatCount, formats.data());
    uint32_t presentCount;
    vkGetPhysicalDeviceSurfacePresentModesKHR(physicalDevice, surface, &presentCount, nullptr);
    std::vector<VkPresentModeKHR> presentModes(presentCount);
    vkGetPhysicalDeviceSurfacePresentModesKHR(physicalDevice, surface, &presentCount, presentModes.data());

    VkSurfaceFormatKHR chosenFormat = formats[0];
    
    for(VkSurfaceFormatKHR f : formats) {
      if(f.format == VK_FORMAT_B8G8R8A8_UNORM && f.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
        chosenFormat = f;
      }
    }
    swapchainImageFormat = chosenFormat.format;

    VkPresentModeKHR chosenPresentMode = presentModes[0];

    for(VkPresentModeKHR p : presentModes) {
      if(p == VK_PRESENT_MODE_MAILBOX_KHR) chosenPresentMode = p;
    }
    
    VkExtent2D chosenExtent = capabilities.currentExtent;
    if(capabilities.currentExtent.width == std::numeric_limits<uint32_t>::max()) {
      int width, height;
      glfwGetFramebufferSize(window, &width, &height);
      chosenExtent = {
        (unsigned short)width,
        (unsigned short)height
      };
      chosenExtent.width = std::clamp(chosenExtent.width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width);
      chosenExtent.height = std::clamp(chosenExtent.height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height);
    }
    if(chosenExtent.height <= 0 || chosenExtent.width <= 0) {
      std::cerr << "Invalid extent" << std::endl;
    }
    swapchainExtent = chosenExtent;

    uint32_t chosenImageCount = capabilities.minImageCount + 1;
    if(capabilities.maxImageCount != 0 || chosenImageCount < capabilities.maxImageCount) {
      chosenImageCount = capabilities.maxImageCount;
    }

    VkSwapchainCreateInfoKHR scCreateInfo = {
      .sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
      .surface = surface,
      .minImageCount = chosenImageCount,
      .imageFormat = chosenFormat.format,
      .imageColorSpace = chosenFormat.colorSpace,
      .imageExtent = chosenExtent,
      .imageArrayLayers = 1,
      .imageUsage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_STORAGE_BIT,
      .preTransform = capabilities.currentTransform,
      .compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
      .presentMode = chosenPresentMode,
      .clipped = VK_TRUE,
    };

    if(queueFamilies[0] == queueFamilies[2]) {
      scCreateInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    }
    else {
      scCreateInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
      scCreateInfo.queueFamilyIndexCount = 2;
      uint32_t familyIndices[2] = {queueFamilies[0], queueFamilies[2]};
      scCreateInfo.pQueueFamilyIndices = familyIndices;
    }


    int retval = vkCreateSwapchainKHR(device, &scCreateInfo, nullptr, &swapchain);
    if(retval != VK_SUCCESS) {
      printf("SWAPCHAIN CREATION FAILED %d", retval);
      exit(1);
    }



    uint32_t numImages;
    vkGetSwapchainImagesKHR(device, swapchain, &numImages, nullptr);
    swapchainImages.resize(numImages);
    vkGetSwapchainImagesKHR(device, swapchain, &numImages, swapchainImages.data());


    computeImages.resize(swapchainImages.size());
    computeImageMemories.resize(swapchainImages.size());
    VkImageCreateInfo computeImageInfo = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
      .imageType = VK_IMAGE_TYPE_2D,
      .format = VK_FORMAT_B8G8R8A8_UNORM,
      .extent = {
        chosenExtent.width,
        chosenExtent.height,
        1
      },
      .mipLevels = 1,
      .arrayLayers = 1,
      .samples = VK_SAMPLE_COUNT_1_BIT,
      .tiling = VK_IMAGE_TILING_OPTIMAL,
      .usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
      .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
      .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
    };
    VkPhysicalDeviceMemoryProperties memoryProperties;
    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memoryProperties);
    VkMemoryRequirements memoryRequirements;
    int memoryTypeIndex = -1;
    for(int i = 0; i < memoryProperties.memoryTypeCount; i++) {
      unsigned int wantedMemoryProperties = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
      if((memoryProperties.memoryTypes[i].propertyFlags & wantedMemoryProperties) == wantedMemoryProperties) {
        memoryTypeIndex = i;
        break;
      }
    }
    if(memoryTypeIndex == -1) {
      std::cout << "could not find appropriate memory type for image allocation\n";
      exit(1);
    }

    for(int i = 0; i < swapchainImages.size(); i++) {
      if(vkCreateImage(device, &computeImageInfo, nullptr, &computeImages[i]) != VK_SUCCESS) {
        printf("Failed to create compute image\n");
        exit(1);
      }
      vkGetImageMemoryRequirements(device, computeImages[i], &memoryRequirements);
      VkMemoryAllocateInfo computeImageMemoryAllocateInfo = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .allocationSize = memoryRequirements.size,
        .memoryTypeIndex = (uint32_t)memoryTypeIndex,
      };
      if(vkAllocateMemory(device, &computeImageMemoryAllocateInfo, nullptr, &computeImageMemories[i]) != VK_SUCCESS) {
      }
      if(vkBindImageMemory(device, computeImages[i], computeImageMemories[i], 0) != VK_SUCCESS) {
      }
    }

    createImageViews();
  }

  void destroySwapchain() {
    for(VkImageView v : swapchainImageViews) {
      vkDestroyImageView(device, v, nullptr);
    }
    for(VkImageView v : computeImageViews) {
      vkDestroyImageView(device, v, nullptr);
    }
    for(VkDeviceMemory m : computeImageMemories) {
      vkFreeMemory(device, m, nullptr);
    }
    for(VkImage i : computeImages) {
      vkDestroyImage(device, i, nullptr);
    }
    vkDestroySwapchainKHR(device, swapchain, nullptr);
  }

  void createImageViews() {
    swapchainImageViews.resize(swapchainImages.size());
    computeImageViews.resize(swapchainImages.size());
    for(int i = 0 ; i < swapchainImages.size(); i++) {
      VkImageViewCreateInfo imageViewCreateInfo = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
        .image = swapchainImages[i],
        .viewType = VK_IMAGE_VIEW_TYPE_2D,
        .format = swapchainImageFormat,
        .components = {
          .r = VK_COMPONENT_SWIZZLE_IDENTITY,
          .g = VK_COMPONENT_SWIZZLE_IDENTITY,
          .b = VK_COMPONENT_SWIZZLE_IDENTITY,
          .a = VK_COMPONENT_SWIZZLE_IDENTITY,
        },
        .subresourceRange = {
          .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
          .baseMipLevel = 0,
          .levelCount = 1,
          .baseArrayLayer = 0,
          .layerCount = 1
        }
      };
      if(vkCreateImageView(device, &imageViewCreateInfo, nullptr, &swapchainImageViews[i]) != VK_SUCCESS) {
        printf("error creating swapchain image view");
        exit(1);
      }
      imageViewCreateInfo.image = computeImages[i];
      imageViewCreateInfo.format = VK_FORMAT_B8G8R8A8_UNORM;
      if(vkCreateImageView(device, &imageViewCreateInfo, nullptr, &computeImageViews[i]) != VK_SUCCESS) {
        printf("error creating compute image view");
        exit(1);
      }
    }
  }

  std::optional<VkShaderModule> createShaderModule(const char* filename) {
    std::optional<VkShaderModule> retval;
    std::ifstream file(filename, std::ios_base::ate | std::ios_base::in | std::ios_base::binary);
    if(!file.is_open()) {
      return retval;
    }
    uint32_t fsize = file.tellg();
    file.seekg(0);
    std::vector<char> code(fsize);
    file.read(code.data(), fsize);

    VkShaderModuleCreateInfo moduleCreateInfo = {
      .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      .codeSize = fsize,
      .pCode = reinterpret_cast<uint32_t *>(code.data()),
    };
    VkShaderModule shaderModule;
    if(vkCreateShaderModule(device, &moduleCreateInfo, nullptr, &shaderModule) == VK_SUCCESS) {
      retval = shaderModule;
    }
    destructQueue.emplace_front([=]{vkDestroyShaderModule(device, shaderModule, nullptr);});
    return retval;
  }

  void createComputePipeline() {
    std::optional<VkShaderModule> computeShader = createShaderModule(COMPUTE_SHADER_PATH);
    if(!computeShader.has_value()) {
      std::cout << "could not create compute shader module\n";
      exit(1);
    }

    VkPipelineShaderStageCreateInfo computeStageInfo = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
      .stage = VK_SHADER_STAGE_COMPUTE_BIT,
      .module = *computeShader,
      .pName = "main",
    };

    VkDescriptorSetLayoutBinding descriptorSetLayoutBindings[] = {
      VkDescriptorSetLayoutBinding {
        .binding = 0,
        .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
        .descriptorCount = 1,
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT
      },
      VkDescriptorSetLayoutBinding {
        .binding = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
        .descriptorCount = 1,
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT
      },
      VkDescriptorSetLayoutBinding {
        .binding = 2,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .descriptorCount = 1,
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT
      },
      VkDescriptorSetLayoutBinding {
        .binding = 3,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .descriptorCount = 1,
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT
      }
    };

    VkDescriptorSetLayoutCreateInfo computeDescriptorSetLayoutInfo = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
      .bindingCount = 4,
      .pBindings = descriptorSetLayoutBindings
    };

    if(vkCreateDescriptorSetLayout(device, &computeDescriptorSetLayoutInfo, nullptr, &computeDescriptorSetLayout) != VK_SUCCESS) {
      std::cout << "could not create compute descriptor set layout\n";
      exit(1);
    }
    destructQueue.emplace_front([=] {vkDestroyDescriptorSetLayout(device, computeDescriptorSetLayout, nullptr);});

    VkDescriptorPoolSize descriptorPoolSizes[3] = {
      VkDescriptorPoolSize {
        VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
        (unsigned int)swapchainImages.size()
      },
      VkDescriptorPoolSize {
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        (unsigned int)swapchainImages.size() * 2
      },
      VkDescriptorPoolSize {
        VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
        (unsigned int)swapchainImages.size()
      }
    };

    VkDescriptorPoolCreateInfo descriptorPoolCreateInfo = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
      .maxSets = (unsigned int)swapchainImages.size(),
      .poolSizeCount = 3,
      .pPoolSizes = &descriptorPoolSizes[0]
    };

    if(vkCreateDescriptorPool(device, &descriptorPoolCreateInfo, nullptr, &descriptorPool) != VK_SUCCESS) {
      std::cout << "could not create descriptor pool\n";
      exit(1);
    }
    destructQueue.emplace_front([=]{vkDestroyDescriptorPool(device, descriptorPool, nullptr);});

    std::vector<VkDescriptorSetLayout> setLayouts(swapchainImages.size(), computeDescriptorSetLayout);

    VkDescriptorSetAllocateInfo allocateInfo = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
      .descriptorPool = descriptorPool,
      .descriptorSetCount = (unsigned int)setLayouts.size(),
      .pSetLayouts = setLayouts.data()
    };

    computeDescriptorSets.resize(setLayouts.size());

    if(vkAllocateDescriptorSets(device, &allocateInfo, &computeDescriptorSets[0]) != VK_SUCCESS) {
      std::cout << "could not allocate compute descriptor sets\n";
      exit(1);
    }

    VkPhysicalDeviceMemoryProperties memoryProperties;
    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memoryProperties);
    int memoryTypeIndex = -1;
    for(int i = 0; i < memoryProperties.memoryTypeCount; i++) {
      unsigned int wantedMemoryProperties = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
      if((memoryProperties.memoryTypes[i].propertyFlags & wantedMemoryProperties) == wantedMemoryProperties) {
        memoryTypeIndex = i;
        break;
      }
    }
    if(memoryTypeIndex == -1) {
      std::cout << "could not find appropriate memory type for buffer allocation\n";
      exit(1);
    }

    VkMemoryRequirements memoryRequirements;

    VkBufferCreateInfo uboInfo = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
      .size = sizeof(ComputeUBO),
      .usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
      .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
      .queueFamilyIndexCount = 0
    };
    if(vkCreateBuffer(device, &uboInfo, nullptr, &ubo) != VK_SUCCESS) {
      std::cout << "could not create ubo\n";
      exit(1);
    }
    vkGetBufferMemoryRequirements(device, ubo, &memoryRequirements);
    VkMemoryAllocateInfo uboAllocateInfo = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = memoryRequirements.size,
      .memoryTypeIndex = (unsigned int)memoryTypeIndex
    };
    vkAllocateMemory(device, &uboAllocateInfo, nullptr, &uboMemory);
    vkBindBufferMemory(device, ubo, uboMemory, 0);
    vkMapMemory(device, uboMemory, 0, VK_WHOLE_SIZE, 0, (void**)&uboMap);
    destructQueue.emplace_front([=]{
      vkUnmapMemory(device, uboMemory);
      vkFreeMemory(device, uboMemory, nullptr);
      vkDestroyBuffer(device, ubo, nullptr);
    });
    uboMap->resolution.x = swapchainExtent.width;
    uboMap->resolution.y = swapchainExtent.height;

    VkBufferCreateInfo neuronBufferInfo = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
      .size = neuronSize[0]*neuronSize[1]*sizeof(float),
      .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
      .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
      .queueFamilyIndexCount = 0
    };
    if(vkCreateBuffer(device, &neuronBufferInfo, nullptr, &neuronBuffer) != VK_SUCCESS) {
      std::cout << "could not create neuron buffer\n";
      exit(1);
    }
    vkGetBufferMemoryRequirements(device, neuronBuffer, &memoryRequirements);
    VkMemoryAllocateInfo neuronBufferAllocateInfo = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = memoryRequirements.size,
      .memoryTypeIndex = (unsigned int)memoryTypeIndex
    };
    vkAllocateMemory(device, &neuronBufferAllocateInfo, nullptr, &neuronBufferMemory);
    vkBindBufferMemory(device, neuronBuffer, neuronBufferMemory, 0);
    vkMapMemory(device, neuronBufferMemory, 0, VK_WHOLE_SIZE, 0, (void**)&neuronBufferMap);
    destructQueue.emplace_front([=]{
      vkUnmapMemory(device, neuronBufferMemory);
      vkFreeMemory(device, neuronBufferMemory, nullptr);
      vkDestroyBuffer(device, neuronBuffer, nullptr);
    });

    VkBufferCreateInfo weightBufferInfo = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
      .size = neuronSize[0]*neuronSize[1]*8*sizeof(float),
      .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
      .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
      .queueFamilyIndexCount = 0
    };
    if(vkCreateBuffer(device, &weightBufferInfo, nullptr, &weightBuffer) != VK_SUCCESS) {
      std::cout << "could not create weight buffer\n";
      exit(1);
    }
    vkGetBufferMemoryRequirements(device, weightBuffer, &memoryRequirements);
    VkMemoryAllocateInfo weightBufferAllocateInfo = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = memoryRequirements.size,
      .memoryTypeIndex = (unsigned int)memoryTypeIndex
    };
    vkAllocateMemory(device, &weightBufferAllocateInfo, nullptr, &weightBufferMemory);
    vkBindBufferMemory(device, weightBuffer, weightBufferMemory, 0);
    vkMapMemory(device, weightBufferMemory, 0, VK_WHOLE_SIZE, 0, (void**)&weightBufferMap);
    destructQueue.emplace_front([=] {
      vkUnmapMemory(device, weightBufferMemory);
      vkFreeMemory(device, weightBufferMemory, nullptr);
      vkDestroyBuffer(device, weightBuffer, nullptr);
    });


    std::vector<VkWriteDescriptorSet> descriptorSetWrites;
    descriptorSetWrites.reserve(4*swapchainImages.size());
    std::vector<VkDescriptorImageInfo> descriptorImageInfos(swapchainImages.size());


    for(int i = 0; i < swapchainImages.size(); i++) {
      VkDescriptorBufferInfo descriptorUboInfo = {
        .buffer = ubo,
        .offset = 0,
        .range = VK_WHOLE_SIZE
      };
      VkDescriptorBufferInfo descriptorNeuronInfo = {
        .buffer = neuronBuffer,
        .offset = 0,
        .range = VK_WHOLE_SIZE
      };

      descriptorImageInfos[i] = {
        .imageView = computeImageViews[i],
        .imageLayout = VK_IMAGE_LAYOUT_GENERAL
      };

      VkDescriptorBufferInfo descriptorWeightInfo = {
        .buffer = weightBuffer,
        .offset = 0,
        .range = VK_WHOLE_SIZE
      };

      descriptorSetWrites.push_back({
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstSet = computeDescriptorSets[i],
        .dstBinding = 0,
        .dstArrayElement = 0,
        .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
        .pBufferInfo = &descriptorUboInfo
      });

      descriptorSetWrites.push_back({
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstSet = computeDescriptorSets[i],
        .dstBinding = 1,
        .dstArrayElement = 0,
        .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
        .pImageInfo = &descriptorImageInfos[i]
      });

      descriptorSetWrites.push_back({
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstSet = computeDescriptorSets[i],
        .dstBinding = 2,
        .dstArrayElement = 0,
        .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .pBufferInfo = &descriptorNeuronInfo
      });

      descriptorSetWrites.push_back({
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstSet = computeDescriptorSets[i],
        .dstBinding = 3,
        .dstArrayElement = 0,
        .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .pBufferInfo = &descriptorWeightInfo
      });
    }
    vkUpdateDescriptorSets(
        device,
        descriptorSetWrites.size(), descriptorSetWrites.data(),
        0, nullptr
        );

    VkPushConstantRange pushConstantRange = {
      .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
      .offset = 0,
      .size = sizeof(ComputePushConstants)
    };

    VkPipelineLayoutCreateInfo pipelineLayoutInfo = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .setLayoutCount = 1,
      .pSetLayouts = &computeDescriptorSetLayout,
      .pushConstantRangeCount = 1,
      .pPushConstantRanges = &pushConstantRange
    };
    if(vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &computePipelineLayout) != VK_SUCCESS) {
      std::cout << "could not create compute pipeline layout\n";
      exit(1);
    }
    destructQueue.emplace_front([=]{vkDestroyPipelineLayout(device, computePipelineLayout, nullptr);});
    
    VkComputePipelineCreateInfo computeInfo = {
      .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
      .stage = computeStageInfo,
      .layout = computePipelineLayout
    };
    vkCreateComputePipelines(
        device,
        VK_NULL_HANDLE,
        1, &computeInfo,
        nullptr,
        &computePipeline
        );
    destructQueue.emplace_front([=]{vkDestroyPipeline(device, computePipeline, nullptr);});
  }

  void createSignalling() {
    VkSemaphoreCreateInfo semaphoreCreateInfo = {
      .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO
    };

    VkFenceCreateInfo fenceCreateInfo = {
      .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
      .flags = VK_FENCE_CREATE_SIGNALED_BIT
    };

    imageAvailableSemaphores.resize(swapchainImages.size());
    computeSemaphores.resize(swapchainImages.size());
    drawFences.resize(swapchainImages.size());

    for(int i = 0; i < swapchainImages.size(); i++) {
      vkCreateSemaphore(device, &semaphoreCreateInfo, nullptr, &imageAvailableSemaphores[i]);
    
      vkCreateSemaphore(device, &semaphoreCreateInfo, nullptr, &computeSemaphores[i]);

      vkCreateFence(device, &fenceCreateInfo, nullptr, &drawFences[i]);
    }
    destructQueue.emplace_front([=] {
      for(int i = 0; i < swapchainImages.size(); i++) {
        vkDestroySemaphore(device, imageAvailableSemaphores[i], nullptr);
        vkDestroySemaphore(device, computeSemaphores[i], nullptr);
        vkDestroyFence(device, drawFences[i], nullptr);
      }
    });
  }

  void createCommandBuffers() {
    VkCommandPoolCreateInfo commandPoolInfo = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
      .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
      .queueFamilyIndex = queueFamilies[1]
    };
    if(vkCreateCommandPool(device, &commandPoolInfo, nullptr, &commandPool) != VK_SUCCESS) {
      std::cout << "could not create command pool\n";
      exit(1);
    }
    destructQueue.emplace_front([=]{vkDestroyCommandPool(device, commandPool, nullptr);});
    
    commandBuffers.resize(swapchainImages.size());
    VkCommandBufferAllocateInfo commandBufferInfo = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
      .commandPool = commandPool,
      .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
      .commandBufferCount = (unsigned int)commandBuffers.size()
    };

    if(vkAllocateCommandBuffers(device, &commandBufferInfo, commandBuffers.data()) != VK_SUCCESS) {
      std::cout << "could not create command buffer\n";
      exit(1);
    }
    destructQueue.emplace_front([=]{vkFreeCommandBuffers(device, commandPool, commandBuffers.size(), commandBuffers.data());});
  }

  void runCompute(uint32_t imageIndex) {
    VkCommandBuffer cmd = commandBuffers[currentFrame];

    vkResetCommandBuffer(cmd, 0);

    VkCommandBufferBeginInfo beginInfo = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO
    };

    if(vkBeginCommandBuffer(cmd, &beginInfo) != VK_SUCCESS) {
      std::cout << "could not record to command buffer\n";
      exit(1);
    }

    VkBufferMemoryBarrier uboBarrier = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
      .srcAccessMask = VK_ACCESS_HOST_READ_BIT | VK_ACCESS_HOST_WRITE_BIT,
      .dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT,
      .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .buffer = ubo,
      .offset = 0,
      .size = sizeof(ComputeUBO)
    };

    VkImageMemoryBarrier imageBarrier = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
      .srcAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT,
      .dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT,
      .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
      .newLayout = VK_IMAGE_LAYOUT_GENERAL,
      .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .image = computeImages[currentFrame],
      .subresourceRange = {
        .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
        .baseMipLevel = 0,
        .levelCount = 1,
        .baseArrayLayer = 0,
        .layerCount = 1
      }
    };

    vkCmdBindDescriptorSets(cmd,
      VK_PIPELINE_BIND_POINT_COMPUTE,
      computePipelineLayout,
      0,
      1, &computeDescriptorSets[currentFrame],
      0, nullptr);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, computePipeline);



    vkCmdPipelineBarrier(cmd,
      VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
      VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
      0,
      0, nullptr,
      1, &uboBarrier,
      1, &imageBarrier
      );


    ComputePushConstants pushConstants {
      .dTime = .001,
      .mode = 0
    };
    vkCmdPushConstants(cmd,
      computePipelineLayout,
      VK_SHADER_STAGE_COMPUTE_BIT,
      0,
      sizeof(ComputePushConstants),
      &pushConstants);
    vkCmdDispatch(cmd, 1+neuronSize[0]/32, 1+neuronSize[1]/32, 1);

    VkBufferMemoryBarrier bufferBarriers[] = {
      {
        .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .buffer = neuronBuffer,
        .offset = 0,
        .size = VK_WHOLE_SIZE
      },
      {
        .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .buffer = weightBuffer,
        .offset = 0,
        .size = VK_WHOLE_SIZE
      }
    };
    vkCmdPipelineBarrier(cmd,
      VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
      VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
      0,
      0, nullptr,
      2, bufferBarriers,
      0, nullptr);


    pushConstants = {
      .dTime = .5,
      .mode = 1
    };
    vkCmdPushConstants(cmd,
      computePipelineLayout,
      VK_SHADER_STAGE_COMPUTE_BIT,
      0,
      sizeof(ComputePushConstants),
      &pushConstants);
    vkCmdDispatch(cmd, 1+swapchainExtent.width/32,1+swapchainExtent.height/32,1);

    imageBarrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
    vkCmdPipelineBarrier(cmd,
      VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
      VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
      0,
      0, nullptr,
      0, nullptr,
      1, &imageBarrier
      );

    VkImageBlit regions = {
      .srcSubresource = {
        .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
        .mipLevel = 0,
        .baseArrayLayer = 0,
        .layerCount = 1
      },
      .srcOffsets = {
        {
          .x = 0,
          .y = 0,
          .z = 0
        },
        {
          .x = (int32_t)swapchainExtent.width,
          .y = (int32_t)swapchainExtent.height,
          .z = 1
        }
      },
      .dstSubresource = {
        .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
        .mipLevel = 0,
        .baseArrayLayer = 0,
        .layerCount = 1
      },
      .dstOffsets = {
        {
          .x = 0,
          .y = 0,
          .z = 0
        },
        {
          .x = (int32_t)swapchainExtent.width,
          .y = (int32_t)swapchainExtent.height,
          .z = 1
        }
      }
    };


    imageBarrier.image = swapchainImages[imageIndex];
    imageBarrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    imageBarrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    vkCmdPipelineBarrier(cmd,
      VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
      VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
      0,
      0, nullptr,
      0, nullptr,
      1, &imageBarrier
      );

    vkCmdBlitImage(cmd,
      computeImages[currentFrame],
      VK_IMAGE_LAYOUT_GENERAL,
      swapchainImages[imageIndex],
      VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
      1, &regions,
      VK_FILTER_NEAREST);

    imageBarrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    imageBarrier.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

    vkCmdPipelineBarrier(cmd,
      VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
      VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
      0,
      0, nullptr,
      0, nullptr,
      1, &imageBarrier
      );



    vkEndCommandBuffer(cmd);

    VkPipelineStageFlags waitDstStageMask = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;

    VkSubmitInfo submitInfo = {
      .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
      .waitSemaphoreCount = 1,
      .pWaitSemaphores = &imageAvailableSemaphores[currentFrame],
      .pWaitDstStageMask = &waitDstStageMask,
      .commandBufferCount = 1,
      .pCommandBuffers = &cmd,
      .signalSemaphoreCount = 1,
      .pSignalSemaphores = &computeSemaphores[currentFrame]
    };

    if(vkQueueSubmit(computeQueue, 1, &submitInfo, drawFences[currentFrame]) != VK_SUCCESS) {
      std::cout << "failed to submit compute queue";
      exit(1);
    }
  }

  void drawFrame() {
    vkWaitForFences(device, 1, &drawFences[currentFrame], VK_TRUE, UINT64_MAX);
    uint32_t imageIndex;

    int successState = vkAcquireNextImageKHR(device, swapchain, UINT64_MAX, imageAvailableSemaphores[currentFrame], VK_NULL_HANDLE, &imageIndex);


    vkResetFences(device, 1, &drawFences[currentFrame]);

    runCompute(imageIndex);

    VkPresentInfoKHR presentInfo {
      .sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
      .waitSemaphoreCount = 1,
      .pWaitSemaphores = &computeSemaphores[currentFrame],
      .swapchainCount = 1,
      .pSwapchains = &swapchain,
      .pImageIndices = &imageIndex
    };

    vkQueuePresentKHR(presentQueue, &presentInfo);

    if(successState != VK_SUCCESS || framebufferResized) {
      framebufferResized = false;
      recreateSwapchain();
    }



    //std::cout<< "rendered frame " << currentFrame << " and image " << imageIndex << "\n";

    currentFrame++;
    if(currentFrame >= swapchainImages.size()) currentFrame = 0;
    
  }

  public:

  NeuronApplication() {
    //start glfw
    glfwInit();
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    window = glfwCreateWindow(800, 600, "Neurons", nullptr, nullptr);

    //avoids globals
    glfwSetWindowUserPointer(window, this);

    const char * enabledLayers[] = {
      "VK_LAYER_KHRONOS_validation"
    };

    //start vulkan
    VkApplicationInfo appInfo = {
      .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
      .pApplicationName = "neuron sim",
      .applicationVersion = VK_MAKE_VERSION(1, 0, 0),
      .pEngineName = "No Engine",
      .engineVersion = VK_MAKE_VERSION(1, 0, 0),
      .apiVersion = VK_API_VERSION_1_1
    };

    uint32_t glfwExtensionCount = 0;
    const char **glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

    VkValidationFeatureEnableEXT enabledFeatures[] = {
      VK_VALIDATION_FEATURE_ENABLE_SYNCHRONIZATION_VALIDATION_EXT,
      VK_VALIDATION_FEATURE_ENABLE_GPU_ASSISTED_EXT,
      VK_VALIDATION_FEATURE_ENABLE_BEST_PRACTICES_EXT
    };

    VkValidationFeaturesEXT validationFeatures = {
      .sType = VK_STRUCTURE_TYPE_VALIDATION_FEATURES_EXT,
      .enabledValidationFeatureCount = 3,
      .pEnabledValidationFeatures = enabledFeatures
    };

    VkInstanceCreateInfo createInfo = {
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
      .enabledExtensionCount = glfwExtensionCount,
      .ppEnabledExtensionNames = glfwExtensions
    };
    vkCreateInstance(&createInfo, nullptr, &instance);
    glfwCreateWindowSurface(instance, window, nullptr, &surface);


    findAdequateHardware();

    createLogicalDevice();

    swapchain = VK_NULL_HANDLE;
    createSwapchain();
    destructQueue.emplace_front([=] {
      vkDestroySwapchainKHR(device, swapchain, nullptr);
      for(int i = 0; i < swapchainImageViews.size(); i++) {
        vkDestroyImageView(device, swapchainImageViews[i], nullptr);
        vkDestroyImageView(device, computeImageViews[i], nullptr);
        vkFreeMemory(device, computeImageMemories[i], nullptr);
        vkDestroyImage(device, computeImages[i], nullptr);
      }
    });
    
    createSignalling();

    createComputePipeline();

    createCommandBuffers();

    currentFrame = 0;

    for(int i = 0; i < neuronSize[0]*neuronSize[1]; i++) {
      neuronBufferMap[i] = 0.5;//(float)rand()/(float)RAND_MAX;
    };
    for(int i = 0; i < neuronSize[0]*neuronSize[1]*8; i++) {
      weightBufferMap[i] = sin((float)i);//(float)rand()/(float)(RAND_MAX/2)-1.0;
    }
    *uboMap = {
      .resolution = glm::ivec2(swapchainExtent.width, swapchainExtent.height),
      .networkDimensions = glm::ivec2(neuronSize[0], neuronSize[1]),
      .zoom = glm::vec4(0, 0, 10, 0)
    };
    VkMappedMemoryRange mappedMemoryRanges[] = {
      {
        .sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE,
        .memory = uboMemory,
        .offset = 0,
        .size = VK_WHOLE_SIZE
      },
      {
        .sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE,
        .memory = neuronBufferMemory,
        .offset = 0,
        .size = VK_WHOLE_SIZE
      }
    };
    vkFlushMappedMemoryRanges(device, 2, mappedMemoryRanges);

    glfwSetFramebufferSizeCallback(window, [](GLFWwindow* window, int width, int height){
        ((NeuronApplication*)glfwGetWindowUserPointer(window))->framebufferResized = true;
    });
  }

  ~NeuronApplication() {
    vkDeviceWaitIdle(device);

    for(const std::function<void()>& destructor : destructQueue) {
      destructor();
    }

    glfwTerminate();
  }

  void run() {
    while(!glfwWindowShouldClose(window)) {
        glfwPollEvents();
        drawFrame();
    }
  }
};


int main() {
  NeuronApplication app;
  app.run();


  return 0;
}
