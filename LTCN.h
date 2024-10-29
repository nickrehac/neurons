//
// Created by nick on 10/16/24.
//

#ifndef LTCN_H
#define LTCN_H

#include <deque>
#include <functional>
#include <optional>
#include <vulkan/vulkan.h>

namespace LTCN {
    typedef std::vector<std::pair<size_t,size_t>> Bounds;

    namespace VU {
        class Instance;
        class Buffer {
            VkPhysicalDevice physicalDevice;
            VkDevice device = VK_NULL_HANDLE;
            VkBuffer buffer = VK_NULL_HANDLE;
            VkDeviceMemory deviceMemory = VK_NULL_HANDLE;
            size_t size;
            std::optional<void *> data;
        public:
            Buffer(VkPhysicalDevice physicalDevice, VkDevice device, size_t size);
            Buffer(Instance * instance, size_t size);
            ~Buffer();

            void * map();
            void * map(size_t offset, size_t size);
            void unmap();
            void flush();
        };

        class Instance {
            VkInstance instance = VK_NULL_HANDLE;
            VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
            VkDevice device = VK_NULL_HANDLE;
            uint32_t computeFamilyIndex = 0;
            VkQueue computeQueue = VK_NULL_HANDLE;
            VkCommandPool commandPool = VK_NULL_HANDLE;
            VkCommandBuffer commandBuffer = VK_NULL_HANDLE;


        public:
            Instance();
            ~Instance();

            VkInstance getInstance();
            VkPhysicalDevice getPhysicalDevice();
            VkDevice getDevice();

            Buffer createBuffer(size_t size);
        };
    }
    class Rect {
        int x;
        int y;
        int width;
        int height;

        Rect(int x, int y, int width, int height);
        Rect(int width, int height);
    };
    enum class LayerType {
        Linear = 0,
        FullyConnected = 1,
        Convolution = 2,
    };
    class NetworkTensor {
        unsigned int id;
        std::vector<size_t> dimensions;
        VU::Buffer * buffer;
        bool visited = false;
    public:
        NetworkTensor(VU::Instance * instance, unsigned int id, std::vector<size_t> shape);
        ~NetworkTensor();
    };
    class Layer {
        unsigned int id;
        LayerType type;
        NetworkTensor * input;
        Bounds inputWindow;
        NetworkTensor * output;
        Bounds outputWindow;
        VU::Buffer * weights;
        Bounds convolutionInWindow;
        Bounds convolutionOutWindow;

    public:
        Layer(VU::Instance * instance, Bounds inWin, Bounds outWin, LayerType type);

        void bindBuffers();
        void writeCommands();
    };
    class LTCN {
        //TODO: import/export to file
        //TODO:

        VU::Instance instance;

        std::vector<Layer> layers;
        std::vector<NetworkTensor> tensors;

        std::deque<std::function<void()>> destructQueue;

    public:
        LTCN();
        LTCN(const char * filename);

        ~LTCN();

        void addTensor(unsigned int id, std::vector<size_t> shape);
        void addLayer(unsigned int id, unsigned int from, unsigned int to, LayerType type);

        void * evaluate(void * input, float timestep);

        void train(void * input, void * output, float strength, float timestep);
    };


}





#endif //LTCN_H
