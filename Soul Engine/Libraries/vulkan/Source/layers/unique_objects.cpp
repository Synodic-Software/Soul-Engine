/* THIS FILE IS GENERATED.  DO NOT EDIT. */

/*
 * Copyright (c) 2015-2016 The Khronos Group Inc.
 * Copyright (c) 2015-2016 Valve Corporation
 * Copyright (c) 2015-2016 LunarG, Inc.
 * Copyright (c) 2015-2016 Google, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Author: Tobin Ehlis <tobine@google.com>
 * Author: Courtney Goeltzenleuchter <courtneygo@google.com>
 * Author: Jon Ashburn <jon@lunarg.com>
 * Author: Mark Lobodzinski <mark@lunarg.com>
 * Author: Mike Stroyan <stroyan@google.com>
 * Author: Tony Barbour <tony@LunarG.com>
 */

// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1360
#include "unique_objects.h"

VK_LAYER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkCreateInstance(const VkInstanceCreateInfo* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkInstance* pInstance)
{
    return explicit_CreateInstance(pCreateInfo, pAllocator, pInstance);
}

VK_LAYER_EXPORT VKAPI_ATTR void VKAPI_CALL vkDestroyInstance(VkInstance instance, const VkAllocationCallbacks* pAllocator)
{
    return explicit_DestroyInstance(instance, pAllocator);
}

VK_LAYER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkCreateDevice(VkPhysicalDevice physicalDevice, const VkDeviceCreateInfo* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkDevice* pDevice)
{
    return explicit_CreateDevice(physicalDevice, pCreateInfo, pAllocator, pDevice);
}

VK_LAYER_EXPORT VKAPI_ATTR void VKAPI_CALL vkDestroyDevice(VkDevice device, const VkAllocationCallbacks* pAllocator)
{
    return explicit_DestroyDevice(device, pAllocator);
}

// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #353

VK_LAYER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkEnumerateInstanceExtensionProperties(const char *pLayerName, uint32_t *pCount,  VkExtensionProperties* pProperties)
{
    return util_GetExtensionProperties(0, NULL, pCount, pProperties);
}

// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #376
static const VkLayerProperties globalLayerProps[] = {
    {
        "VK_LAYER_GOOGLE_unique_objects",
        VK_LAYER_API_VERSION, // specVersion
        1, // implementationVersion
        "Google Validation Layer"
    }
};

// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #392

VK_LAYER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkEnumerateInstanceLayerProperties(uint32_t *pCount,  VkLayerProperties* pProperties)
{
    return util_GetLayerProperties(ARRAY_SIZE(globalLayerProps), globalLayerProps, pCount, pProperties);
}

// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #402
static const VkLayerProperties deviceLayerProps[] = {
    {
        "VK_LAYER_GOOGLE_unique_objects",
        VK_LAYER_API_VERSION, // specVersion
        1, // implementationVersion
        "Google Validation Layer"
    }
};
VK_LAYER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkEnumerateDeviceLayerProperties(VkPhysicalDevice physicalDevice, uint32_t *pCount, VkLayerProperties* pProperties)
{
    return util_GetLayerProperties(ARRAY_SIZE(deviceLayerProps), deviceLayerProps, pCount, pProperties);
}


VK_LAYER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkQueueSubmit(VkQueue queue, uint32_t submitCount, const VkSubmitInfo* pSubmits, VkFence fence)
{
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1490
    layer_data *my_map_data = get_my_data_ptr(get_dispatch_key(queue), layer_data_map);
// STRUCT USES:['fence', 'pSubmits[submitCount]']
//LOCAL DECLS:['pSubmits']
    safe_VkSubmitInfo* local_pSubmits = NULL;
    {
    std::lock_guard<std::mutex> lock(global_lock);
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1450
    fence = (VkFence)my_map_data->unique_id_mapping[reinterpret_cast<uint64_t &>(fence)];
    if (pSubmits) {
        local_pSubmits = new safe_VkSubmitInfo[submitCount];
        for (uint32_t idx0=0; idx0<submitCount; ++idx0) {
            local_pSubmits[idx0].initialize(&pSubmits[idx0]);
            if (local_pSubmits[idx0].pSignalSemaphores) {
                for (uint32_t idx1=0; idx1<pSubmits[idx0].signalSemaphoreCount; ++idx1) {
                    local_pSubmits[idx0].pSignalSemaphores[idx1] = (VkSemaphore)my_map_data->unique_id_mapping[reinterpret_cast<const uint64_t &>(pSubmits[idx0].pSignalSemaphores[idx1])];
                }
            }
            if (local_pSubmits[idx0].pWaitSemaphores) {
                for (uint32_t idx2=0; idx2<pSubmits[idx0].waitSemaphoreCount; ++idx2) {
                    local_pSubmits[idx0].pWaitSemaphores[idx2] = (VkSemaphore)my_map_data->unique_id_mapping[reinterpret_cast<const uint64_t &>(pSubmits[idx0].pWaitSemaphores[idx2])];
                }
            }
        }
    }
    }
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1596
    VkResult result = get_dispatch_table(unique_objects_device_table_map, queue)->QueueSubmit(queue, submitCount, (const VkSubmitInfo*)local_pSubmits, fence);
    if (local_pSubmits)
        delete[] local_pSubmits;
    return result;
}


VK_LAYER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkAllocateMemory(VkDevice device, const VkMemoryAllocateInfo* pAllocateInfo, const VkAllocationCallbacks* pAllocator, VkDeviceMemory* pMemory)
{
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1490
    layer_data *my_map_data = get_my_data_ptr(get_dispatch_key(device), layer_data_map);
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1596
    VkResult result = get_dispatch_table(unique_objects_device_table_map, device)->AllocateMemory(device, pAllocateInfo, pAllocator, pMemory);
    if (VK_SUCCESS == result) {
        std::lock_guard<std::mutex> lock(global_lock);
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1567
        uint64_t unique_id = global_unique_id++;
        my_map_data->unique_id_mapping[unique_id] = reinterpret_cast<uint64_t &>(*pMemory);
        *pMemory = reinterpret_cast<VkDeviceMemory&>(unique_id);
    }
    return result;
}


VK_LAYER_EXPORT VKAPI_ATTR void VKAPI_CALL vkFreeMemory(VkDevice device, VkDeviceMemory memory, const VkAllocationCallbacks* pAllocator)
{
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1490
    layer_data *my_map_data = get_my_data_ptr(get_dispatch_key(device), layer_data_map);
// STRUCT USES:['memory']
    std::unique_lock<std::mutex> lock(global_lock);
    uint64_t local_memory = reinterpret_cast<uint64_t &>(memory);
    memory = (VkDeviceMemory)my_map_data->unique_id_mapping[local_memory];
    lock.unlock();
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1596
    get_dispatch_table(unique_objects_device_table_map, device)->FreeMemory(device, memory, pAllocator);
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1584
    lock.lock();
    my_map_data->unique_id_mapping.erase(local_memory);
}


VK_LAYER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkMapMemory(VkDevice device, VkDeviceMemory memory, VkDeviceSize offset, VkDeviceSize size, VkMemoryMapFlags flags, void** ppData)
{
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1490
    layer_data *my_map_data = get_my_data_ptr(get_dispatch_key(device), layer_data_map);
// STRUCT USES:['memory']
    {
    std::lock_guard<std::mutex> lock(global_lock);
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1450
    memory = (VkDeviceMemory)my_map_data->unique_id_mapping[reinterpret_cast<uint64_t &>(memory)];
    }
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1596
    VkResult result = get_dispatch_table(unique_objects_device_table_map, device)->MapMemory(device, memory, offset, size, flags, ppData);
    return result;
}


VK_LAYER_EXPORT VKAPI_ATTR void VKAPI_CALL vkUnmapMemory(VkDevice device, VkDeviceMemory memory)
{
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1490
    layer_data *my_map_data = get_my_data_ptr(get_dispatch_key(device), layer_data_map);
// STRUCT USES:['memory']
    {
    std::lock_guard<std::mutex> lock(global_lock);
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1450
    memory = (VkDeviceMemory)my_map_data->unique_id_mapping[reinterpret_cast<uint64_t &>(memory)];
    }
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1596
    get_dispatch_table(unique_objects_device_table_map, device)->UnmapMemory(device, memory);
}


VK_LAYER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkFlushMappedMemoryRanges(VkDevice device, uint32_t memoryRangeCount, const VkMappedMemoryRange* pMemoryRanges)
{
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1490
    layer_data *my_map_data = get_my_data_ptr(get_dispatch_key(device), layer_data_map);
// STRUCT USES:['pMemoryRanges[memoryRangeCount]']
//LOCAL DECLS:['pMemoryRanges']
    safe_VkMappedMemoryRange* local_pMemoryRanges = NULL;
    {
    std::lock_guard<std::mutex> lock(global_lock);
    if (pMemoryRanges) {
        local_pMemoryRanges = new safe_VkMappedMemoryRange[memoryRangeCount];
        for (uint32_t idx0=0; idx0<memoryRangeCount; ++idx0) {
            local_pMemoryRanges[idx0].initialize(&pMemoryRanges[idx0]);
            if (pMemoryRanges[idx0].memory) {
                local_pMemoryRanges[idx0].memory = (VkDeviceMemory)my_map_data->unique_id_mapping[reinterpret_cast<const uint64_t &>(pMemoryRanges[idx0].memory)];
            }
        }
    }
    }
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1596
    VkResult result = get_dispatch_table(unique_objects_device_table_map, device)->FlushMappedMemoryRanges(device, memoryRangeCount, (const VkMappedMemoryRange*)local_pMemoryRanges);
    if (local_pMemoryRanges)
        delete[] local_pMemoryRanges;
    return result;
}


VK_LAYER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkInvalidateMappedMemoryRanges(VkDevice device, uint32_t memoryRangeCount, const VkMappedMemoryRange* pMemoryRanges)
{
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1490
    layer_data *my_map_data = get_my_data_ptr(get_dispatch_key(device), layer_data_map);
// STRUCT USES:['pMemoryRanges[memoryRangeCount]']
//LOCAL DECLS:['pMemoryRanges']
    safe_VkMappedMemoryRange* local_pMemoryRanges = NULL;
    {
    std::lock_guard<std::mutex> lock(global_lock);
    if (pMemoryRanges) {
        local_pMemoryRanges = new safe_VkMappedMemoryRange[memoryRangeCount];
        for (uint32_t idx0=0; idx0<memoryRangeCount; ++idx0) {
            local_pMemoryRanges[idx0].initialize(&pMemoryRanges[idx0]);
            if (pMemoryRanges[idx0].memory) {
                local_pMemoryRanges[idx0].memory = (VkDeviceMemory)my_map_data->unique_id_mapping[reinterpret_cast<const uint64_t &>(pMemoryRanges[idx0].memory)];
            }
        }
    }
    }
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1596
    VkResult result = get_dispatch_table(unique_objects_device_table_map, device)->InvalidateMappedMemoryRanges(device, memoryRangeCount, (const VkMappedMemoryRange*)local_pMemoryRanges);
    if (local_pMemoryRanges)
        delete[] local_pMemoryRanges;
    return result;
}


VK_LAYER_EXPORT VKAPI_ATTR void VKAPI_CALL vkGetDeviceMemoryCommitment(VkDevice device, VkDeviceMemory memory, VkDeviceSize* pCommittedMemoryInBytes)
{
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1490
    layer_data *my_map_data = get_my_data_ptr(get_dispatch_key(device), layer_data_map);
// STRUCT USES:['memory']
    {
    std::lock_guard<std::mutex> lock(global_lock);
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1450
    memory = (VkDeviceMemory)my_map_data->unique_id_mapping[reinterpret_cast<uint64_t &>(memory)];
    }
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1596
    get_dispatch_table(unique_objects_device_table_map, device)->GetDeviceMemoryCommitment(device, memory, pCommittedMemoryInBytes);
}


VK_LAYER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkBindBufferMemory(VkDevice device, VkBuffer buffer, VkDeviceMemory memory, VkDeviceSize memoryOffset)
{
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1490
    layer_data *my_map_data = get_my_data_ptr(get_dispatch_key(device), layer_data_map);
// STRUCT USES:['buffer', 'memory']
    {
    std::lock_guard<std::mutex> lock(global_lock);
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1450
    buffer = (VkBuffer)my_map_data->unique_id_mapping[reinterpret_cast<uint64_t &>(buffer)];
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1450
    memory = (VkDeviceMemory)my_map_data->unique_id_mapping[reinterpret_cast<uint64_t &>(memory)];
    }
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1596
    VkResult result = get_dispatch_table(unique_objects_device_table_map, device)->BindBufferMemory(device, buffer, memory, memoryOffset);
    return result;
}


VK_LAYER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkBindImageMemory(VkDevice device, VkImage image, VkDeviceMemory memory, VkDeviceSize memoryOffset)
{
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1490
    layer_data *my_map_data = get_my_data_ptr(get_dispatch_key(device), layer_data_map);
// STRUCT USES:['image', 'memory']
    {
    std::lock_guard<std::mutex> lock(global_lock);
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1450
    image = (VkImage)my_map_data->unique_id_mapping[reinterpret_cast<uint64_t &>(image)];
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1450
    memory = (VkDeviceMemory)my_map_data->unique_id_mapping[reinterpret_cast<uint64_t &>(memory)];
    }
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1596
    VkResult result = get_dispatch_table(unique_objects_device_table_map, device)->BindImageMemory(device, image, memory, memoryOffset);
    return result;
}


VK_LAYER_EXPORT VKAPI_ATTR void VKAPI_CALL vkGetBufferMemoryRequirements(VkDevice device, VkBuffer buffer, VkMemoryRequirements* pMemoryRequirements)
{
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1490
    layer_data *my_map_data = get_my_data_ptr(get_dispatch_key(device), layer_data_map);
// STRUCT USES:['buffer']
    {
    std::lock_guard<std::mutex> lock(global_lock);
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1450
    buffer = (VkBuffer)my_map_data->unique_id_mapping[reinterpret_cast<uint64_t &>(buffer)];
    }
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1596
    get_dispatch_table(unique_objects_device_table_map, device)->GetBufferMemoryRequirements(device, buffer, pMemoryRequirements);
}


VK_LAYER_EXPORT VKAPI_ATTR void VKAPI_CALL vkGetImageMemoryRequirements(VkDevice device, VkImage image, VkMemoryRequirements* pMemoryRequirements)
{
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1490
    layer_data *my_map_data = get_my_data_ptr(get_dispatch_key(device), layer_data_map);
// STRUCT USES:['image']
    {
    std::lock_guard<std::mutex> lock(global_lock);
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1450
    image = (VkImage)my_map_data->unique_id_mapping[reinterpret_cast<uint64_t &>(image)];
    }
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1596
    get_dispatch_table(unique_objects_device_table_map, device)->GetImageMemoryRequirements(device, image, pMemoryRequirements);
}


VK_LAYER_EXPORT VKAPI_ATTR void VKAPI_CALL vkGetImageSparseMemoryRequirements(VkDevice device, VkImage image, uint32_t* pSparseMemoryRequirementCount, VkSparseImageMemoryRequirements* pSparseMemoryRequirements)
{
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1490
    layer_data *my_map_data = get_my_data_ptr(get_dispatch_key(device), layer_data_map);
// STRUCT USES:['image']
    {
    std::lock_guard<std::mutex> lock(global_lock);
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1450
    image = (VkImage)my_map_data->unique_id_mapping[reinterpret_cast<uint64_t &>(image)];
    }
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1596
    get_dispatch_table(unique_objects_device_table_map, device)->GetImageSparseMemoryRequirements(device, image, pSparseMemoryRequirementCount, pSparseMemoryRequirements);
}


VK_LAYER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkQueueBindSparse(VkQueue queue, uint32_t bindInfoCount, const VkBindSparseInfo* pBindInfo, VkFence fence)
{
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1490
    layer_data *my_map_data = get_my_data_ptr(get_dispatch_key(queue), layer_data_map);
// STRUCT USES:['fence', 'pBindInfo[bindInfoCount]']
//LOCAL DECLS:['pBindInfo']
    safe_VkBindSparseInfo* local_pBindInfo = NULL;
    {
    std::lock_guard<std::mutex> lock(global_lock);
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1450
    fence = (VkFence)my_map_data->unique_id_mapping[reinterpret_cast<uint64_t &>(fence)];
    if (pBindInfo) {
        local_pBindInfo = new safe_VkBindSparseInfo[bindInfoCount];
        for (uint32_t idx0=0; idx0<bindInfoCount; ++idx0) {
            local_pBindInfo[idx0].initialize(&pBindInfo[idx0]);
            if (local_pBindInfo[idx0].pBufferBinds) {
                for (uint32_t idx1=0; idx1<pBindInfo[idx0].bufferBindCount; ++idx1) {
                    if (pBindInfo[idx0].pBufferBinds[idx1].buffer) {
                        local_pBindInfo[idx0].pBufferBinds[idx1].buffer = (VkBuffer)my_map_data->unique_id_mapping[reinterpret_cast<const uint64_t &>(pBindInfo[idx0].pBufferBinds[idx1].buffer)];
                    }
                    if (local_pBindInfo[idx0].pBufferBinds[idx1].pBinds) {
                        for (uint32_t idx2=0; idx2<pBindInfo[idx0].pBufferBinds[idx1].bindCount; ++idx2) {
                            if (pBindInfo[idx0].pBufferBinds[idx1].pBinds[idx2].memory) {
                                local_pBindInfo[idx0].pBufferBinds[idx1].pBinds[idx2].memory = (VkDeviceMemory)my_map_data->unique_id_mapping[reinterpret_cast<const uint64_t &>(pBindInfo[idx0].pBufferBinds[idx1].pBinds[idx2].memory)];
                            }
                        }
                    }
                }
            }
            if (local_pBindInfo[idx0].pImageBinds) {
                for (uint32_t idx2=0; idx2<pBindInfo[idx0].imageBindCount; ++idx2) {
                    if (pBindInfo[idx0].pImageBinds[idx2].image) {
                        local_pBindInfo[idx0].pImageBinds[idx2].image = (VkImage)my_map_data->unique_id_mapping[reinterpret_cast<const uint64_t &>(pBindInfo[idx0].pImageBinds[idx2].image)];
                    }
                    if (local_pBindInfo[idx0].pImageBinds[idx2].pBinds) {
                        for (uint32_t idx3=0; idx3<pBindInfo[idx0].pImageBinds[idx2].bindCount; ++idx3) {
                            if (pBindInfo[idx0].pImageBinds[idx2].pBinds[idx3].memory) {
                                local_pBindInfo[idx0].pImageBinds[idx2].pBinds[idx3].memory = (VkDeviceMemory)my_map_data->unique_id_mapping[reinterpret_cast<const uint64_t &>(pBindInfo[idx0].pImageBinds[idx2].pBinds[idx3].memory)];
                            }
                        }
                    }
                }
            }
            if (local_pBindInfo[idx0].pImageOpaqueBinds) {
                for (uint32_t idx3=0; idx3<pBindInfo[idx0].imageOpaqueBindCount; ++idx3) {
                    if (pBindInfo[idx0].pImageOpaqueBinds[idx3].image) {
                        local_pBindInfo[idx0].pImageOpaqueBinds[idx3].image = (VkImage)my_map_data->unique_id_mapping[reinterpret_cast<const uint64_t &>(pBindInfo[idx0].pImageOpaqueBinds[idx3].image)];
                    }
                    if (local_pBindInfo[idx0].pImageOpaqueBinds[idx3].pBinds) {
                        for (uint32_t idx4=0; idx4<pBindInfo[idx0].pImageOpaqueBinds[idx3].bindCount; ++idx4) {
                            if (pBindInfo[idx0].pImageOpaqueBinds[idx3].pBinds[idx4].memory) {
                                local_pBindInfo[idx0].pImageOpaqueBinds[idx3].pBinds[idx4].memory = (VkDeviceMemory)my_map_data->unique_id_mapping[reinterpret_cast<const uint64_t &>(pBindInfo[idx0].pImageOpaqueBinds[idx3].pBinds[idx4].memory)];
                            }
                        }
                    }
                }
            }
            if (local_pBindInfo[idx0].pSignalSemaphores) {
                for (uint32_t idx4=0; idx4<pBindInfo[idx0].signalSemaphoreCount; ++idx4) {
                    local_pBindInfo[idx0].pSignalSemaphores[idx4] = (VkSemaphore)my_map_data->unique_id_mapping[reinterpret_cast<const uint64_t &>(pBindInfo[idx0].pSignalSemaphores[idx4])];
                }
            }
            if (local_pBindInfo[idx0].pWaitSemaphores) {
                for (uint32_t idx5=0; idx5<pBindInfo[idx0].waitSemaphoreCount; ++idx5) {
                    local_pBindInfo[idx0].pWaitSemaphores[idx5] = (VkSemaphore)my_map_data->unique_id_mapping[reinterpret_cast<const uint64_t &>(pBindInfo[idx0].pWaitSemaphores[idx5])];
                }
            }
        }
    }
    }
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1596
    VkResult result = get_dispatch_table(unique_objects_device_table_map, queue)->QueueBindSparse(queue, bindInfoCount, (const VkBindSparseInfo*)local_pBindInfo, fence);
    if (local_pBindInfo)
        delete[] local_pBindInfo;
    return result;
}


VK_LAYER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkCreateFence(VkDevice device, const VkFenceCreateInfo* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkFence* pFence)
{
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1490
    layer_data *my_map_data = get_my_data_ptr(get_dispatch_key(device), layer_data_map);
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1596
    VkResult result = get_dispatch_table(unique_objects_device_table_map, device)->CreateFence(device, pCreateInfo, pAllocator, pFence);
    if (VK_SUCCESS == result) {
        std::lock_guard<std::mutex> lock(global_lock);
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1567
        uint64_t unique_id = global_unique_id++;
        my_map_data->unique_id_mapping[unique_id] = reinterpret_cast<uint64_t &>(*pFence);
        *pFence = reinterpret_cast<VkFence&>(unique_id);
    }
    return result;
}


VK_LAYER_EXPORT VKAPI_ATTR void VKAPI_CALL vkDestroyFence(VkDevice device, VkFence fence, const VkAllocationCallbacks* pAllocator)
{
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1490
    layer_data *my_map_data = get_my_data_ptr(get_dispatch_key(device), layer_data_map);
// STRUCT USES:['fence']
    std::unique_lock<std::mutex> lock(global_lock);
    uint64_t local_fence = reinterpret_cast<uint64_t &>(fence);
    fence = (VkFence)my_map_data->unique_id_mapping[local_fence];
    lock.unlock();
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1596
    get_dispatch_table(unique_objects_device_table_map, device)->DestroyFence(device, fence, pAllocator);
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1584
    lock.lock();
    my_map_data->unique_id_mapping.erase(local_fence);
}


VK_LAYER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkResetFences(VkDevice device, uint32_t fenceCount, const VkFence* pFences)
{
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1490
    layer_data *my_map_data = get_my_data_ptr(get_dispatch_key(device), layer_data_map);
// STRUCT USES:['pFences[fenceCount]']
//LOCAL DECLS:['pFences']
    VkFence* local_pFences = NULL;
    {
    std::lock_guard<std::mutex> lock(global_lock);
    if (pFences) {
        local_pFences = new VkFence[fenceCount];
        for (uint32_t idx0=0; idx0<fenceCount; ++idx0) {
            local_pFences[idx0] = (VkFence)my_map_data->unique_id_mapping[reinterpret_cast<const uint64_t &>(pFences[idx0])];
        }
    }
    }
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1596
    VkResult result = get_dispatch_table(unique_objects_device_table_map, device)->ResetFences(device, fenceCount, (const VkFence*)local_pFences);
    if (local_pFences)
        delete[] local_pFences;
    return result;
}


VK_LAYER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkGetFenceStatus(VkDevice device, VkFence fence)
{
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1490
    layer_data *my_map_data = get_my_data_ptr(get_dispatch_key(device), layer_data_map);
// STRUCT USES:['fence']
    {
    std::lock_guard<std::mutex> lock(global_lock);
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1450
    fence = (VkFence)my_map_data->unique_id_mapping[reinterpret_cast<uint64_t &>(fence)];
    }
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1596
    VkResult result = get_dispatch_table(unique_objects_device_table_map, device)->GetFenceStatus(device, fence);
    return result;
}


VK_LAYER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkWaitForFences(VkDevice device, uint32_t fenceCount, const VkFence* pFences, VkBool32 waitAll, uint64_t timeout)
{
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1490
    layer_data *my_map_data = get_my_data_ptr(get_dispatch_key(device), layer_data_map);
// STRUCT USES:['pFences[fenceCount]']
//LOCAL DECLS:['pFences']
    VkFence* local_pFences = NULL;
    {
    std::lock_guard<std::mutex> lock(global_lock);
    if (pFences) {
        local_pFences = new VkFence[fenceCount];
        for (uint32_t idx0=0; idx0<fenceCount; ++idx0) {
            local_pFences[idx0] = (VkFence)my_map_data->unique_id_mapping[reinterpret_cast<const uint64_t &>(pFences[idx0])];
        }
    }
    }
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1596
    VkResult result = get_dispatch_table(unique_objects_device_table_map, device)->WaitForFences(device, fenceCount, (const VkFence*)local_pFences, waitAll, timeout);
    if (local_pFences)
        delete[] local_pFences;
    return result;
}


VK_LAYER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkCreateSemaphore(VkDevice device, const VkSemaphoreCreateInfo* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkSemaphore* pSemaphore)
{
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1490
    layer_data *my_map_data = get_my_data_ptr(get_dispatch_key(device), layer_data_map);
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1596
    VkResult result = get_dispatch_table(unique_objects_device_table_map, device)->CreateSemaphore(device, pCreateInfo, pAllocator, pSemaphore);
    if (VK_SUCCESS == result) {
        std::lock_guard<std::mutex> lock(global_lock);
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1567
        uint64_t unique_id = global_unique_id++;
        my_map_data->unique_id_mapping[unique_id] = reinterpret_cast<uint64_t &>(*pSemaphore);
        *pSemaphore = reinterpret_cast<VkSemaphore&>(unique_id);
    }
    return result;
}


VK_LAYER_EXPORT VKAPI_ATTR void VKAPI_CALL vkDestroySemaphore(VkDevice device, VkSemaphore semaphore, const VkAllocationCallbacks* pAllocator)
{
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1490
    layer_data *my_map_data = get_my_data_ptr(get_dispatch_key(device), layer_data_map);
// STRUCT USES:['semaphore']
    std::unique_lock<std::mutex> lock(global_lock);
    uint64_t local_semaphore = reinterpret_cast<uint64_t &>(semaphore);
    semaphore = (VkSemaphore)my_map_data->unique_id_mapping[local_semaphore];
    lock.unlock();
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1596
    get_dispatch_table(unique_objects_device_table_map, device)->DestroySemaphore(device, semaphore, pAllocator);
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1584
    lock.lock();
    my_map_data->unique_id_mapping.erase(local_semaphore);
}


VK_LAYER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkCreateEvent(VkDevice device, const VkEventCreateInfo* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkEvent* pEvent)
{
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1490
    layer_data *my_map_data = get_my_data_ptr(get_dispatch_key(device), layer_data_map);
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1596
    VkResult result = get_dispatch_table(unique_objects_device_table_map, device)->CreateEvent(device, pCreateInfo, pAllocator, pEvent);
    if (VK_SUCCESS == result) {
        std::lock_guard<std::mutex> lock(global_lock);
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1567
        uint64_t unique_id = global_unique_id++;
        my_map_data->unique_id_mapping[unique_id] = reinterpret_cast<uint64_t &>(*pEvent);
        *pEvent = reinterpret_cast<VkEvent&>(unique_id);
    }
    return result;
}


VK_LAYER_EXPORT VKAPI_ATTR void VKAPI_CALL vkDestroyEvent(VkDevice device, VkEvent event, const VkAllocationCallbacks* pAllocator)
{
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1490
    layer_data *my_map_data = get_my_data_ptr(get_dispatch_key(device), layer_data_map);
// STRUCT USES:['event']
    std::unique_lock<std::mutex> lock(global_lock);
    uint64_t local_event = reinterpret_cast<uint64_t &>(event);
    event = (VkEvent)my_map_data->unique_id_mapping[local_event];
    lock.unlock();
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1596
    get_dispatch_table(unique_objects_device_table_map, device)->DestroyEvent(device, event, pAllocator);
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1584
    lock.lock();
    my_map_data->unique_id_mapping.erase(local_event);
}


VK_LAYER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkGetEventStatus(VkDevice device, VkEvent event)
{
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1490
    layer_data *my_map_data = get_my_data_ptr(get_dispatch_key(device), layer_data_map);
// STRUCT USES:['event']
    {
    std::lock_guard<std::mutex> lock(global_lock);
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1450
    event = (VkEvent)my_map_data->unique_id_mapping[reinterpret_cast<uint64_t &>(event)];
    }
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1596
    VkResult result = get_dispatch_table(unique_objects_device_table_map, device)->GetEventStatus(device, event);
    return result;
}


VK_LAYER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkSetEvent(VkDevice device, VkEvent event)
{
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1490
    layer_data *my_map_data = get_my_data_ptr(get_dispatch_key(device), layer_data_map);
// STRUCT USES:['event']
    {
    std::lock_guard<std::mutex> lock(global_lock);
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1450
    event = (VkEvent)my_map_data->unique_id_mapping[reinterpret_cast<uint64_t &>(event)];
    }
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1596
    VkResult result = get_dispatch_table(unique_objects_device_table_map, device)->SetEvent(device, event);
    return result;
}


VK_LAYER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkResetEvent(VkDevice device, VkEvent event)
{
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1490
    layer_data *my_map_data = get_my_data_ptr(get_dispatch_key(device), layer_data_map);
// STRUCT USES:['event']
    {
    std::lock_guard<std::mutex> lock(global_lock);
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1450
    event = (VkEvent)my_map_data->unique_id_mapping[reinterpret_cast<uint64_t &>(event)];
    }
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1596
    VkResult result = get_dispatch_table(unique_objects_device_table_map, device)->ResetEvent(device, event);
    return result;
}


VK_LAYER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkCreateQueryPool(VkDevice device, const VkQueryPoolCreateInfo* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkQueryPool* pQueryPool)
{
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1490
    layer_data *my_map_data = get_my_data_ptr(get_dispatch_key(device), layer_data_map);
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1596
    VkResult result = get_dispatch_table(unique_objects_device_table_map, device)->CreateQueryPool(device, pCreateInfo, pAllocator, pQueryPool);
    if (VK_SUCCESS == result) {
        std::lock_guard<std::mutex> lock(global_lock);
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1567
        uint64_t unique_id = global_unique_id++;
        my_map_data->unique_id_mapping[unique_id] = reinterpret_cast<uint64_t &>(*pQueryPool);
        *pQueryPool = reinterpret_cast<VkQueryPool&>(unique_id);
    }
    return result;
}


VK_LAYER_EXPORT VKAPI_ATTR void VKAPI_CALL vkDestroyQueryPool(VkDevice device, VkQueryPool queryPool, const VkAllocationCallbacks* pAllocator)
{
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1490
    layer_data *my_map_data = get_my_data_ptr(get_dispatch_key(device), layer_data_map);
// STRUCT USES:['queryPool']
    std::unique_lock<std::mutex> lock(global_lock);
    uint64_t local_queryPool = reinterpret_cast<uint64_t &>(queryPool);
    queryPool = (VkQueryPool)my_map_data->unique_id_mapping[local_queryPool];
    lock.unlock();
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1596
    get_dispatch_table(unique_objects_device_table_map, device)->DestroyQueryPool(device, queryPool, pAllocator);
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1584
    lock.lock();
    my_map_data->unique_id_mapping.erase(local_queryPool);
}


VK_LAYER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkGetQueryPoolResults(VkDevice device, VkQueryPool queryPool, uint32_t firstQuery, uint32_t queryCount, size_t dataSize, void* pData, VkDeviceSize stride, VkQueryResultFlags flags)
{
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1490
    layer_data *my_map_data = get_my_data_ptr(get_dispatch_key(device), layer_data_map);
// STRUCT USES:['queryPool']
    {
    std::lock_guard<std::mutex> lock(global_lock);
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1450
    queryPool = (VkQueryPool)my_map_data->unique_id_mapping[reinterpret_cast<uint64_t &>(queryPool)];
    }
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1596
    VkResult result = get_dispatch_table(unique_objects_device_table_map, device)->GetQueryPoolResults(device, queryPool, firstQuery, queryCount, dataSize, pData, stride, flags);
    return result;
}


VK_LAYER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkCreateBuffer(VkDevice device, const VkBufferCreateInfo* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkBuffer* pBuffer)
{
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1490
    layer_data *my_map_data = get_my_data_ptr(get_dispatch_key(device), layer_data_map);
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1596
    VkResult result = get_dispatch_table(unique_objects_device_table_map, device)->CreateBuffer(device, pCreateInfo, pAllocator, pBuffer);
    if (VK_SUCCESS == result) {
        std::lock_guard<std::mutex> lock(global_lock);
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1567
        uint64_t unique_id = global_unique_id++;
        my_map_data->unique_id_mapping[unique_id] = reinterpret_cast<uint64_t &>(*pBuffer);
        *pBuffer = reinterpret_cast<VkBuffer&>(unique_id);
    }
    return result;
}


VK_LAYER_EXPORT VKAPI_ATTR void VKAPI_CALL vkDestroyBuffer(VkDevice device, VkBuffer buffer, const VkAllocationCallbacks* pAllocator)
{
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1490
    layer_data *my_map_data = get_my_data_ptr(get_dispatch_key(device), layer_data_map);
// STRUCT USES:['buffer']
    std::unique_lock<std::mutex> lock(global_lock);
    uint64_t local_buffer = reinterpret_cast<uint64_t &>(buffer);
    buffer = (VkBuffer)my_map_data->unique_id_mapping[local_buffer];
    lock.unlock();
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1596
    get_dispatch_table(unique_objects_device_table_map, device)->DestroyBuffer(device, buffer, pAllocator);
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1584
    lock.lock();
    my_map_data->unique_id_mapping.erase(local_buffer);
}


VK_LAYER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkCreateBufferView(VkDevice device, const VkBufferViewCreateInfo* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkBufferView* pView)
{
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1490
    layer_data *my_map_data = get_my_data_ptr(get_dispatch_key(device), layer_data_map);
// STRUCT USES:['pCreateInfo']
//LOCAL DECLS:['pCreateInfo']
    safe_VkBufferViewCreateInfo* local_pCreateInfo = NULL;
    {
    std::lock_guard<std::mutex> lock(global_lock);
    if (pCreateInfo) {
        local_pCreateInfo = new safe_VkBufferViewCreateInfo(pCreateInfo);
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1450
        local_pCreateInfo->buffer = (VkBuffer)my_map_data->unique_id_mapping[reinterpret_cast<const uint64_t &>(pCreateInfo->buffer)];
    }
    }
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1596
    VkResult result = get_dispatch_table(unique_objects_device_table_map, device)->CreateBufferView(device, (const VkBufferViewCreateInfo*)local_pCreateInfo, pAllocator, pView);
    if (local_pCreateInfo)
        delete local_pCreateInfo;
    if (VK_SUCCESS == result) {
        std::lock_guard<std::mutex> lock(global_lock);
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1567
        uint64_t unique_id = global_unique_id++;
        my_map_data->unique_id_mapping[unique_id] = reinterpret_cast<uint64_t &>(*pView);
        *pView = reinterpret_cast<VkBufferView&>(unique_id);
    }
    return result;
}


VK_LAYER_EXPORT VKAPI_ATTR void VKAPI_CALL vkDestroyBufferView(VkDevice device, VkBufferView bufferView, const VkAllocationCallbacks* pAllocator)
{
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1490
    layer_data *my_map_data = get_my_data_ptr(get_dispatch_key(device), layer_data_map);
// STRUCT USES:['bufferView']
    std::unique_lock<std::mutex> lock(global_lock);
    uint64_t local_bufferView = reinterpret_cast<uint64_t &>(bufferView);
    bufferView = (VkBufferView)my_map_data->unique_id_mapping[local_bufferView];
    lock.unlock();
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1596
    get_dispatch_table(unique_objects_device_table_map, device)->DestroyBufferView(device, bufferView, pAllocator);
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1584
    lock.lock();
    my_map_data->unique_id_mapping.erase(local_bufferView);
}


VK_LAYER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkCreateImage(VkDevice device, const VkImageCreateInfo* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkImage* pImage)
{
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1490
    layer_data *my_map_data = get_my_data_ptr(get_dispatch_key(device), layer_data_map);
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1596
    VkResult result = get_dispatch_table(unique_objects_device_table_map, device)->CreateImage(device, pCreateInfo, pAllocator, pImage);
    if (VK_SUCCESS == result) {
        std::lock_guard<std::mutex> lock(global_lock);
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1567
        uint64_t unique_id = global_unique_id++;
        my_map_data->unique_id_mapping[unique_id] = reinterpret_cast<uint64_t &>(*pImage);
        *pImage = reinterpret_cast<VkImage&>(unique_id);
    }
    return result;
}


VK_LAYER_EXPORT VKAPI_ATTR void VKAPI_CALL vkDestroyImage(VkDevice device, VkImage image, const VkAllocationCallbacks* pAllocator)
{
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1490
    layer_data *my_map_data = get_my_data_ptr(get_dispatch_key(device), layer_data_map);
// STRUCT USES:['image']
    std::unique_lock<std::mutex> lock(global_lock);
    uint64_t local_image = reinterpret_cast<uint64_t &>(image);
    image = (VkImage)my_map_data->unique_id_mapping[local_image];
    lock.unlock();
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1596
    get_dispatch_table(unique_objects_device_table_map, device)->DestroyImage(device, image, pAllocator);
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1584
    lock.lock();
    my_map_data->unique_id_mapping.erase(local_image);
}


VK_LAYER_EXPORT VKAPI_ATTR void VKAPI_CALL vkGetImageSubresourceLayout(VkDevice device, VkImage image, const VkImageSubresource* pSubresource, VkSubresourceLayout* pLayout)
{
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1490
    layer_data *my_map_data = get_my_data_ptr(get_dispatch_key(device), layer_data_map);
// STRUCT USES:['image']
    {
    std::lock_guard<std::mutex> lock(global_lock);
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1450
    image = (VkImage)my_map_data->unique_id_mapping[reinterpret_cast<uint64_t &>(image)];
    }
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1596
    get_dispatch_table(unique_objects_device_table_map, device)->GetImageSubresourceLayout(device, image, pSubresource, pLayout);
}


VK_LAYER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkCreateImageView(VkDevice device, const VkImageViewCreateInfo* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkImageView* pView)
{
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1490
    layer_data *my_map_data = get_my_data_ptr(get_dispatch_key(device), layer_data_map);
// STRUCT USES:['pCreateInfo']
//LOCAL DECLS:['pCreateInfo']
    safe_VkImageViewCreateInfo* local_pCreateInfo = NULL;
    {
    std::lock_guard<std::mutex> lock(global_lock);
    if (pCreateInfo) {
        local_pCreateInfo = new safe_VkImageViewCreateInfo(pCreateInfo);
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1450
        local_pCreateInfo->image = (VkImage)my_map_data->unique_id_mapping[reinterpret_cast<const uint64_t &>(pCreateInfo->image)];
    }
    }
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1596
    VkResult result = get_dispatch_table(unique_objects_device_table_map, device)->CreateImageView(device, (const VkImageViewCreateInfo*)local_pCreateInfo, pAllocator, pView);
    if (local_pCreateInfo)
        delete local_pCreateInfo;
    if (VK_SUCCESS == result) {
        std::lock_guard<std::mutex> lock(global_lock);
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1567
        uint64_t unique_id = global_unique_id++;
        my_map_data->unique_id_mapping[unique_id] = reinterpret_cast<uint64_t &>(*pView);
        *pView = reinterpret_cast<VkImageView&>(unique_id);
    }
    return result;
}


VK_LAYER_EXPORT VKAPI_ATTR void VKAPI_CALL vkDestroyImageView(VkDevice device, VkImageView imageView, const VkAllocationCallbacks* pAllocator)
{
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1490
    layer_data *my_map_data = get_my_data_ptr(get_dispatch_key(device), layer_data_map);
// STRUCT USES:['imageView']
    std::unique_lock<std::mutex> lock(global_lock);
    uint64_t local_imageView = reinterpret_cast<uint64_t &>(imageView);
    imageView = (VkImageView)my_map_data->unique_id_mapping[local_imageView];
    lock.unlock();
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1596
    get_dispatch_table(unique_objects_device_table_map, device)->DestroyImageView(device, imageView, pAllocator);
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1584
    lock.lock();
    my_map_data->unique_id_mapping.erase(local_imageView);
}


VK_LAYER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkCreateShaderModule(VkDevice device, const VkShaderModuleCreateInfo* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkShaderModule* pShaderModule)
{
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1490
    layer_data *my_map_data = get_my_data_ptr(get_dispatch_key(device), layer_data_map);
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1596
    VkResult result = get_dispatch_table(unique_objects_device_table_map, device)->CreateShaderModule(device, pCreateInfo, pAllocator, pShaderModule);
    if (VK_SUCCESS == result) {
        std::lock_guard<std::mutex> lock(global_lock);
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1567
        uint64_t unique_id = global_unique_id++;
        my_map_data->unique_id_mapping[unique_id] = reinterpret_cast<uint64_t &>(*pShaderModule);
        *pShaderModule = reinterpret_cast<VkShaderModule&>(unique_id);
    }
    return result;
}


VK_LAYER_EXPORT VKAPI_ATTR void VKAPI_CALL vkDestroyShaderModule(VkDevice device, VkShaderModule shaderModule, const VkAllocationCallbacks* pAllocator)
{
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1490
    layer_data *my_map_data = get_my_data_ptr(get_dispatch_key(device), layer_data_map);
// STRUCT USES:['shaderModule']
    std::unique_lock<std::mutex> lock(global_lock);
    uint64_t local_shaderModule = reinterpret_cast<uint64_t &>(shaderModule);
    shaderModule = (VkShaderModule)my_map_data->unique_id_mapping[local_shaderModule];
    lock.unlock();
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1596
    get_dispatch_table(unique_objects_device_table_map, device)->DestroyShaderModule(device, shaderModule, pAllocator);
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1584
    lock.lock();
    my_map_data->unique_id_mapping.erase(local_shaderModule);
}


VK_LAYER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkCreatePipelineCache(VkDevice device, const VkPipelineCacheCreateInfo* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkPipelineCache* pPipelineCache)
{
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1490
    layer_data *my_map_data = get_my_data_ptr(get_dispatch_key(device), layer_data_map);
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1596
    VkResult result = get_dispatch_table(unique_objects_device_table_map, device)->CreatePipelineCache(device, pCreateInfo, pAllocator, pPipelineCache);
    if (VK_SUCCESS == result) {
        std::lock_guard<std::mutex> lock(global_lock);
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1567
        uint64_t unique_id = global_unique_id++;
        my_map_data->unique_id_mapping[unique_id] = reinterpret_cast<uint64_t &>(*pPipelineCache);
        *pPipelineCache = reinterpret_cast<VkPipelineCache&>(unique_id);
    }
    return result;
}


VK_LAYER_EXPORT VKAPI_ATTR void VKAPI_CALL vkDestroyPipelineCache(VkDevice device, VkPipelineCache pipelineCache, const VkAllocationCallbacks* pAllocator)
{
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1490
    layer_data *my_map_data = get_my_data_ptr(get_dispatch_key(device), layer_data_map);
// STRUCT USES:['pipelineCache']
    std::unique_lock<std::mutex> lock(global_lock);
    uint64_t local_pipelineCache = reinterpret_cast<uint64_t &>(pipelineCache);
    pipelineCache = (VkPipelineCache)my_map_data->unique_id_mapping[local_pipelineCache];
    lock.unlock();
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1596
    get_dispatch_table(unique_objects_device_table_map, device)->DestroyPipelineCache(device, pipelineCache, pAllocator);
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1584
    lock.lock();
    my_map_data->unique_id_mapping.erase(local_pipelineCache);
}


VK_LAYER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkGetPipelineCacheData(VkDevice device, VkPipelineCache pipelineCache, size_t* pDataSize, void* pData)
{
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1490
    layer_data *my_map_data = get_my_data_ptr(get_dispatch_key(device), layer_data_map);
// STRUCT USES:['pipelineCache']
    {
    std::lock_guard<std::mutex> lock(global_lock);
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1450
    pipelineCache = (VkPipelineCache)my_map_data->unique_id_mapping[reinterpret_cast<uint64_t &>(pipelineCache)];
    }
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1596
    VkResult result = get_dispatch_table(unique_objects_device_table_map, device)->GetPipelineCacheData(device, pipelineCache, pDataSize, pData);
    return result;
}


VK_LAYER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkMergePipelineCaches(VkDevice device, VkPipelineCache dstCache, uint32_t srcCacheCount, const VkPipelineCache* pSrcCaches)
{
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1490
    layer_data *my_map_data = get_my_data_ptr(get_dispatch_key(device), layer_data_map);
// STRUCT USES:['dstCache', 'pSrcCaches[srcCacheCount]']
//LOCAL DECLS:['pSrcCaches']
    VkPipelineCache* local_pSrcCaches = NULL;
    {
    std::lock_guard<std::mutex> lock(global_lock);
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1450
    dstCache = (VkPipelineCache)my_map_data->unique_id_mapping[reinterpret_cast<uint64_t &>(dstCache)];
    if (pSrcCaches) {
        local_pSrcCaches = new VkPipelineCache[srcCacheCount];
        for (uint32_t idx0=0; idx0<srcCacheCount; ++idx0) {
            local_pSrcCaches[idx0] = (VkPipelineCache)my_map_data->unique_id_mapping[reinterpret_cast<const uint64_t &>(pSrcCaches[idx0])];
        }
    }
    }
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1596
    VkResult result = get_dispatch_table(unique_objects_device_table_map, device)->MergePipelineCaches(device, dstCache, srcCacheCount, (const VkPipelineCache*)local_pSrcCaches);
    if (local_pSrcCaches)
        delete[] local_pSrcCaches;
    return result;
}


VK_LAYER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkCreateGraphicsPipelines(VkDevice device, VkPipelineCache pipelineCache, uint32_t createInfoCount, const VkGraphicsPipelineCreateInfo* pCreateInfos, const VkAllocationCallbacks* pAllocator, VkPipeline* pPipelines)
{
    return explicit_CreateGraphicsPipelines(device, pipelineCache, createInfoCount, pCreateInfos, pAllocator, pPipelines);
}

VK_LAYER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkCreateComputePipelines(VkDevice device, VkPipelineCache pipelineCache, uint32_t createInfoCount, const VkComputePipelineCreateInfo* pCreateInfos, const VkAllocationCallbacks* pAllocator, VkPipeline* pPipelines)
{
    return explicit_CreateComputePipelines(device, pipelineCache, createInfoCount, pCreateInfos, pAllocator, pPipelines);
}

VK_LAYER_EXPORT VKAPI_ATTR void VKAPI_CALL vkDestroyPipeline(VkDevice device, VkPipeline pipeline, const VkAllocationCallbacks* pAllocator)
{
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1490
    layer_data *my_map_data = get_my_data_ptr(get_dispatch_key(device), layer_data_map);
// STRUCT USES:['pipeline']
    std::unique_lock<std::mutex> lock(global_lock);
    uint64_t local_pipeline = reinterpret_cast<uint64_t &>(pipeline);
    pipeline = (VkPipeline)my_map_data->unique_id_mapping[local_pipeline];
    lock.unlock();
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1596
    get_dispatch_table(unique_objects_device_table_map, device)->DestroyPipeline(device, pipeline, pAllocator);
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1584
    lock.lock();
    my_map_data->unique_id_mapping.erase(local_pipeline);
}


VK_LAYER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkCreatePipelineLayout(VkDevice device, const VkPipelineLayoutCreateInfo* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkPipelineLayout* pPipelineLayout)
{
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1490
    layer_data *my_map_data = get_my_data_ptr(get_dispatch_key(device), layer_data_map);
// STRUCT USES:['pCreateInfo']
//LOCAL DECLS:['pCreateInfo']
    safe_VkPipelineLayoutCreateInfo* local_pCreateInfo = NULL;
    {
    std::lock_guard<std::mutex> lock(global_lock);
    if (pCreateInfo) {
        local_pCreateInfo = new safe_VkPipelineLayoutCreateInfo(pCreateInfo);
        if (local_pCreateInfo->pSetLayouts) {
            for (uint32_t idx0=0; idx0<pCreateInfo->setLayoutCount; ++idx0) {
                local_pCreateInfo->pSetLayouts[idx0] = (VkDescriptorSetLayout)my_map_data->unique_id_mapping[reinterpret_cast<const uint64_t &>(pCreateInfo->pSetLayouts[idx0])];
            }
        }
    }
    }
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1596
    VkResult result = get_dispatch_table(unique_objects_device_table_map, device)->CreatePipelineLayout(device, (const VkPipelineLayoutCreateInfo*)local_pCreateInfo, pAllocator, pPipelineLayout);
    if (local_pCreateInfo)
        delete local_pCreateInfo;
    if (VK_SUCCESS == result) {
        std::lock_guard<std::mutex> lock(global_lock);
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1567
        uint64_t unique_id = global_unique_id++;
        my_map_data->unique_id_mapping[unique_id] = reinterpret_cast<uint64_t &>(*pPipelineLayout);
        *pPipelineLayout = reinterpret_cast<VkPipelineLayout&>(unique_id);
    }
    return result;
}


VK_LAYER_EXPORT VKAPI_ATTR void VKAPI_CALL vkDestroyPipelineLayout(VkDevice device, VkPipelineLayout pipelineLayout, const VkAllocationCallbacks* pAllocator)
{
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1490
    layer_data *my_map_data = get_my_data_ptr(get_dispatch_key(device), layer_data_map);
// STRUCT USES:['pipelineLayout']
    std::unique_lock<std::mutex> lock(global_lock);
    uint64_t local_pipelineLayout = reinterpret_cast<uint64_t &>(pipelineLayout);
    pipelineLayout = (VkPipelineLayout)my_map_data->unique_id_mapping[local_pipelineLayout];
    lock.unlock();
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1596
    get_dispatch_table(unique_objects_device_table_map, device)->DestroyPipelineLayout(device, pipelineLayout, pAllocator);
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1584
    lock.lock();
    my_map_data->unique_id_mapping.erase(local_pipelineLayout);
}


VK_LAYER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkCreateSampler(VkDevice device, const VkSamplerCreateInfo* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkSampler* pSampler)
{
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1490
    layer_data *my_map_data = get_my_data_ptr(get_dispatch_key(device), layer_data_map);
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1596
    VkResult result = get_dispatch_table(unique_objects_device_table_map, device)->CreateSampler(device, pCreateInfo, pAllocator, pSampler);
    if (VK_SUCCESS == result) {
        std::lock_guard<std::mutex> lock(global_lock);
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1567
        uint64_t unique_id = global_unique_id++;
        my_map_data->unique_id_mapping[unique_id] = reinterpret_cast<uint64_t &>(*pSampler);
        *pSampler = reinterpret_cast<VkSampler&>(unique_id);
    }
    return result;
}


VK_LAYER_EXPORT VKAPI_ATTR void VKAPI_CALL vkDestroySampler(VkDevice device, VkSampler sampler, const VkAllocationCallbacks* pAllocator)
{
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1490
    layer_data *my_map_data = get_my_data_ptr(get_dispatch_key(device), layer_data_map);
// STRUCT USES:['sampler']
    std::unique_lock<std::mutex> lock(global_lock);
    uint64_t local_sampler = reinterpret_cast<uint64_t &>(sampler);
    sampler = (VkSampler)my_map_data->unique_id_mapping[local_sampler];
    lock.unlock();
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1596
    get_dispatch_table(unique_objects_device_table_map, device)->DestroySampler(device, sampler, pAllocator);
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1584
    lock.lock();
    my_map_data->unique_id_mapping.erase(local_sampler);
}


VK_LAYER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkCreateDescriptorSetLayout(VkDevice device, const VkDescriptorSetLayoutCreateInfo* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkDescriptorSetLayout* pSetLayout)
{
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1490
    layer_data *my_map_data = get_my_data_ptr(get_dispatch_key(device), layer_data_map);
// STRUCT USES:['pCreateInfo']
//LOCAL DECLS:['pCreateInfo']
    safe_VkDescriptorSetLayoutCreateInfo* local_pCreateInfo = NULL;
    {
    std::lock_guard<std::mutex> lock(global_lock);
    if (pCreateInfo) {
        local_pCreateInfo = new safe_VkDescriptorSetLayoutCreateInfo(pCreateInfo);
        if (local_pCreateInfo->pBindings) {
            for (uint32_t idx0=0; idx0<pCreateInfo->bindingCount; ++idx0) {
                if (local_pCreateInfo->pBindings[idx0].pImmutableSamplers) {
                    for (uint32_t idx1=0; idx1<pCreateInfo->pBindings[idx0].descriptorCount; ++idx1) {
                        local_pCreateInfo->pBindings[idx0].pImmutableSamplers[idx1] = (VkSampler)my_map_data->unique_id_mapping[reinterpret_cast<const uint64_t &>(pCreateInfo->pBindings[idx0].pImmutableSamplers[idx1])];
                    }
                }
            }
        }
    }
    }
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1596
    VkResult result = get_dispatch_table(unique_objects_device_table_map, device)->CreateDescriptorSetLayout(device, (const VkDescriptorSetLayoutCreateInfo*)local_pCreateInfo, pAllocator, pSetLayout);
    if (local_pCreateInfo)
        delete local_pCreateInfo;
    if (VK_SUCCESS == result) {
        std::lock_guard<std::mutex> lock(global_lock);
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1567
        uint64_t unique_id = global_unique_id++;
        my_map_data->unique_id_mapping[unique_id] = reinterpret_cast<uint64_t &>(*pSetLayout);
        *pSetLayout = reinterpret_cast<VkDescriptorSetLayout&>(unique_id);
    }
    return result;
}


VK_LAYER_EXPORT VKAPI_ATTR void VKAPI_CALL vkDestroyDescriptorSetLayout(VkDevice device, VkDescriptorSetLayout descriptorSetLayout, const VkAllocationCallbacks* pAllocator)
{
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1490
    layer_data *my_map_data = get_my_data_ptr(get_dispatch_key(device), layer_data_map);
// STRUCT USES:['descriptorSetLayout']
    std::unique_lock<std::mutex> lock(global_lock);
    uint64_t local_descriptorSetLayout = reinterpret_cast<uint64_t &>(descriptorSetLayout);
    descriptorSetLayout = (VkDescriptorSetLayout)my_map_data->unique_id_mapping[local_descriptorSetLayout];
    lock.unlock();
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1596
    get_dispatch_table(unique_objects_device_table_map, device)->DestroyDescriptorSetLayout(device, descriptorSetLayout, pAllocator);
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1584
    lock.lock();
    my_map_data->unique_id_mapping.erase(local_descriptorSetLayout);
}


VK_LAYER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkCreateDescriptorPool(VkDevice device, const VkDescriptorPoolCreateInfo* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkDescriptorPool* pDescriptorPool)
{
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1490
    layer_data *my_map_data = get_my_data_ptr(get_dispatch_key(device), layer_data_map);
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1596
    VkResult result = get_dispatch_table(unique_objects_device_table_map, device)->CreateDescriptorPool(device, pCreateInfo, pAllocator, pDescriptorPool);
    if (VK_SUCCESS == result) {
        std::lock_guard<std::mutex> lock(global_lock);
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1567
        uint64_t unique_id = global_unique_id++;
        my_map_data->unique_id_mapping[unique_id] = reinterpret_cast<uint64_t &>(*pDescriptorPool);
        *pDescriptorPool = reinterpret_cast<VkDescriptorPool&>(unique_id);
    }
    return result;
}


VK_LAYER_EXPORT VKAPI_ATTR void VKAPI_CALL vkDestroyDescriptorPool(VkDevice device, VkDescriptorPool descriptorPool, const VkAllocationCallbacks* pAllocator)
{
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1490
    layer_data *my_map_data = get_my_data_ptr(get_dispatch_key(device), layer_data_map);
// STRUCT USES:['descriptorPool']
    std::unique_lock<std::mutex> lock(global_lock);
    uint64_t local_descriptorPool = reinterpret_cast<uint64_t &>(descriptorPool);
    descriptorPool = (VkDescriptorPool)my_map_data->unique_id_mapping[local_descriptorPool];
    lock.unlock();
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1596
    get_dispatch_table(unique_objects_device_table_map, device)->DestroyDescriptorPool(device, descriptorPool, pAllocator);
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1584
    lock.lock();
    my_map_data->unique_id_mapping.erase(local_descriptorPool);
}


VK_LAYER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkResetDescriptorPool(VkDevice device, VkDescriptorPool descriptorPool, VkDescriptorPoolResetFlags flags)
{
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1490
    layer_data *my_map_data = get_my_data_ptr(get_dispatch_key(device), layer_data_map);
// STRUCT USES:['descriptorPool']
    {
    std::lock_guard<std::mutex> lock(global_lock);
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1450
    descriptorPool = (VkDescriptorPool)my_map_data->unique_id_mapping[reinterpret_cast<uint64_t &>(descriptorPool)];
    }
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1596
    VkResult result = get_dispatch_table(unique_objects_device_table_map, device)->ResetDescriptorPool(device, descriptorPool, flags);
    return result;
}


VK_LAYER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkAllocateDescriptorSets(VkDevice device, const VkDescriptorSetAllocateInfo* pAllocateInfo, VkDescriptorSet* pDescriptorSets)
{
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1490
    layer_data *my_map_data = get_my_data_ptr(get_dispatch_key(device), layer_data_map);
// STRUCT USES:['pAllocateInfo']
//LOCAL DECLS:['pAllocateInfo']
    safe_VkDescriptorSetAllocateInfo* local_pAllocateInfo = NULL;
    {
    std::lock_guard<std::mutex> lock(global_lock);
    if (pAllocateInfo) {
        local_pAllocateInfo = new safe_VkDescriptorSetAllocateInfo(pAllocateInfo);
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1450
        local_pAllocateInfo->descriptorPool = (VkDescriptorPool)my_map_data->unique_id_mapping[reinterpret_cast<const uint64_t &>(pAllocateInfo->descriptorPool)];
        if (local_pAllocateInfo->pSetLayouts) {
            for (uint32_t idx0=0; idx0<pAllocateInfo->descriptorSetCount; ++idx0) {
                local_pAllocateInfo->pSetLayouts[idx0] = (VkDescriptorSetLayout)my_map_data->unique_id_mapping[reinterpret_cast<const uint64_t &>(pAllocateInfo->pSetLayouts[idx0])];
            }
        }
    }
    }
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1596
    VkResult result = get_dispatch_table(unique_objects_device_table_map, device)->AllocateDescriptorSets(device, (const VkDescriptorSetAllocateInfo*)local_pAllocateInfo, pDescriptorSets);
    if (local_pAllocateInfo)
        delete local_pAllocateInfo;
    if (VK_SUCCESS == result) {
        std::lock_guard<std::mutex> lock(global_lock);
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1557
        for (uint32_t i=0; i<pAllocateInfo->descriptorSetCount; ++i) {
            uint64_t unique_id = global_unique_id++;
            my_map_data->unique_id_mapping[unique_id] = reinterpret_cast<uint64_t &>(pDescriptorSets[i]);
            pDescriptorSets[i] = reinterpret_cast<VkDescriptorSet&>(unique_id);
        }
    }
    return result;
}


VK_LAYER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkFreeDescriptorSets(VkDevice device, VkDescriptorPool descriptorPool, uint32_t descriptorSetCount, const VkDescriptorSet* pDescriptorSets)
{
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1490
    layer_data *my_map_data = get_my_data_ptr(get_dispatch_key(device), layer_data_map);
// STRUCT USES:['descriptorPool', 'pDescriptorSets[descriptorSetCount]']
//LOCAL DECLS:['pDescriptorSets']
    VkDescriptorSet* local_pDescriptorSets = NULL;
    {
    std::lock_guard<std::mutex> lock(global_lock);
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1450
    descriptorPool = (VkDescriptorPool)my_map_data->unique_id_mapping[reinterpret_cast<uint64_t &>(descriptorPool)];
    if (pDescriptorSets) {
        local_pDescriptorSets = new VkDescriptorSet[descriptorSetCount];
        for (uint32_t idx0=0; idx0<descriptorSetCount; ++idx0) {
            local_pDescriptorSets[idx0] = (VkDescriptorSet)my_map_data->unique_id_mapping[reinterpret_cast<const uint64_t &>(pDescriptorSets[idx0])];
        }
    }
    }
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1596
    VkResult result = get_dispatch_table(unique_objects_device_table_map, device)->FreeDescriptorSets(device, descriptorPool, descriptorSetCount, (const VkDescriptorSet*)local_pDescriptorSets);
    if (local_pDescriptorSets)
        delete[] local_pDescriptorSets;
    return result;
}


VK_LAYER_EXPORT VKAPI_ATTR void VKAPI_CALL vkUpdateDescriptorSets(VkDevice device, uint32_t descriptorWriteCount, const VkWriteDescriptorSet* pDescriptorWrites, uint32_t descriptorCopyCount, const VkCopyDescriptorSet* pDescriptorCopies)
{
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1490
    layer_data *my_map_data = get_my_data_ptr(get_dispatch_key(device), layer_data_map);
// STRUCT USES:['pDescriptorCopies[descriptorCopyCount]', 'pDescriptorWrites[descriptorWriteCount]']
//LOCAL DECLS:['pDescriptorCopies', 'pDescriptorWrites']
    safe_VkCopyDescriptorSet* local_pDescriptorCopies = NULL;
    safe_VkWriteDescriptorSet* local_pDescriptorWrites = NULL;
    {
    std::lock_guard<std::mutex> lock(global_lock);
    if (pDescriptorCopies) {
        local_pDescriptorCopies = new safe_VkCopyDescriptorSet[descriptorCopyCount];
        for (uint32_t idx0=0; idx0<descriptorCopyCount; ++idx0) {
            local_pDescriptorCopies[idx0].initialize(&pDescriptorCopies[idx0]);
            if (pDescriptorCopies[idx0].dstSet) {
                local_pDescriptorCopies[idx0].dstSet = (VkDescriptorSet)my_map_data->unique_id_mapping[reinterpret_cast<const uint64_t &>(pDescriptorCopies[idx0].dstSet)];
            }
            if (pDescriptorCopies[idx0].srcSet) {
                local_pDescriptorCopies[idx0].srcSet = (VkDescriptorSet)my_map_data->unique_id_mapping[reinterpret_cast<const uint64_t &>(pDescriptorCopies[idx0].srcSet)];
            }
        }
    }
    if (pDescriptorWrites) {
        local_pDescriptorWrites = new safe_VkWriteDescriptorSet[descriptorWriteCount];
        for (uint32_t idx1=0; idx1<descriptorWriteCount; ++idx1) {
            local_pDescriptorWrites[idx1].initialize(&pDescriptorWrites[idx1]);
            if (pDescriptorWrites[idx1].dstSet) {
                local_pDescriptorWrites[idx1].dstSet = (VkDescriptorSet)my_map_data->unique_id_mapping[reinterpret_cast<const uint64_t &>(pDescriptorWrites[idx1].dstSet)];
            }
            if (local_pDescriptorWrites[idx1].pBufferInfo) {
                for (uint32_t idx2=0; idx2<pDescriptorWrites[idx1].descriptorCount; ++idx2) {
                    if (pDescriptorWrites[idx1].pBufferInfo[idx2].buffer) {
                        local_pDescriptorWrites[idx1].pBufferInfo[idx2].buffer = (VkBuffer)my_map_data->unique_id_mapping[reinterpret_cast<const uint64_t &>(pDescriptorWrites[idx1].pBufferInfo[idx2].buffer)];
                    }
                }
            }
            if (local_pDescriptorWrites[idx1].pImageInfo) {
                for (uint32_t idx3=0; idx3<pDescriptorWrites[idx1].descriptorCount; ++idx3) {
                    if (pDescriptorWrites[idx1].pImageInfo[idx3].imageView) {
                        local_pDescriptorWrites[idx1].pImageInfo[idx3].imageView = (VkImageView)my_map_data->unique_id_mapping[reinterpret_cast<const uint64_t &>(pDescriptorWrites[idx1].pImageInfo[idx3].imageView)];
                    }
                    if (pDescriptorWrites[idx1].pImageInfo[idx3].sampler) {
                        local_pDescriptorWrites[idx1].pImageInfo[idx3].sampler = (VkSampler)my_map_data->unique_id_mapping[reinterpret_cast<const uint64_t &>(pDescriptorWrites[idx1].pImageInfo[idx3].sampler)];
                    }
                }
            }
            if (local_pDescriptorWrites[idx1].pTexelBufferView) {
                for (uint32_t idx4=0; idx4<pDescriptorWrites[idx1].descriptorCount; ++idx4) {
                    local_pDescriptorWrites[idx1].pTexelBufferView[idx4] = (VkBufferView)my_map_data->unique_id_mapping[reinterpret_cast<const uint64_t &>(pDescriptorWrites[idx1].pTexelBufferView[idx4])];
                }
            }
        }
    }
    }
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1596
    get_dispatch_table(unique_objects_device_table_map, device)->UpdateDescriptorSets(device, descriptorWriteCount, (const VkWriteDescriptorSet*)local_pDescriptorWrites, descriptorCopyCount, (const VkCopyDescriptorSet*)local_pDescriptorCopies);
    if (local_pDescriptorCopies)
        delete[] local_pDescriptorCopies;
    if (local_pDescriptorWrites)
        delete[] local_pDescriptorWrites;
}


VK_LAYER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkCreateFramebuffer(VkDevice device, const VkFramebufferCreateInfo* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkFramebuffer* pFramebuffer)
{
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1490
    layer_data *my_map_data = get_my_data_ptr(get_dispatch_key(device), layer_data_map);
// STRUCT USES:['pCreateInfo']
//LOCAL DECLS:['pCreateInfo']
    safe_VkFramebufferCreateInfo* local_pCreateInfo = NULL;
    {
    std::lock_guard<std::mutex> lock(global_lock);
    if (pCreateInfo) {
        local_pCreateInfo = new safe_VkFramebufferCreateInfo(pCreateInfo);
        if (local_pCreateInfo->pAttachments) {
            for (uint32_t idx0=0; idx0<pCreateInfo->attachmentCount; ++idx0) {
                local_pCreateInfo->pAttachments[idx0] = (VkImageView)my_map_data->unique_id_mapping[reinterpret_cast<const uint64_t &>(pCreateInfo->pAttachments[idx0])];
            }
        }
        if (pCreateInfo->renderPass) {
            local_pCreateInfo->renderPass = (VkRenderPass)my_map_data->unique_id_mapping[reinterpret_cast<const uint64_t &>(pCreateInfo->renderPass)];
        }
    }
    }
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1596
    VkResult result = get_dispatch_table(unique_objects_device_table_map, device)->CreateFramebuffer(device, (const VkFramebufferCreateInfo*)local_pCreateInfo, pAllocator, pFramebuffer);
    if (local_pCreateInfo)
        delete local_pCreateInfo;
    if (VK_SUCCESS == result) {
        std::lock_guard<std::mutex> lock(global_lock);
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1567
        uint64_t unique_id = global_unique_id++;
        my_map_data->unique_id_mapping[unique_id] = reinterpret_cast<uint64_t &>(*pFramebuffer);
        *pFramebuffer = reinterpret_cast<VkFramebuffer&>(unique_id);
    }
    return result;
}


VK_LAYER_EXPORT VKAPI_ATTR void VKAPI_CALL vkDestroyFramebuffer(VkDevice device, VkFramebuffer framebuffer, const VkAllocationCallbacks* pAllocator)
{
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1490
    layer_data *my_map_data = get_my_data_ptr(get_dispatch_key(device), layer_data_map);
// STRUCT USES:['framebuffer']
    std::unique_lock<std::mutex> lock(global_lock);
    uint64_t local_framebuffer = reinterpret_cast<uint64_t &>(framebuffer);
    framebuffer = (VkFramebuffer)my_map_data->unique_id_mapping[local_framebuffer];
    lock.unlock();
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1596
    get_dispatch_table(unique_objects_device_table_map, device)->DestroyFramebuffer(device, framebuffer, pAllocator);
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1584
    lock.lock();
    my_map_data->unique_id_mapping.erase(local_framebuffer);
}


VK_LAYER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkCreateRenderPass(VkDevice device, const VkRenderPassCreateInfo* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkRenderPass* pRenderPass)
{
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1490
    layer_data *my_map_data = get_my_data_ptr(get_dispatch_key(device), layer_data_map);
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1596
    VkResult result = get_dispatch_table(unique_objects_device_table_map, device)->CreateRenderPass(device, pCreateInfo, pAllocator, pRenderPass);
    if (VK_SUCCESS == result) {
        std::lock_guard<std::mutex> lock(global_lock);
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1567
        uint64_t unique_id = global_unique_id++;
        my_map_data->unique_id_mapping[unique_id] = reinterpret_cast<uint64_t &>(*pRenderPass);
        *pRenderPass = reinterpret_cast<VkRenderPass&>(unique_id);
    }
    return result;
}


VK_LAYER_EXPORT VKAPI_ATTR void VKAPI_CALL vkDestroyRenderPass(VkDevice device, VkRenderPass renderPass, const VkAllocationCallbacks* pAllocator)
{
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1490
    layer_data *my_map_data = get_my_data_ptr(get_dispatch_key(device), layer_data_map);
// STRUCT USES:['renderPass']
    std::unique_lock<std::mutex> lock(global_lock);
    uint64_t local_renderPass = reinterpret_cast<uint64_t &>(renderPass);
    renderPass = (VkRenderPass)my_map_data->unique_id_mapping[local_renderPass];
    lock.unlock();
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1596
    get_dispatch_table(unique_objects_device_table_map, device)->DestroyRenderPass(device, renderPass, pAllocator);
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1584
    lock.lock();
    my_map_data->unique_id_mapping.erase(local_renderPass);
}


VK_LAYER_EXPORT VKAPI_ATTR void VKAPI_CALL vkGetRenderAreaGranularity(VkDevice device, VkRenderPass renderPass, VkExtent2D* pGranularity)
{
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1490
    layer_data *my_map_data = get_my_data_ptr(get_dispatch_key(device), layer_data_map);
// STRUCT USES:['renderPass']
    {
    std::lock_guard<std::mutex> lock(global_lock);
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1450
    renderPass = (VkRenderPass)my_map_data->unique_id_mapping[reinterpret_cast<uint64_t &>(renderPass)];
    }
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1596
    get_dispatch_table(unique_objects_device_table_map, device)->GetRenderAreaGranularity(device, renderPass, pGranularity);
}


VK_LAYER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkCreateCommandPool(VkDevice device, const VkCommandPoolCreateInfo* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkCommandPool* pCommandPool)
{
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1490
    layer_data *my_map_data = get_my_data_ptr(get_dispatch_key(device), layer_data_map);
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1596
    VkResult result = get_dispatch_table(unique_objects_device_table_map, device)->CreateCommandPool(device, pCreateInfo, pAllocator, pCommandPool);
    if (VK_SUCCESS == result) {
        std::lock_guard<std::mutex> lock(global_lock);
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1567
        uint64_t unique_id = global_unique_id++;
        my_map_data->unique_id_mapping[unique_id] = reinterpret_cast<uint64_t &>(*pCommandPool);
        *pCommandPool = reinterpret_cast<VkCommandPool&>(unique_id);
    }
    return result;
}


VK_LAYER_EXPORT VKAPI_ATTR void VKAPI_CALL vkDestroyCommandPool(VkDevice device, VkCommandPool commandPool, const VkAllocationCallbacks* pAllocator)
{
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1490
    layer_data *my_map_data = get_my_data_ptr(get_dispatch_key(device), layer_data_map);
// STRUCT USES:['commandPool']
    std::unique_lock<std::mutex> lock(global_lock);
    uint64_t local_commandPool = reinterpret_cast<uint64_t &>(commandPool);
    commandPool = (VkCommandPool)my_map_data->unique_id_mapping[local_commandPool];
    lock.unlock();
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1596
    get_dispatch_table(unique_objects_device_table_map, device)->DestroyCommandPool(device, commandPool, pAllocator);
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1584
    lock.lock();
    my_map_data->unique_id_mapping.erase(local_commandPool);
}


VK_LAYER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkResetCommandPool(VkDevice device, VkCommandPool commandPool, VkCommandPoolResetFlags flags)
{
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1490
    layer_data *my_map_data = get_my_data_ptr(get_dispatch_key(device), layer_data_map);
// STRUCT USES:['commandPool']
    {
    std::lock_guard<std::mutex> lock(global_lock);
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1450
    commandPool = (VkCommandPool)my_map_data->unique_id_mapping[reinterpret_cast<uint64_t &>(commandPool)];
    }
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1596
    VkResult result = get_dispatch_table(unique_objects_device_table_map, device)->ResetCommandPool(device, commandPool, flags);
    return result;
}


VK_LAYER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkAllocateCommandBuffers(VkDevice device, const VkCommandBufferAllocateInfo* pAllocateInfo, VkCommandBuffer* pCommandBuffers)
{
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1490
    layer_data *my_map_data = get_my_data_ptr(get_dispatch_key(device), layer_data_map);
// STRUCT USES:['pAllocateInfo']
//LOCAL DECLS:['pAllocateInfo']
    safe_VkCommandBufferAllocateInfo* local_pAllocateInfo = NULL;
    {
    std::lock_guard<std::mutex> lock(global_lock);
    if (pAllocateInfo) {
        local_pAllocateInfo = new safe_VkCommandBufferAllocateInfo(pAllocateInfo);
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1450
        local_pAllocateInfo->commandPool = (VkCommandPool)my_map_data->unique_id_mapping[reinterpret_cast<const uint64_t &>(pAllocateInfo->commandPool)];
    }
    }
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1596
    VkResult result = get_dispatch_table(unique_objects_device_table_map, device)->AllocateCommandBuffers(device, (const VkCommandBufferAllocateInfo*)local_pAllocateInfo, pCommandBuffers);
    if (local_pAllocateInfo)
        delete local_pAllocateInfo;
    return result;
}


VK_LAYER_EXPORT VKAPI_ATTR void VKAPI_CALL vkFreeCommandBuffers(VkDevice device, VkCommandPool commandPool, uint32_t commandBufferCount, const VkCommandBuffer* pCommandBuffers)
{
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1490
    layer_data *my_map_data = get_my_data_ptr(get_dispatch_key(device), layer_data_map);
// STRUCT USES:['commandPool']
    {
    std::lock_guard<std::mutex> lock(global_lock);
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1450
    commandPool = (VkCommandPool)my_map_data->unique_id_mapping[reinterpret_cast<uint64_t &>(commandPool)];
    }
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1596
    get_dispatch_table(unique_objects_device_table_map, device)->FreeCommandBuffers(device, commandPool, commandBufferCount, pCommandBuffers);
}


VK_LAYER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkBeginCommandBuffer(VkCommandBuffer commandBuffer, const VkCommandBufferBeginInfo* pBeginInfo)
{
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1490
    layer_data *my_map_data = get_my_data_ptr(get_dispatch_key(commandBuffer), layer_data_map);
// STRUCT USES:['pBeginInfo']
//LOCAL DECLS:['pBeginInfo']
    safe_VkCommandBufferBeginInfo* local_pBeginInfo = NULL;
    {
    std::lock_guard<std::mutex> lock(global_lock);
    if (pBeginInfo) {
        local_pBeginInfo = new safe_VkCommandBufferBeginInfo(pBeginInfo);
        if (local_pBeginInfo->pInheritanceInfo) {
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1450
            local_pBeginInfo->pInheritanceInfo->framebuffer = (VkFramebuffer)my_map_data->unique_id_mapping[reinterpret_cast<const uint64_t &>(pBeginInfo->pInheritanceInfo->framebuffer)];
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1450
            local_pBeginInfo->pInheritanceInfo->renderPass = (VkRenderPass)my_map_data->unique_id_mapping[reinterpret_cast<const uint64_t &>(pBeginInfo->pInheritanceInfo->renderPass)];
        }
    }
    }
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1596
    VkResult result = get_dispatch_table(unique_objects_device_table_map, commandBuffer)->BeginCommandBuffer(commandBuffer, (const VkCommandBufferBeginInfo*)local_pBeginInfo);
    if (local_pBeginInfo)
        delete local_pBeginInfo;
    return result;
}


VK_LAYER_EXPORT VKAPI_ATTR void VKAPI_CALL vkCmdBindPipeline(VkCommandBuffer commandBuffer, VkPipelineBindPoint pipelineBindPoint, VkPipeline pipeline)
{
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1490
    layer_data *my_map_data = get_my_data_ptr(get_dispatch_key(commandBuffer), layer_data_map);
// STRUCT USES:['pipeline']
    {
    std::lock_guard<std::mutex> lock(global_lock);
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1450
    pipeline = (VkPipeline)my_map_data->unique_id_mapping[reinterpret_cast<uint64_t &>(pipeline)];
    }
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1596
    get_dispatch_table(unique_objects_device_table_map, commandBuffer)->CmdBindPipeline(commandBuffer, pipelineBindPoint, pipeline);
}


VK_LAYER_EXPORT VKAPI_ATTR void VKAPI_CALL vkCmdBindDescriptorSets(VkCommandBuffer commandBuffer, VkPipelineBindPoint pipelineBindPoint, VkPipelineLayout layout, uint32_t firstSet, uint32_t descriptorSetCount, const VkDescriptorSet* pDescriptorSets, uint32_t dynamicOffsetCount, const uint32_t* pDynamicOffsets)
{
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1490
    layer_data *my_map_data = get_my_data_ptr(get_dispatch_key(commandBuffer), layer_data_map);
// STRUCT USES:['layout', 'pDescriptorSets[descriptorSetCount]']
//LOCAL DECLS:['pDescriptorSets']
    VkDescriptorSet* local_pDescriptorSets = NULL;
    {
    std::lock_guard<std::mutex> lock(global_lock);
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1450
    layout = (VkPipelineLayout)my_map_data->unique_id_mapping[reinterpret_cast<uint64_t &>(layout)];
    if (pDescriptorSets) {
        local_pDescriptorSets = new VkDescriptorSet[descriptorSetCount];
        for (uint32_t idx0=0; idx0<descriptorSetCount; ++idx0) {
            local_pDescriptorSets[idx0] = (VkDescriptorSet)my_map_data->unique_id_mapping[reinterpret_cast<const uint64_t &>(pDescriptorSets[idx0])];
        }
    }
    }
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1596
    get_dispatch_table(unique_objects_device_table_map, commandBuffer)->CmdBindDescriptorSets(commandBuffer, pipelineBindPoint, layout, firstSet, descriptorSetCount, (const VkDescriptorSet*)local_pDescriptorSets, dynamicOffsetCount, pDynamicOffsets);
    if (local_pDescriptorSets)
        delete[] local_pDescriptorSets;
}


VK_LAYER_EXPORT VKAPI_ATTR void VKAPI_CALL vkCmdBindIndexBuffer(VkCommandBuffer commandBuffer, VkBuffer buffer, VkDeviceSize offset, VkIndexType indexType)
{
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1490
    layer_data *my_map_data = get_my_data_ptr(get_dispatch_key(commandBuffer), layer_data_map);
// STRUCT USES:['buffer']
    {
    std::lock_guard<std::mutex> lock(global_lock);
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1450
    buffer = (VkBuffer)my_map_data->unique_id_mapping[reinterpret_cast<uint64_t &>(buffer)];
    }
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1596
    get_dispatch_table(unique_objects_device_table_map, commandBuffer)->CmdBindIndexBuffer(commandBuffer, buffer, offset, indexType);
}


VK_LAYER_EXPORT VKAPI_ATTR void VKAPI_CALL vkCmdBindVertexBuffers(VkCommandBuffer commandBuffer, uint32_t firstBinding, uint32_t bindingCount, const VkBuffer* pBuffers, const VkDeviceSize* pOffsets)
{
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1490
    layer_data *my_map_data = get_my_data_ptr(get_dispatch_key(commandBuffer), layer_data_map);
// STRUCT USES:['pBuffers[bindingCount]']
//LOCAL DECLS:['pBuffers']
    VkBuffer* local_pBuffers = NULL;
    {
    std::lock_guard<std::mutex> lock(global_lock);
    if (pBuffers) {
        local_pBuffers = new VkBuffer[bindingCount];
        for (uint32_t idx0=0; idx0<bindingCount; ++idx0) {
            local_pBuffers[idx0] = (VkBuffer)my_map_data->unique_id_mapping[reinterpret_cast<const uint64_t &>(pBuffers[idx0])];
        }
    }
    }
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1596
    get_dispatch_table(unique_objects_device_table_map, commandBuffer)->CmdBindVertexBuffers(commandBuffer, firstBinding, bindingCount, (const VkBuffer*)local_pBuffers, pOffsets);
    if (local_pBuffers)
        delete[] local_pBuffers;
}


VK_LAYER_EXPORT VKAPI_ATTR void VKAPI_CALL vkCmdDrawIndirect(VkCommandBuffer commandBuffer, VkBuffer buffer, VkDeviceSize offset, uint32_t drawCount, uint32_t stride)
{
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1490
    layer_data *my_map_data = get_my_data_ptr(get_dispatch_key(commandBuffer), layer_data_map);
// STRUCT USES:['buffer']
    {
    std::lock_guard<std::mutex> lock(global_lock);
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1450
    buffer = (VkBuffer)my_map_data->unique_id_mapping[reinterpret_cast<uint64_t &>(buffer)];
    }
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1596
    get_dispatch_table(unique_objects_device_table_map, commandBuffer)->CmdDrawIndirect(commandBuffer, buffer, offset, drawCount, stride);
}


VK_LAYER_EXPORT VKAPI_ATTR void VKAPI_CALL vkCmdDrawIndexedIndirect(VkCommandBuffer commandBuffer, VkBuffer buffer, VkDeviceSize offset, uint32_t drawCount, uint32_t stride)
{
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1490
    layer_data *my_map_data = get_my_data_ptr(get_dispatch_key(commandBuffer), layer_data_map);
// STRUCT USES:['buffer']
    {
    std::lock_guard<std::mutex> lock(global_lock);
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1450
    buffer = (VkBuffer)my_map_data->unique_id_mapping[reinterpret_cast<uint64_t &>(buffer)];
    }
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1596
    get_dispatch_table(unique_objects_device_table_map, commandBuffer)->CmdDrawIndexedIndirect(commandBuffer, buffer, offset, drawCount, stride);
}


VK_LAYER_EXPORT VKAPI_ATTR void VKAPI_CALL vkCmdDispatchIndirect(VkCommandBuffer commandBuffer, VkBuffer buffer, VkDeviceSize offset)
{
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1490
    layer_data *my_map_data = get_my_data_ptr(get_dispatch_key(commandBuffer), layer_data_map);
// STRUCT USES:['buffer']
    {
    std::lock_guard<std::mutex> lock(global_lock);
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1450
    buffer = (VkBuffer)my_map_data->unique_id_mapping[reinterpret_cast<uint64_t &>(buffer)];
    }
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1596
    get_dispatch_table(unique_objects_device_table_map, commandBuffer)->CmdDispatchIndirect(commandBuffer, buffer, offset);
}


VK_LAYER_EXPORT VKAPI_ATTR void VKAPI_CALL vkCmdCopyBuffer(VkCommandBuffer commandBuffer, VkBuffer srcBuffer, VkBuffer dstBuffer, uint32_t regionCount, const VkBufferCopy* pRegions)
{
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1490
    layer_data *my_map_data = get_my_data_ptr(get_dispatch_key(commandBuffer), layer_data_map);
// STRUCT USES:['dstBuffer', 'srcBuffer']
    {
    std::lock_guard<std::mutex> lock(global_lock);
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1450
    dstBuffer = (VkBuffer)my_map_data->unique_id_mapping[reinterpret_cast<uint64_t &>(dstBuffer)];
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1450
    srcBuffer = (VkBuffer)my_map_data->unique_id_mapping[reinterpret_cast<uint64_t &>(srcBuffer)];
    }
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1596
    get_dispatch_table(unique_objects_device_table_map, commandBuffer)->CmdCopyBuffer(commandBuffer, srcBuffer, dstBuffer, regionCount, pRegions);
}


VK_LAYER_EXPORT VKAPI_ATTR void VKAPI_CALL vkCmdCopyImage(VkCommandBuffer commandBuffer, VkImage srcImage, VkImageLayout srcImageLayout, VkImage dstImage, VkImageLayout dstImageLayout, uint32_t regionCount, const VkImageCopy* pRegions)
{
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1490
    layer_data *my_map_data = get_my_data_ptr(get_dispatch_key(commandBuffer), layer_data_map);
// STRUCT USES:['dstImage', 'srcImage']
    {
    std::lock_guard<std::mutex> lock(global_lock);
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1450
    dstImage = (VkImage)my_map_data->unique_id_mapping[reinterpret_cast<uint64_t &>(dstImage)];
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1450
    srcImage = (VkImage)my_map_data->unique_id_mapping[reinterpret_cast<uint64_t &>(srcImage)];
    }
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1596
    get_dispatch_table(unique_objects_device_table_map, commandBuffer)->CmdCopyImage(commandBuffer, srcImage, srcImageLayout, dstImage, dstImageLayout, regionCount, pRegions);
}


VK_LAYER_EXPORT VKAPI_ATTR void VKAPI_CALL vkCmdBlitImage(VkCommandBuffer commandBuffer, VkImage srcImage, VkImageLayout srcImageLayout, VkImage dstImage, VkImageLayout dstImageLayout, uint32_t regionCount, const VkImageBlit* pRegions, VkFilter filter)
{
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1490
    layer_data *my_map_data = get_my_data_ptr(get_dispatch_key(commandBuffer), layer_data_map);
// STRUCT USES:['dstImage', 'srcImage']
    {
    std::lock_guard<std::mutex> lock(global_lock);
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1450
    dstImage = (VkImage)my_map_data->unique_id_mapping[reinterpret_cast<uint64_t &>(dstImage)];
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1450
    srcImage = (VkImage)my_map_data->unique_id_mapping[reinterpret_cast<uint64_t &>(srcImage)];
    }
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1596
    get_dispatch_table(unique_objects_device_table_map, commandBuffer)->CmdBlitImage(commandBuffer, srcImage, srcImageLayout, dstImage, dstImageLayout, regionCount, pRegions, filter);
}


VK_LAYER_EXPORT VKAPI_ATTR void VKAPI_CALL vkCmdCopyBufferToImage(VkCommandBuffer commandBuffer, VkBuffer srcBuffer, VkImage dstImage, VkImageLayout dstImageLayout, uint32_t regionCount, const VkBufferImageCopy* pRegions)
{
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1490
    layer_data *my_map_data = get_my_data_ptr(get_dispatch_key(commandBuffer), layer_data_map);
// STRUCT USES:['dstImage', 'srcBuffer']
    {
    std::lock_guard<std::mutex> lock(global_lock);
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1450
    dstImage = (VkImage)my_map_data->unique_id_mapping[reinterpret_cast<uint64_t &>(dstImage)];
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1450
    srcBuffer = (VkBuffer)my_map_data->unique_id_mapping[reinterpret_cast<uint64_t &>(srcBuffer)];
    }
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1596
    get_dispatch_table(unique_objects_device_table_map, commandBuffer)->CmdCopyBufferToImage(commandBuffer, srcBuffer, dstImage, dstImageLayout, regionCount, pRegions);
}


VK_LAYER_EXPORT VKAPI_ATTR void VKAPI_CALL vkCmdCopyImageToBuffer(VkCommandBuffer commandBuffer, VkImage srcImage, VkImageLayout srcImageLayout, VkBuffer dstBuffer, uint32_t regionCount, const VkBufferImageCopy* pRegions)
{
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1490
    layer_data *my_map_data = get_my_data_ptr(get_dispatch_key(commandBuffer), layer_data_map);
// STRUCT USES:['dstBuffer', 'srcImage']
    {
    std::lock_guard<std::mutex> lock(global_lock);
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1450
    dstBuffer = (VkBuffer)my_map_data->unique_id_mapping[reinterpret_cast<uint64_t &>(dstBuffer)];
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1450
    srcImage = (VkImage)my_map_data->unique_id_mapping[reinterpret_cast<uint64_t &>(srcImage)];
    }
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1596
    get_dispatch_table(unique_objects_device_table_map, commandBuffer)->CmdCopyImageToBuffer(commandBuffer, srcImage, srcImageLayout, dstBuffer, regionCount, pRegions);
}


VK_LAYER_EXPORT VKAPI_ATTR void VKAPI_CALL vkCmdUpdateBuffer(VkCommandBuffer commandBuffer, VkBuffer dstBuffer, VkDeviceSize dstOffset, VkDeviceSize dataSize, const uint32_t* pData)
{
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1490
    layer_data *my_map_data = get_my_data_ptr(get_dispatch_key(commandBuffer), layer_data_map);
// STRUCT USES:['dstBuffer']
    {
    std::lock_guard<std::mutex> lock(global_lock);
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1450
    dstBuffer = (VkBuffer)my_map_data->unique_id_mapping[reinterpret_cast<uint64_t &>(dstBuffer)];
    }
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1596
    get_dispatch_table(unique_objects_device_table_map, commandBuffer)->CmdUpdateBuffer(commandBuffer, dstBuffer, dstOffset, dataSize, pData);
}


VK_LAYER_EXPORT VKAPI_ATTR void VKAPI_CALL vkCmdFillBuffer(VkCommandBuffer commandBuffer, VkBuffer dstBuffer, VkDeviceSize dstOffset, VkDeviceSize size, uint32_t data)
{
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1490
    layer_data *my_map_data = get_my_data_ptr(get_dispatch_key(commandBuffer), layer_data_map);
// STRUCT USES:['dstBuffer']
    {
    std::lock_guard<std::mutex> lock(global_lock);
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1450
    dstBuffer = (VkBuffer)my_map_data->unique_id_mapping[reinterpret_cast<uint64_t &>(dstBuffer)];
    }
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1596
    get_dispatch_table(unique_objects_device_table_map, commandBuffer)->CmdFillBuffer(commandBuffer, dstBuffer, dstOffset, size, data);
}


VK_LAYER_EXPORT VKAPI_ATTR void VKAPI_CALL vkCmdClearColorImage(VkCommandBuffer commandBuffer, VkImage image, VkImageLayout imageLayout, const VkClearColorValue* pColor, uint32_t rangeCount, const VkImageSubresourceRange* pRanges)
{
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1490
    layer_data *my_map_data = get_my_data_ptr(get_dispatch_key(commandBuffer), layer_data_map);
// STRUCT USES:['image']
    {
    std::lock_guard<std::mutex> lock(global_lock);
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1450
    image = (VkImage)my_map_data->unique_id_mapping[reinterpret_cast<uint64_t &>(image)];
    }
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1596
    get_dispatch_table(unique_objects_device_table_map, commandBuffer)->CmdClearColorImage(commandBuffer, image, imageLayout, pColor, rangeCount, pRanges);
}


VK_LAYER_EXPORT VKAPI_ATTR void VKAPI_CALL vkCmdClearDepthStencilImage(VkCommandBuffer commandBuffer, VkImage image, VkImageLayout imageLayout, const VkClearDepthStencilValue* pDepthStencil, uint32_t rangeCount, const VkImageSubresourceRange* pRanges)
{
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1490
    layer_data *my_map_data = get_my_data_ptr(get_dispatch_key(commandBuffer), layer_data_map);
// STRUCT USES:['image']
    {
    std::lock_guard<std::mutex> lock(global_lock);
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1450
    image = (VkImage)my_map_data->unique_id_mapping[reinterpret_cast<uint64_t &>(image)];
    }
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1596
    get_dispatch_table(unique_objects_device_table_map, commandBuffer)->CmdClearDepthStencilImage(commandBuffer, image, imageLayout, pDepthStencil, rangeCount, pRanges);
}


VK_LAYER_EXPORT VKAPI_ATTR void VKAPI_CALL vkCmdResolveImage(VkCommandBuffer commandBuffer, VkImage srcImage, VkImageLayout srcImageLayout, VkImage dstImage, VkImageLayout dstImageLayout, uint32_t regionCount, const VkImageResolve* pRegions)
{
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1490
    layer_data *my_map_data = get_my_data_ptr(get_dispatch_key(commandBuffer), layer_data_map);
// STRUCT USES:['dstImage', 'srcImage']
    {
    std::lock_guard<std::mutex> lock(global_lock);
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1450
    dstImage = (VkImage)my_map_data->unique_id_mapping[reinterpret_cast<uint64_t &>(dstImage)];
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1450
    srcImage = (VkImage)my_map_data->unique_id_mapping[reinterpret_cast<uint64_t &>(srcImage)];
    }
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1596
    get_dispatch_table(unique_objects_device_table_map, commandBuffer)->CmdResolveImage(commandBuffer, srcImage, srcImageLayout, dstImage, dstImageLayout, regionCount, pRegions);
}


VK_LAYER_EXPORT VKAPI_ATTR void VKAPI_CALL vkCmdSetEvent(VkCommandBuffer commandBuffer, VkEvent event, VkPipelineStageFlags stageMask)
{
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1490
    layer_data *my_map_data = get_my_data_ptr(get_dispatch_key(commandBuffer), layer_data_map);
// STRUCT USES:['event']
    {
    std::lock_guard<std::mutex> lock(global_lock);
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1450
    event = (VkEvent)my_map_data->unique_id_mapping[reinterpret_cast<uint64_t &>(event)];
    }
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1596
    get_dispatch_table(unique_objects_device_table_map, commandBuffer)->CmdSetEvent(commandBuffer, event, stageMask);
}


VK_LAYER_EXPORT VKAPI_ATTR void VKAPI_CALL vkCmdResetEvent(VkCommandBuffer commandBuffer, VkEvent event, VkPipelineStageFlags stageMask)
{
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1490
    layer_data *my_map_data = get_my_data_ptr(get_dispatch_key(commandBuffer), layer_data_map);
// STRUCT USES:['event']
    {
    std::lock_guard<std::mutex> lock(global_lock);
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1450
    event = (VkEvent)my_map_data->unique_id_mapping[reinterpret_cast<uint64_t &>(event)];
    }
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1596
    get_dispatch_table(unique_objects_device_table_map, commandBuffer)->CmdResetEvent(commandBuffer, event, stageMask);
}


VK_LAYER_EXPORT VKAPI_ATTR void VKAPI_CALL vkCmdWaitEvents(VkCommandBuffer commandBuffer, uint32_t eventCount, const VkEvent* pEvents, VkPipelineStageFlags srcStageMask, VkPipelineStageFlags dstStageMask, uint32_t memoryBarrierCount, const VkMemoryBarrier* pMemoryBarriers, uint32_t bufferMemoryBarrierCount, const VkBufferMemoryBarrier* pBufferMemoryBarriers, uint32_t imageMemoryBarrierCount, const VkImageMemoryBarrier* pImageMemoryBarriers)
{
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1490
    layer_data *my_map_data = get_my_data_ptr(get_dispatch_key(commandBuffer), layer_data_map);
// STRUCT USES:['pBufferMemoryBarriers[bufferMemoryBarrierCount]', 'pEvents[eventCount]', 'pImageMemoryBarriers[imageMemoryBarrierCount]']
//LOCAL DECLS:['pBufferMemoryBarriers', 'pEvents', 'pImageMemoryBarriers']
    VkEvent* local_pEvents = NULL;
    safe_VkBufferMemoryBarrier* local_pBufferMemoryBarriers = NULL;
    safe_VkImageMemoryBarrier* local_pImageMemoryBarriers = NULL;
    {
    std::lock_guard<std::mutex> lock(global_lock);
    if (pBufferMemoryBarriers) {
        local_pBufferMemoryBarriers = new safe_VkBufferMemoryBarrier[bufferMemoryBarrierCount];
        for (uint32_t idx0=0; idx0<bufferMemoryBarrierCount; ++idx0) {
            local_pBufferMemoryBarriers[idx0].initialize(&pBufferMemoryBarriers[idx0]);
            if (pBufferMemoryBarriers[idx0].buffer) {
                local_pBufferMemoryBarriers[idx0].buffer = (VkBuffer)my_map_data->unique_id_mapping[reinterpret_cast<const uint64_t &>(pBufferMemoryBarriers[idx0].buffer)];
            }
        }
    }
    if (pEvents) {
        local_pEvents = new VkEvent[eventCount];
        for (uint32_t idx1=0; idx1<eventCount; ++idx1) {
            local_pEvents[idx1] = (VkEvent)my_map_data->unique_id_mapping[reinterpret_cast<const uint64_t &>(pEvents[idx1])];
        }
    }
    if (pImageMemoryBarriers) {
        local_pImageMemoryBarriers = new safe_VkImageMemoryBarrier[imageMemoryBarrierCount];
        for (uint32_t idx2=0; idx2<imageMemoryBarrierCount; ++idx2) {
            local_pImageMemoryBarriers[idx2].initialize(&pImageMemoryBarriers[idx2]);
            if (pImageMemoryBarriers[idx2].image) {
                local_pImageMemoryBarriers[idx2].image = (VkImage)my_map_data->unique_id_mapping[reinterpret_cast<const uint64_t &>(pImageMemoryBarriers[idx2].image)];
            }
        }
    }
    }
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1596
    get_dispatch_table(unique_objects_device_table_map, commandBuffer)->CmdWaitEvents(commandBuffer, eventCount, (const VkEvent*)local_pEvents, srcStageMask, dstStageMask, memoryBarrierCount, pMemoryBarriers, bufferMemoryBarrierCount, (const VkBufferMemoryBarrier*)local_pBufferMemoryBarriers, imageMemoryBarrierCount, (const VkImageMemoryBarrier*)local_pImageMemoryBarriers);
    if (local_pBufferMemoryBarriers)
        delete[] local_pBufferMemoryBarriers;
    if (local_pEvents)
        delete[] local_pEvents;
    if (local_pImageMemoryBarriers)
        delete[] local_pImageMemoryBarriers;
}


VK_LAYER_EXPORT VKAPI_ATTR void VKAPI_CALL vkCmdPipelineBarrier(VkCommandBuffer commandBuffer, VkPipelineStageFlags srcStageMask, VkPipelineStageFlags dstStageMask, VkDependencyFlags dependencyFlags, uint32_t memoryBarrierCount, const VkMemoryBarrier* pMemoryBarriers, uint32_t bufferMemoryBarrierCount, const VkBufferMemoryBarrier* pBufferMemoryBarriers, uint32_t imageMemoryBarrierCount, const VkImageMemoryBarrier* pImageMemoryBarriers)
{
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1490
    layer_data *my_map_data = get_my_data_ptr(get_dispatch_key(commandBuffer), layer_data_map);
// STRUCT USES:['pBufferMemoryBarriers[bufferMemoryBarrierCount]', 'pImageMemoryBarriers[imageMemoryBarrierCount]']
//LOCAL DECLS:['pBufferMemoryBarriers', 'pImageMemoryBarriers']
    safe_VkBufferMemoryBarrier* local_pBufferMemoryBarriers = NULL;
    safe_VkImageMemoryBarrier* local_pImageMemoryBarriers = NULL;
    {
    std::lock_guard<std::mutex> lock(global_lock);
    if (pBufferMemoryBarriers) {
        local_pBufferMemoryBarriers = new safe_VkBufferMemoryBarrier[bufferMemoryBarrierCount];
        for (uint32_t idx0=0; idx0<bufferMemoryBarrierCount; ++idx0) {
            local_pBufferMemoryBarriers[idx0].initialize(&pBufferMemoryBarriers[idx0]);
            if (pBufferMemoryBarriers[idx0].buffer) {
                local_pBufferMemoryBarriers[idx0].buffer = (VkBuffer)my_map_data->unique_id_mapping[reinterpret_cast<const uint64_t &>(pBufferMemoryBarriers[idx0].buffer)];
            }
        }
    }
    if (pImageMemoryBarriers) {
        local_pImageMemoryBarriers = new safe_VkImageMemoryBarrier[imageMemoryBarrierCount];
        for (uint32_t idx1=0; idx1<imageMemoryBarrierCount; ++idx1) {
            local_pImageMemoryBarriers[idx1].initialize(&pImageMemoryBarriers[idx1]);
            if (pImageMemoryBarriers[idx1].image) {
                local_pImageMemoryBarriers[idx1].image = (VkImage)my_map_data->unique_id_mapping[reinterpret_cast<const uint64_t &>(pImageMemoryBarriers[idx1].image)];
            }
        }
    }
    }
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1596
    get_dispatch_table(unique_objects_device_table_map, commandBuffer)->CmdPipelineBarrier(commandBuffer, srcStageMask, dstStageMask, dependencyFlags, memoryBarrierCount, pMemoryBarriers, bufferMemoryBarrierCount, (const VkBufferMemoryBarrier*)local_pBufferMemoryBarriers, imageMemoryBarrierCount, (const VkImageMemoryBarrier*)local_pImageMemoryBarriers);
    if (local_pBufferMemoryBarriers)
        delete[] local_pBufferMemoryBarriers;
    if (local_pImageMemoryBarriers)
        delete[] local_pImageMemoryBarriers;
}


VK_LAYER_EXPORT VKAPI_ATTR void VKAPI_CALL vkCmdBeginQuery(VkCommandBuffer commandBuffer, VkQueryPool queryPool, uint32_t query, VkQueryControlFlags flags)
{
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1490
    layer_data *my_map_data = get_my_data_ptr(get_dispatch_key(commandBuffer), layer_data_map);
// STRUCT USES:['queryPool']
    {
    std::lock_guard<std::mutex> lock(global_lock);
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1450
    queryPool = (VkQueryPool)my_map_data->unique_id_mapping[reinterpret_cast<uint64_t &>(queryPool)];
    }
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1596
    get_dispatch_table(unique_objects_device_table_map, commandBuffer)->CmdBeginQuery(commandBuffer, queryPool, query, flags);
}


VK_LAYER_EXPORT VKAPI_ATTR void VKAPI_CALL vkCmdEndQuery(VkCommandBuffer commandBuffer, VkQueryPool queryPool, uint32_t query)
{
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1490
    layer_data *my_map_data = get_my_data_ptr(get_dispatch_key(commandBuffer), layer_data_map);
// STRUCT USES:['queryPool']
    {
    std::lock_guard<std::mutex> lock(global_lock);
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1450
    queryPool = (VkQueryPool)my_map_data->unique_id_mapping[reinterpret_cast<uint64_t &>(queryPool)];
    }
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1596
    get_dispatch_table(unique_objects_device_table_map, commandBuffer)->CmdEndQuery(commandBuffer, queryPool, query);
}


VK_LAYER_EXPORT VKAPI_ATTR void VKAPI_CALL vkCmdResetQueryPool(VkCommandBuffer commandBuffer, VkQueryPool queryPool, uint32_t firstQuery, uint32_t queryCount)
{
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1490
    layer_data *my_map_data = get_my_data_ptr(get_dispatch_key(commandBuffer), layer_data_map);
// STRUCT USES:['queryPool']
    {
    std::lock_guard<std::mutex> lock(global_lock);
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1450
    queryPool = (VkQueryPool)my_map_data->unique_id_mapping[reinterpret_cast<uint64_t &>(queryPool)];
    }
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1596
    get_dispatch_table(unique_objects_device_table_map, commandBuffer)->CmdResetQueryPool(commandBuffer, queryPool, firstQuery, queryCount);
}


VK_LAYER_EXPORT VKAPI_ATTR void VKAPI_CALL vkCmdWriteTimestamp(VkCommandBuffer commandBuffer, VkPipelineStageFlagBits pipelineStage, VkQueryPool queryPool, uint32_t query)
{
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1490
    layer_data *my_map_data = get_my_data_ptr(get_dispatch_key(commandBuffer), layer_data_map);
// STRUCT USES:['queryPool']
    {
    std::lock_guard<std::mutex> lock(global_lock);
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1450
    queryPool = (VkQueryPool)my_map_data->unique_id_mapping[reinterpret_cast<uint64_t &>(queryPool)];
    }
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1596
    get_dispatch_table(unique_objects_device_table_map, commandBuffer)->CmdWriteTimestamp(commandBuffer, pipelineStage, queryPool, query);
}


VK_LAYER_EXPORT VKAPI_ATTR void VKAPI_CALL vkCmdCopyQueryPoolResults(VkCommandBuffer commandBuffer, VkQueryPool queryPool, uint32_t firstQuery, uint32_t queryCount, VkBuffer dstBuffer, VkDeviceSize dstOffset, VkDeviceSize stride, VkQueryResultFlags flags)
{
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1490
    layer_data *my_map_data = get_my_data_ptr(get_dispatch_key(commandBuffer), layer_data_map);
// STRUCT USES:['dstBuffer', 'queryPool']
    {
    std::lock_guard<std::mutex> lock(global_lock);
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1450
    dstBuffer = (VkBuffer)my_map_data->unique_id_mapping[reinterpret_cast<uint64_t &>(dstBuffer)];
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1450
    queryPool = (VkQueryPool)my_map_data->unique_id_mapping[reinterpret_cast<uint64_t &>(queryPool)];
    }
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1596
    get_dispatch_table(unique_objects_device_table_map, commandBuffer)->CmdCopyQueryPoolResults(commandBuffer, queryPool, firstQuery, queryCount, dstBuffer, dstOffset, stride, flags);
}


VK_LAYER_EXPORT VKAPI_ATTR void VKAPI_CALL vkCmdPushConstants(VkCommandBuffer commandBuffer, VkPipelineLayout layout, VkShaderStageFlags stageFlags, uint32_t offset, uint32_t size, const void* pValues)
{
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1490
    layer_data *my_map_data = get_my_data_ptr(get_dispatch_key(commandBuffer), layer_data_map);
// STRUCT USES:['layout']
    {
    std::lock_guard<std::mutex> lock(global_lock);
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1450
    layout = (VkPipelineLayout)my_map_data->unique_id_mapping[reinterpret_cast<uint64_t &>(layout)];
    }
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1596
    get_dispatch_table(unique_objects_device_table_map, commandBuffer)->CmdPushConstants(commandBuffer, layout, stageFlags, offset, size, pValues);
}


VK_LAYER_EXPORT VKAPI_ATTR void VKAPI_CALL vkCmdBeginRenderPass(VkCommandBuffer commandBuffer, const VkRenderPassBeginInfo* pRenderPassBegin, VkSubpassContents contents)
{
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1490
    layer_data *my_map_data = get_my_data_ptr(get_dispatch_key(commandBuffer), layer_data_map);
// STRUCT USES:['pRenderPassBegin']
//LOCAL DECLS:['pRenderPassBegin']
    safe_VkRenderPassBeginInfo* local_pRenderPassBegin = NULL;
    {
    std::lock_guard<std::mutex> lock(global_lock);
    if (pRenderPassBegin) {
        local_pRenderPassBegin = new safe_VkRenderPassBeginInfo(pRenderPassBegin);
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1450
        local_pRenderPassBegin->framebuffer = (VkFramebuffer)my_map_data->unique_id_mapping[reinterpret_cast<const uint64_t &>(pRenderPassBegin->framebuffer)];
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1450
        local_pRenderPassBegin->renderPass = (VkRenderPass)my_map_data->unique_id_mapping[reinterpret_cast<const uint64_t &>(pRenderPassBegin->renderPass)];
    }
    }
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1596
    get_dispatch_table(unique_objects_device_table_map, commandBuffer)->CmdBeginRenderPass(commandBuffer, (const VkRenderPassBeginInfo*)local_pRenderPassBegin, contents);
    if (local_pRenderPassBegin)
        delete local_pRenderPassBegin;
}


VK_LAYER_EXPORT VKAPI_ATTR void VKAPI_CALL vkDestroySurfaceKHR(VkInstance instance, VkSurfaceKHR surface, const VkAllocationCallbacks* pAllocator)
{
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1490
    layer_data *my_map_data = get_my_data_ptr(get_dispatch_key(instance), layer_data_map);
// STRUCT USES:['surface']
    std::unique_lock<std::mutex> lock(global_lock);
    uint64_t local_surface = reinterpret_cast<uint64_t &>(surface);
    surface = (VkSurfaceKHR)my_map_data->unique_id_mapping[local_surface];
    lock.unlock();
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1596
    get_dispatch_table(unique_objects_instance_table_map, instance)->DestroySurfaceKHR(instance, surface, pAllocator);
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1584
    lock.lock();
    my_map_data->unique_id_mapping.erase(local_surface);
}


VK_LAYER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkGetPhysicalDeviceSurfaceSupportKHR(VkPhysicalDevice physicalDevice, uint32_t queueFamilyIndex, VkSurfaceKHR surface, VkBool32* pSupported)
{
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1490
    layer_data *my_map_data = get_my_data_ptr(get_dispatch_key(physicalDevice), layer_data_map);
// STRUCT USES:['surface']
    {
    std::lock_guard<std::mutex> lock(global_lock);
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1450
    surface = (VkSurfaceKHR)my_map_data->unique_id_mapping[reinterpret_cast<uint64_t &>(surface)];
    }
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1596
    VkResult result = get_dispatch_table(unique_objects_instance_table_map, physicalDevice)->GetPhysicalDeviceSurfaceSupportKHR(physicalDevice, queueFamilyIndex, surface, pSupported);
    return result;
}


VK_LAYER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkGetPhysicalDeviceSurfaceCapabilitiesKHR(VkPhysicalDevice physicalDevice, VkSurfaceKHR surface, VkSurfaceCapabilitiesKHR* pSurfaceCapabilities)
{
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1490
    layer_data *my_map_data = get_my_data_ptr(get_dispatch_key(physicalDevice), layer_data_map);
// STRUCT USES:['surface']
    {
    std::lock_guard<std::mutex> lock(global_lock);
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1450
    surface = (VkSurfaceKHR)my_map_data->unique_id_mapping[reinterpret_cast<uint64_t &>(surface)];
    }
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1596
    VkResult result = get_dispatch_table(unique_objects_instance_table_map, physicalDevice)->GetPhysicalDeviceSurfaceCapabilitiesKHR(physicalDevice, surface, pSurfaceCapabilities);
    return result;
}


VK_LAYER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkGetPhysicalDeviceSurfaceFormatsKHR(VkPhysicalDevice physicalDevice, VkSurfaceKHR surface, uint32_t* pSurfaceFormatCount, VkSurfaceFormatKHR* pSurfaceFormats)
{
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1490
    layer_data *my_map_data = get_my_data_ptr(get_dispatch_key(physicalDevice), layer_data_map);
// STRUCT USES:['surface']
    {
    std::lock_guard<std::mutex> lock(global_lock);
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1450
    surface = (VkSurfaceKHR)my_map_data->unique_id_mapping[reinterpret_cast<uint64_t &>(surface)];
    }
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1596
    VkResult result = get_dispatch_table(unique_objects_instance_table_map, physicalDevice)->GetPhysicalDeviceSurfaceFormatsKHR(physicalDevice, surface, pSurfaceFormatCount, pSurfaceFormats);
    return result;
}


VK_LAYER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkGetPhysicalDeviceSurfacePresentModesKHR(VkPhysicalDevice physicalDevice, VkSurfaceKHR surface, uint32_t* pPresentModeCount, VkPresentModeKHR* pPresentModes)
{
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1490
    layer_data *my_map_data = get_my_data_ptr(get_dispatch_key(physicalDevice), layer_data_map);
// STRUCT USES:['surface']
    {
    std::lock_guard<std::mutex> lock(global_lock);
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1450
    surface = (VkSurfaceKHR)my_map_data->unique_id_mapping[reinterpret_cast<uint64_t &>(surface)];
    }
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1596
    VkResult result = get_dispatch_table(unique_objects_instance_table_map, physicalDevice)->GetPhysicalDeviceSurfacePresentModesKHR(physicalDevice, surface, pPresentModeCount, pPresentModes);
    return result;
}


VK_LAYER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkCreateSwapchainKHR(VkDevice device, const VkSwapchainCreateInfoKHR* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkSwapchainKHR* pSwapchain)
{
    return explicit_CreateSwapchainKHR(device, pCreateInfo, pAllocator, pSwapchain);
}

VK_LAYER_EXPORT VKAPI_ATTR void VKAPI_CALL vkDestroySwapchainKHR(VkDevice device, VkSwapchainKHR swapchain, const VkAllocationCallbacks* pAllocator)
{
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1490
    layer_data *my_map_data = get_my_data_ptr(get_dispatch_key(device), layer_data_map);
// STRUCT USES:['swapchain']
    std::unique_lock<std::mutex> lock(global_lock);
    uint64_t local_swapchain = reinterpret_cast<uint64_t &>(swapchain);
    swapchain = (VkSwapchainKHR)my_map_data->unique_id_mapping[local_swapchain];
    lock.unlock();
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1596
    get_dispatch_table(unique_objects_device_table_map, device)->DestroySwapchainKHR(device, swapchain, pAllocator);
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1584
    lock.lock();
    my_map_data->unique_id_mapping.erase(local_swapchain);
}


VK_LAYER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkGetSwapchainImagesKHR(VkDevice device, VkSwapchainKHR swapchain, uint32_t* pSwapchainImageCount, VkImage* pSwapchainImages)
{
    return explicit_GetSwapchainImagesKHR(device, swapchain, pSwapchainImageCount, pSwapchainImages);
}

VK_LAYER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkAcquireNextImageKHR(VkDevice device, VkSwapchainKHR swapchain, uint64_t timeout, VkSemaphore semaphore, VkFence fence, uint32_t* pImageIndex)
{
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1490
    layer_data *my_map_data = get_my_data_ptr(get_dispatch_key(device), layer_data_map);
// STRUCT USES:['fence', 'semaphore', 'swapchain']
    {
    std::lock_guard<std::mutex> lock(global_lock);
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1450
    fence = (VkFence)my_map_data->unique_id_mapping[reinterpret_cast<uint64_t &>(fence)];
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1450
    semaphore = (VkSemaphore)my_map_data->unique_id_mapping[reinterpret_cast<uint64_t &>(semaphore)];
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1450
    swapchain = (VkSwapchainKHR)my_map_data->unique_id_mapping[reinterpret_cast<uint64_t &>(swapchain)];
    }
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1596
    VkResult result = get_dispatch_table(unique_objects_device_table_map, device)->AcquireNextImageKHR(device, swapchain, timeout, semaphore, fence, pImageIndex);
    return result;
}


VK_LAYER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkQueuePresentKHR(VkQueue queue, const VkPresentInfoKHR* pPresentInfo)
{
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1490
    layer_data *my_map_data = get_my_data_ptr(get_dispatch_key(queue), layer_data_map);
// STRUCT USES:['pPresentInfo']
//LOCAL DECLS:['pPresentInfo']
    safe_VkPresentInfoKHR* local_pPresentInfo = NULL;
    {
    std::lock_guard<std::mutex> lock(global_lock);
    if (pPresentInfo) {
        local_pPresentInfo = new safe_VkPresentInfoKHR(pPresentInfo);
        if (local_pPresentInfo->pSwapchains) {
            for (uint32_t idx0=0; idx0<pPresentInfo->swapchainCount; ++idx0) {
                local_pPresentInfo->pSwapchains[idx0] = (VkSwapchainKHR)my_map_data->unique_id_mapping[reinterpret_cast<const uint64_t &>(pPresentInfo->pSwapchains[idx0])];
            }
        }
        if (local_pPresentInfo->pWaitSemaphores) {
            for (uint32_t idx1=0; idx1<pPresentInfo->waitSemaphoreCount; ++idx1) {
                local_pPresentInfo->pWaitSemaphores[idx1] = (VkSemaphore)my_map_data->unique_id_mapping[reinterpret_cast<const uint64_t &>(pPresentInfo->pWaitSemaphores[idx1])];
            }
        }
    }
    }
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1596
    VkResult result = get_dispatch_table(unique_objects_device_table_map, queue)->QueuePresentKHR(queue, (const VkPresentInfoKHR*)local_pPresentInfo);
    if (local_pPresentInfo)
        delete local_pPresentInfo;
    return result;
}


#ifdef VK_USE_PLATFORM_WIN32_KHR
VK_LAYER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkCreateWin32SurfaceKHR(VkInstance instance, const VkWin32SurfaceCreateInfoKHR* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkSurfaceKHR* pSurface)
{
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1490
    layer_data *my_map_data = get_my_data_ptr(get_dispatch_key(instance), layer_data_map);
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1596
    VkResult result = get_dispatch_table(unique_objects_instance_table_map, instance)->CreateWin32SurfaceKHR(instance, pCreateInfo, pAllocator, pSurface);
    if (VK_SUCCESS == result) {
        std::lock_guard<std::mutex> lock(global_lock);
// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #1567
        uint64_t unique_id = global_unique_id++;
        my_map_data->unique_id_mapping[unique_id] = reinterpret_cast<uint64_t &>(*pSurface);
        *pSurface = reinterpret_cast<VkSurfaceKHR&>(unique_id);
    }
    return result;
}
#endif


// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #466
static inline PFN_vkVoidFunction layer_intercept_proc(const char *name)
{
    if (!name || name[0] != 'v' || name[1] != 'k')
        return NULL;

    name += 2;
    if (!strcmp(name, "CreateInstance"))
        return (PFN_vkVoidFunction) vkCreateInstance;
    if (!strcmp(name, "DestroyInstance"))
        return (PFN_vkVoidFunction) vkDestroyInstance;
    if (!strcmp(name, "CreateDevice"))
        return (PFN_vkVoidFunction) vkCreateDevice;
    if (!strcmp(name, "DestroyDevice"))
        return (PFN_vkVoidFunction) vkDestroyDevice;
    if (!strcmp(name, "EnumerateInstanceExtensionProperties"))
        return (PFN_vkVoidFunction) vkEnumerateInstanceExtensionProperties;
    if (!strcmp(name, "EnumerateInstanceLayerProperties"))
        return (PFN_vkVoidFunction) vkEnumerateInstanceLayerProperties;
    if (!strcmp(name, "EnumerateDeviceLayerProperties"))
        return (PFN_vkVoidFunction) vkEnumerateDeviceLayerProperties;
    if (!strcmp(name, "QueueSubmit"))
        return (PFN_vkVoidFunction) vkQueueSubmit;
    if (!strcmp(name, "AllocateMemory"))
        return (PFN_vkVoidFunction) vkAllocateMemory;
    if (!strcmp(name, "FreeMemory"))
        return (PFN_vkVoidFunction) vkFreeMemory;
    if (!strcmp(name, "MapMemory"))
        return (PFN_vkVoidFunction) vkMapMemory;
    if (!strcmp(name, "UnmapMemory"))
        return (PFN_vkVoidFunction) vkUnmapMemory;
    if (!strcmp(name, "FlushMappedMemoryRanges"))
        return (PFN_vkVoidFunction) vkFlushMappedMemoryRanges;
    if (!strcmp(name, "InvalidateMappedMemoryRanges"))
        return (PFN_vkVoidFunction) vkInvalidateMappedMemoryRanges;
    if (!strcmp(name, "GetDeviceMemoryCommitment"))
        return (PFN_vkVoidFunction) vkGetDeviceMemoryCommitment;
    if (!strcmp(name, "BindBufferMemory"))
        return (PFN_vkVoidFunction) vkBindBufferMemory;
    if (!strcmp(name, "BindImageMemory"))
        return (PFN_vkVoidFunction) vkBindImageMemory;
    if (!strcmp(name, "GetBufferMemoryRequirements"))
        return (PFN_vkVoidFunction) vkGetBufferMemoryRequirements;
    if (!strcmp(name, "GetImageMemoryRequirements"))
        return (PFN_vkVoidFunction) vkGetImageMemoryRequirements;
    if (!strcmp(name, "GetImageSparseMemoryRequirements"))
        return (PFN_vkVoidFunction) vkGetImageSparseMemoryRequirements;
    if (!strcmp(name, "QueueBindSparse"))
        return (PFN_vkVoidFunction) vkQueueBindSparse;
    if (!strcmp(name, "CreateFence"))
        return (PFN_vkVoidFunction) vkCreateFence;
    if (!strcmp(name, "DestroyFence"))
        return (PFN_vkVoidFunction) vkDestroyFence;
    if (!strcmp(name, "ResetFences"))
        return (PFN_vkVoidFunction) vkResetFences;
    if (!strcmp(name, "GetFenceStatus"))
        return (PFN_vkVoidFunction) vkGetFenceStatus;
    if (!strcmp(name, "WaitForFences"))
        return (PFN_vkVoidFunction) vkWaitForFences;
    if (!strcmp(name, "CreateSemaphore"))
        return (PFN_vkVoidFunction) vkCreateSemaphore;
    if (!strcmp(name, "DestroySemaphore"))
        return (PFN_vkVoidFunction) vkDestroySemaphore;
    if (!strcmp(name, "CreateEvent"))
        return (PFN_vkVoidFunction) vkCreateEvent;
    if (!strcmp(name, "DestroyEvent"))
        return (PFN_vkVoidFunction) vkDestroyEvent;
    if (!strcmp(name, "GetEventStatus"))
        return (PFN_vkVoidFunction) vkGetEventStatus;
    if (!strcmp(name, "SetEvent"))
        return (PFN_vkVoidFunction) vkSetEvent;
    if (!strcmp(name, "ResetEvent"))
        return (PFN_vkVoidFunction) vkResetEvent;
    if (!strcmp(name, "CreateQueryPool"))
        return (PFN_vkVoidFunction) vkCreateQueryPool;
    if (!strcmp(name, "DestroyQueryPool"))
        return (PFN_vkVoidFunction) vkDestroyQueryPool;
    if (!strcmp(name, "GetQueryPoolResults"))
        return (PFN_vkVoidFunction) vkGetQueryPoolResults;
    if (!strcmp(name, "CreateBuffer"))
        return (PFN_vkVoidFunction) vkCreateBuffer;
    if (!strcmp(name, "DestroyBuffer"))
        return (PFN_vkVoidFunction) vkDestroyBuffer;
    if (!strcmp(name, "CreateBufferView"))
        return (PFN_vkVoidFunction) vkCreateBufferView;
    if (!strcmp(name, "DestroyBufferView"))
        return (PFN_vkVoidFunction) vkDestroyBufferView;
    if (!strcmp(name, "CreateImage"))
        return (PFN_vkVoidFunction) vkCreateImage;
    if (!strcmp(name, "DestroyImage"))
        return (PFN_vkVoidFunction) vkDestroyImage;
    if (!strcmp(name, "GetImageSubresourceLayout"))
        return (PFN_vkVoidFunction) vkGetImageSubresourceLayout;
    if (!strcmp(name, "CreateImageView"))
        return (PFN_vkVoidFunction) vkCreateImageView;
    if (!strcmp(name, "DestroyImageView"))
        return (PFN_vkVoidFunction) vkDestroyImageView;
    if (!strcmp(name, "CreateShaderModule"))
        return (PFN_vkVoidFunction) vkCreateShaderModule;
    if (!strcmp(name, "DestroyShaderModule"))
        return (PFN_vkVoidFunction) vkDestroyShaderModule;
    if (!strcmp(name, "CreatePipelineCache"))
        return (PFN_vkVoidFunction) vkCreatePipelineCache;
    if (!strcmp(name, "DestroyPipelineCache"))
        return (PFN_vkVoidFunction) vkDestroyPipelineCache;
    if (!strcmp(name, "GetPipelineCacheData"))
        return (PFN_vkVoidFunction) vkGetPipelineCacheData;
    if (!strcmp(name, "MergePipelineCaches"))
        return (PFN_vkVoidFunction) vkMergePipelineCaches;
    if (!strcmp(name, "CreateGraphicsPipelines"))
        return (PFN_vkVoidFunction) vkCreateGraphicsPipelines;
    if (!strcmp(name, "CreateComputePipelines"))
        return (PFN_vkVoidFunction) vkCreateComputePipelines;
    if (!strcmp(name, "DestroyPipeline"))
        return (PFN_vkVoidFunction) vkDestroyPipeline;
    if (!strcmp(name, "CreatePipelineLayout"))
        return (PFN_vkVoidFunction) vkCreatePipelineLayout;
    if (!strcmp(name, "DestroyPipelineLayout"))
        return (PFN_vkVoidFunction) vkDestroyPipelineLayout;
    if (!strcmp(name, "CreateSampler"))
        return (PFN_vkVoidFunction) vkCreateSampler;
    if (!strcmp(name, "DestroySampler"))
        return (PFN_vkVoidFunction) vkDestroySampler;
    if (!strcmp(name, "CreateDescriptorSetLayout"))
        return (PFN_vkVoidFunction) vkCreateDescriptorSetLayout;
    if (!strcmp(name, "DestroyDescriptorSetLayout"))
        return (PFN_vkVoidFunction) vkDestroyDescriptorSetLayout;
    if (!strcmp(name, "CreateDescriptorPool"))
        return (PFN_vkVoidFunction) vkCreateDescriptorPool;
    if (!strcmp(name, "DestroyDescriptorPool"))
        return (PFN_vkVoidFunction) vkDestroyDescriptorPool;
    if (!strcmp(name, "ResetDescriptorPool"))
        return (PFN_vkVoidFunction) vkResetDescriptorPool;
    if (!strcmp(name, "AllocateDescriptorSets"))
        return (PFN_vkVoidFunction) vkAllocateDescriptorSets;
    if (!strcmp(name, "FreeDescriptorSets"))
        return (PFN_vkVoidFunction) vkFreeDescriptorSets;
    if (!strcmp(name, "UpdateDescriptorSets"))
        return (PFN_vkVoidFunction) vkUpdateDescriptorSets;
    if (!strcmp(name, "CreateFramebuffer"))
        return (PFN_vkVoidFunction) vkCreateFramebuffer;
    if (!strcmp(name, "DestroyFramebuffer"))
        return (PFN_vkVoidFunction) vkDestroyFramebuffer;
    if (!strcmp(name, "CreateRenderPass"))
        return (PFN_vkVoidFunction) vkCreateRenderPass;
    if (!strcmp(name, "DestroyRenderPass"))
        return (PFN_vkVoidFunction) vkDestroyRenderPass;
    if (!strcmp(name, "GetRenderAreaGranularity"))
        return (PFN_vkVoidFunction) vkGetRenderAreaGranularity;
    if (!strcmp(name, "CreateCommandPool"))
        return (PFN_vkVoidFunction) vkCreateCommandPool;
    if (!strcmp(name, "DestroyCommandPool"))
        return (PFN_vkVoidFunction) vkDestroyCommandPool;
    if (!strcmp(name, "ResetCommandPool"))
        return (PFN_vkVoidFunction) vkResetCommandPool;
    if (!strcmp(name, "AllocateCommandBuffers"))
        return (PFN_vkVoidFunction) vkAllocateCommandBuffers;
    if (!strcmp(name, "FreeCommandBuffers"))
        return (PFN_vkVoidFunction) vkFreeCommandBuffers;
    if (!strcmp(name, "BeginCommandBuffer"))
        return (PFN_vkVoidFunction) vkBeginCommandBuffer;
    if (!strcmp(name, "CmdBindPipeline"))
        return (PFN_vkVoidFunction) vkCmdBindPipeline;
    if (!strcmp(name, "CmdBindDescriptorSets"))
        return (PFN_vkVoidFunction) vkCmdBindDescriptorSets;
    if (!strcmp(name, "CmdBindIndexBuffer"))
        return (PFN_vkVoidFunction) vkCmdBindIndexBuffer;
    if (!strcmp(name, "CmdBindVertexBuffers"))
        return (PFN_vkVoidFunction) vkCmdBindVertexBuffers;
    if (!strcmp(name, "CmdDrawIndirect"))
        return (PFN_vkVoidFunction) vkCmdDrawIndirect;
    if (!strcmp(name, "CmdDrawIndexedIndirect"))
        return (PFN_vkVoidFunction) vkCmdDrawIndexedIndirect;
    if (!strcmp(name, "CmdDispatchIndirect"))
        return (PFN_vkVoidFunction) vkCmdDispatchIndirect;
    if (!strcmp(name, "CmdCopyBuffer"))
        return (PFN_vkVoidFunction) vkCmdCopyBuffer;
    if (!strcmp(name, "CmdCopyImage"))
        return (PFN_vkVoidFunction) vkCmdCopyImage;
    if (!strcmp(name, "CmdBlitImage"))
        return (PFN_vkVoidFunction) vkCmdBlitImage;
    if (!strcmp(name, "CmdCopyBufferToImage"))
        return (PFN_vkVoidFunction) vkCmdCopyBufferToImage;
    if (!strcmp(name, "CmdCopyImageToBuffer"))
        return (PFN_vkVoidFunction) vkCmdCopyImageToBuffer;
    if (!strcmp(name, "CmdUpdateBuffer"))
        return (PFN_vkVoidFunction) vkCmdUpdateBuffer;
    if (!strcmp(name, "CmdFillBuffer"))
        return (PFN_vkVoidFunction) vkCmdFillBuffer;
    if (!strcmp(name, "CmdClearColorImage"))
        return (PFN_vkVoidFunction) vkCmdClearColorImage;
    if (!strcmp(name, "CmdClearDepthStencilImage"))
        return (PFN_vkVoidFunction) vkCmdClearDepthStencilImage;
    if (!strcmp(name, "CmdResolveImage"))
        return (PFN_vkVoidFunction) vkCmdResolveImage;
    if (!strcmp(name, "CmdSetEvent"))
        return (PFN_vkVoidFunction) vkCmdSetEvent;
    if (!strcmp(name, "CmdResetEvent"))
        return (PFN_vkVoidFunction) vkCmdResetEvent;
    if (!strcmp(name, "CmdWaitEvents"))
        return (PFN_vkVoidFunction) vkCmdWaitEvents;
    if (!strcmp(name, "CmdPipelineBarrier"))
        return (PFN_vkVoidFunction) vkCmdPipelineBarrier;
    if (!strcmp(name, "CmdBeginQuery"))
        return (PFN_vkVoidFunction) vkCmdBeginQuery;
    if (!strcmp(name, "CmdEndQuery"))
        return (PFN_vkVoidFunction) vkCmdEndQuery;
    if (!strcmp(name, "CmdResetQueryPool"))
        return (PFN_vkVoidFunction) vkCmdResetQueryPool;
    if (!strcmp(name, "CmdWriteTimestamp"))
        return (PFN_vkVoidFunction) vkCmdWriteTimestamp;
    if (!strcmp(name, "CmdCopyQueryPoolResults"))
        return (PFN_vkVoidFunction) vkCmdCopyQueryPoolResults;
    if (!strcmp(name, "CmdPushConstants"))
        return (PFN_vkVoidFunction) vkCmdPushConstants;
    if (!strcmp(name, "CmdBeginRenderPass"))
        return (PFN_vkVoidFunction) vkCmdBeginRenderPass;

    return NULL;
}
static inline PFN_vkVoidFunction layer_intercept_instance_proc(const char *name)
{
    if (!name || name[0] != 'v' || name[1] != 'k')
        return NULL;

    name += 2;
    if (!strcmp(name, "DestroyInstance"))
        return (PFN_vkVoidFunction) vkDestroyInstance;
    if (!strcmp(name, "EnumerateInstanceExtensionProperties"))
        return (PFN_vkVoidFunction) vkEnumerateInstanceExtensionProperties;
    if (!strcmp(name, "EnumerateInstanceLayerProperties"))
        return (PFN_vkVoidFunction) vkEnumerateInstanceLayerProperties;
    if (!strcmp(name, "EnumerateDeviceLayerProperties"))
        return (PFN_vkVoidFunction) vkEnumerateDeviceLayerProperties;

    return NULL;
}

VK_LAYER_EXPORT VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL vkGetDeviceProcAddr(VkDevice device, const char* funcName)
{
    PFN_vkVoidFunction addr;
    if (!strcmp("vkGetDeviceProcAddr", funcName)) {
        return (PFN_vkVoidFunction) vkGetDeviceProcAddr;
    }

    addr = layer_intercept_proc(funcName);
    if (addr)
        return addr;
    if (device == VK_NULL_HANDLE) {
        return NULL;
    }

// CODEGEN : file C:/releasebuild/LoaderAndValidationLayers/vk-layer-generate.py line #531
    layer_data *my_device_data = get_my_data_ptr(get_dispatch_key(device), layer_data_map);
    if (my_device_data->wsi_enabled) {
        if (!strcmp("vkCreateSwapchainKHR", funcName))
            return reinterpret_cast<PFN_vkVoidFunction>(vkCreateSwapchainKHR);
        if (!strcmp("vkDestroySwapchainKHR", funcName))
            return reinterpret_cast<PFN_vkVoidFunction>(vkDestroySwapchainKHR);
        if (!strcmp("vkGetSwapchainImagesKHR", funcName))
            return reinterpret_cast<PFN_vkVoidFunction>(vkGetSwapchainImagesKHR);
        if (!strcmp("vkAcquireNextImageKHR", funcName))
            return reinterpret_cast<PFN_vkVoidFunction>(vkAcquireNextImageKHR);
        if (!strcmp("vkQueuePresentKHR", funcName))
            return reinterpret_cast<PFN_vkVoidFunction>(vkQueuePresentKHR);
    }


    if (get_dispatch_table(unique_objects_device_table_map, device)->GetDeviceProcAddr == NULL)
        return NULL;
    return get_dispatch_table(unique_objects_device_table_map, device)->GetDeviceProcAddr(device, funcName);
}

VK_LAYER_EXPORT VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL vkGetInstanceProcAddr(VkInstance instance, const char* funcName)
{
    PFN_vkVoidFunction addr;
    if (!strcmp(funcName, "vkGetInstanceProcAddr"))
        return (PFN_vkVoidFunction) vkGetInstanceProcAddr;
    if (!strcmp(funcName, "vkCreateInstance"))
        return (PFN_vkVoidFunction) vkCreateInstance;
    if (!strcmp(funcName, "vkCreateDevice"))
        return (PFN_vkVoidFunction) vkCreateDevice;
    addr = layer_intercept_instance_proc(funcName);
    if (addr) {
        return addr;    }
    if (instance == VK_NULL_HANDLE) {
        return NULL;
    }

    VkLayerInstanceDispatchTable* pTable = get_dispatch_table(unique_objects_instance_table_map, instance);
    if (instanceExtMap.size() != 0 && instanceExtMap[pTable].wsi_enabled)
    {
        if (!strcmp("vkDestroySurfaceKHR", funcName))
            return reinterpret_cast<PFN_vkVoidFunction>(vkDestroySurfaceKHR);
        if (!strcmp("vkGetPhysicalDeviceSurfaceSupportKHR", funcName))
            return reinterpret_cast<PFN_vkVoidFunction>(vkGetPhysicalDeviceSurfaceSupportKHR);
        if (!strcmp("vkGetPhysicalDeviceSurfaceCapabilitiesKHR", funcName))
            return reinterpret_cast<PFN_vkVoidFunction>(vkGetPhysicalDeviceSurfaceCapabilitiesKHR);
        if (!strcmp("vkGetPhysicalDeviceSurfaceFormatsKHR", funcName))
            return reinterpret_cast<PFN_vkVoidFunction>(vkGetPhysicalDeviceSurfaceFormatsKHR);
        if (!strcmp("vkGetPhysicalDeviceSurfacePresentModesKHR", funcName))
            return reinterpret_cast<PFN_vkVoidFunction>(vkGetPhysicalDeviceSurfacePresentModesKHR);
#ifdef VK_USE_PLATFORM_WIN32_KHR
        if (!strcmp("vkCreateWin32SurfaceKHR", funcName))
            return reinterpret_cast<PFN_vkVoidFunction>(vkCreateWin32SurfaceKHR);
#endif  // VK_USE_PLATFORM_WIN32_KHR
    }

    if (get_dispatch_table(unique_objects_instance_table_map, instance)->GetInstanceProcAddr == NULL) {
        return NULL;
    }
    return get_dispatch_table(unique_objects_instance_table_map, instance)->GetInstanceProcAddr(instance, funcName);
}

