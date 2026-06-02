#include "filter_pipeline.h"
#include "../kernels/pipeline_kernels.h"
#include "../kernels/kernels.h"
#include <plog/Log.h>
#include <stdexcept>
#include <sstream>
#include <algorithm>

namespace cuda_filter
{

PipelineStage::PipelineStage(const std::string& name, FilterType type, int kSize, float intensity)
    : name(name), filterType(type), kernelSize(kSize), intensity(intensity), enabled(true)
{
    kernel = FilterUtils::createFilterKernel(type, kSize, intensity);
}

// PIMPL wrapper
struct FilterPipeline::GPUResources
{
    GPUPipelineResources res;
};

FilterPipeline::FilterPipeline()
    : m_resourcesAllocated(false), m_allocWidth(0), m_allocHeight(0), m_allocChannels(0),
      m_gpuResources(std::make_unique<GPUResources>())
{
}

FilterPipeline::~FilterPipeline()
{
    freeResources();
}

void FilterPipeline::addStage(const std::string& name, FilterType type, int kernelSize, float intensity)
{
    m_stages.emplace_back(name, type, kernelSize, intensity);
    m_resourcesAllocated = false; // Force reallocation if dimensions or limits change
}

void FilterPipeline::removeStage(size_t index)
{
    if (index >= m_stages.size())
    {
        throw std::out_of_range("Invalid stage index");
    }
    m_stages.erase(m_stages.begin() + index);
}

void FilterPipeline::removeStageByName(const std::string& name)
{
    for (auto it = m_stages.begin(); it != m_stages.end(); ++it)
    {
        if (it->name == name)
        {
            m_stages.erase(it);
            return;
        }
    }
    PLOG_WARNING << "Stage not found: " << name;
}

void FilterPipeline::clearStages()
{
    m_stages.clear();
}

void FilterPipeline::enableStage(size_t index, bool enabled)
{
    if (index < m_stages.size())
    {
        m_stages[index].enabled = enabled;
    }
}

void FilterPipeline::moveStage(size_t from, size_t to)
{
    if (from >= m_stages.size() || to >= m_stages.size()) return;
    auto stage = m_stages[from];
    m_stages.erase(m_stages.begin() + from);
    m_stages.insert(m_stages.begin() + to, stage);
}

void FilterPipeline::executeGPU(const cv::Mat& input, cv::Mat& output)
{
    if (m_stages.empty())
    {
        input.copyTo(output);
        return;
    }

    if (!m_resourcesAllocated || input.cols != m_allocWidth || input.rows != m_allocHeight || input.channels() != m_allocChannels)
    {
        allocateResources(input.cols, input.rows, input.channels());
    }

    output.create(input.size(), input.type());

    std::vector<PipelineStageInfo> stageInfos;
    int activeStages = 0;
    
    // Upload kernels for enabled stages
    for (const auto& stage : m_stages)
    {
        if (!stage.enabled) continue;
        
        if (!m_gpuResources->res.d_kernels[activeStages])
        {
            cudaMalloc(&m_gpuResources->res.d_kernels[activeStages], stage.kernel.total() * stage.kernel.elemSize());
        }
        cudaMemcpy(m_gpuResources->res.d_kernels[activeStages], stage.kernel.data, stage.kernel.total() * stage.kernel.elemSize(), cudaMemcpyHostToDevice);

        PipelineStageInfo info;
        info.d_kernel = m_gpuResources->res.d_kernels[activeStages];
        info.kernelSize = stage.kernelSize;
        stageInfos.push_back(info);
        activeStages++;
    }

    if (stageInfos.empty())
    {
        input.copyTo(output);
        return;
    }

    const size_t imageBytes = static_cast<size_t>(m_allocWidth) * m_allocHeight * m_allocChannels;
    cudaMemcpy(m_gpuResources->res.d_input, input.data, imageBytes, cudaMemcpyHostToDevice);

    cuda_filter::executePipelineGPU(m_gpuResources->res.d_input, m_gpuResources->res.d_output, 
        m_allocWidth, m_allocHeight, m_allocChannels, stageInfos, 0);

    cudaMemcpy(output.data, m_gpuResources->res.d_output, imageBytes, cudaMemcpyDeviceToHost);
}

void FilterPipeline::executeGPUMultiStream(const cv::Mat& input, cv::Mat& output)
{
    if (m_stages.empty())
    {
        input.copyTo(output);
        return;
    }

    if (!m_resourcesAllocated || input.cols != m_allocWidth || input.rows != m_allocHeight || input.channels() != m_allocChannels)
    {
        allocateResources(input.cols, input.rows, input.channels());
    }

    output.create(input.size(), input.type());

    std::vector<PipelineStageInfo> stageInfos;
    int activeStages = 0;
    
    // Upload kernels
    for (const auto& stage : m_stages)
    {
        if (!stage.enabled) continue;
        
        if (!m_gpuResources->res.d_kernels[activeStages])
        {
            cudaMalloc(&m_gpuResources->res.d_kernels[activeStages], stage.kernel.total() * stage.kernel.elemSize());
        }
        // Asynchronous copy on the respective stream
        cudaMemcpyAsync(m_gpuResources->res.d_kernels[activeStages], stage.kernel.data, 
            stage.kernel.total() * stage.kernel.elemSize(), cudaMemcpyHostToDevice, 
            m_gpuResources->res.getStream(activeStages));

        PipelineStageInfo info;
        info.d_kernel = m_gpuResources->res.d_kernels[activeStages];
        info.kernelSize = stage.kernelSize;
        stageInfos.push_back(info);
        activeStages++;
    }

    if (stageInfos.empty())
    {
        input.copyTo(output);
        return;
    }

    const size_t imageBytes = static_cast<size_t>(m_allocWidth) * m_allocHeight * m_allocChannels;
    cudaMemcpyAsync(m_gpuResources->res.d_input, input.data, imageBytes, cudaMemcpyHostToDevice, m_gpuResources->res.getStream(0));

    cuda_filter::executePipelineMultiStreamGPU(m_gpuResources->res.d_input, m_gpuResources->res.d_output, 
        m_allocWidth, m_allocHeight, m_allocChannels, stageInfos);

    // Final copy back
    cudaMemcpy(output.data, m_gpuResources->res.d_output, imageBytes, cudaMemcpyDeviceToHost);
}

void FilterPipeline::executeCPU(const cv::Mat& input, cv::Mat& output)
{
    if (m_stages.empty())
    {
        input.copyTo(output);
        return;
    }
    
    cv::Mat current = input.clone();
    cv::Mat temp;
    bool hasActiveStages = false;

    for (const auto& stage : m_stages)
    {
        if (!stage.enabled) continue;
        hasActiveStages = true;
        cuda_filter::applyFilterCPU(current, temp, stage.kernel);
        current = temp.clone();
    }
    
    if (hasActiveStages)
    {
        output = current;
    }
    else
    {
        input.copyTo(output);
    }
}

size_t FilterPipeline::getStageCount() const { return m_stages.size(); }
const PipelineStage& FilterPipeline::getStage(size_t index) const { return m_stages.at(index); }
std::vector<PipelineStage>& FilterPipeline::getStages() { return m_stages; }

std::string FilterPipeline::getPipelineDescription() const
{
    std::stringstream ss;
    ss << "Pipeline [" << m_stages.size() << " stages]: ";
    for (size_t i = 0; i < m_stages.size(); ++i)
    {
        if (i > 0) ss << " -> ";
        ss << m_stages[i].name << (m_stages[i].enabled ? "" : " (disabled)");
    }
    return ss.str();
}

void FilterPipeline::allocateResources(int width, int height, int channels)
{
    int maxStages = std::max(static_cast<int>(m_stages.size()), 10); // Reserve slots
    m_gpuResources->res.allocate(width, height, channels, maxStages);
    m_allocWidth = width;
    m_allocHeight = height;
    m_allocChannels = channels;
    m_resourcesAllocated = true;
}

void FilterPipeline::freeResources()
{
    m_gpuResources->res.free();
    m_resourcesAllocated = false;
}

bool FilterPipeline::resourcesAllocated() const
{
    return m_resourcesAllocated;
}

} // namespace cuda_filter
