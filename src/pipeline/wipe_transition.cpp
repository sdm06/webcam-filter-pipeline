#include "wipe_transition.h"
#include "../kernels/pipeline_kernels.h"
#include <algorithm>

namespace cuda_filter
{

WipeTransition::WipeTransition()
    : m_duration(1.0f), m_active(false)
{
}

void WipeTransition::setDuration(float seconds)
{
    m_duration = seconds > 0.0f ? seconds : 0.1f;
}

float WipeTransition::getDuration() const
{
    return m_duration;
}

void WipeTransition::start()
{
    m_startTime = std::chrono::steady_clock::now();
    m_active = true;
}

void WipeTransition::reset()
{
    m_active = false;
}

bool WipeTransition::isActive() const
{
    if (!m_active) return false;
    if (getProgress() >= 1.0f)
    {
        // Auto-deactivate once complete
        return false;
    }
    return true;
}

float WipeTransition::getProgress() const
{
    if (!m_active) return 0.0f;

    auto now = std::chrono::steady_clock::now();
    std::chrono::duration<float> elapsed = now - m_startTime;
    float progress = elapsed.count() / m_duration;
    
    return std::clamp(progress, 0.0f, 1.0f);
}

void WipeTransition::apply(const cv::Mat& input, cv::Mat& output,
                           FilterPipeline& pipelineA, FilterPipeline& pipelineB)
{
    if (!isActive())
    {
        // If not active, just run pipeline B (or A depending on semantics, but usually we transition A->B)
        // Let's assume we output pipeline B when finished, or pipeline A if progress == 0
        if (getProgress() >= 1.0f) {
            pipelineB.executeGPU(input, output);
        } else {
            pipelineA.executeGPU(input, output);
        }
        return;
    }

    cv::Mat outputA, outputB;
    
    // Execute both pipelines
    pipelineA.executeGPU(input, outputA);
    pipelineB.executeGPU(input, outputB);

    // Apply wipe transition kernel
    applyWipeTransition(outputA, outputB, output, getProgress());
    
    if (getProgress() >= 1.0f)
    {
        m_active = false;
    }
}

} // namespace cuda_filter
