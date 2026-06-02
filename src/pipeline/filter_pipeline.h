#pragma once

#include <vector>
#include <string>
#include <memory>
#include <functional>
#include <opencv2/opencv.hpp>
#include "../utils/filter_utils.h"

namespace cuda_filter
{

    /// Describes a single filter stage within the pipeline.
    /// The convolution kernel is pre-computed at construction time so that
    /// it can be uploaded to the GPU once rather than recomputed per-frame.
    struct PipelineStage
    {
        std::string name;
        FilterType filterType;
        int kernelSize;
        float intensity;
        cv::Mat kernel; ///< Pre-computed convolution kernel (CV_32F)
        bool enabled;

        PipelineStage(const std::string& name, FilterType type, int kSize, float intensity = 1.0f);
    };

    /// Manages an ordered chain of image filters and their GPU resources.
    ///
    /// Typical usage:
    /// @code
    ///   FilterPipeline pipe;
    ///   pipe.addStage("blur",  FilterType::BLUR,   5);
    ///   pipe.addStage("sharp", FilterType::SHARPEN, 3, 1.5f);
    ///   pipe.allocateResources(frame.cols, frame.rows, frame.channels());
    ///   pipe.executeGPU(frame, output);
    /// @endcode
    class FilterPipeline
    {
    public:
        FilterPipeline();
        ~FilterPipeline();

        // -- Pipeline construction -------------------------------------------

        /// Append a new stage to the end of the pipeline.
        void addStage(const std::string& name, FilterType type, int kernelSize, float intensity = 1.0f);

        /// Remove the stage at @p index. Throws std::out_of_range if invalid.
        void removeStage(size_t index);

        /// Remove the first stage whose name matches @p name.
        /// Does nothing (with a warning log) if no match is found.
        void removeStageByName(const std::string& name);

        /// Remove all stages.
        void clearStages();

        /// Enable or disable the stage at @p index without removing it.
        void enableStage(size_t index, bool enabled);

        /// Reorder: move the stage at @p from to position @p to.
        void moveStage(size_t from, size_t to);

        // -- Pipeline execution -----------------------------------------------

        /// Execute all enabled stages on the GPU using a single CUDA stream.
        /// Ping-pong buffers are used so intermediate copies are avoided.
        void executeGPU(const cv::Mat& input, cv::Mat& output);

        /// Execute all enabled stages on the GPU using multiple CUDA streams
        /// for overlapped upload/compute where possible.
        void executeGPUMultiStream(const cv::Mat& input, cv::Mat& output);

        /// Execute all enabled stages on the CPU (sequential cv::filter2D).
        void executeCPU(const cv::Mat& input, cv::Mat& output);

        // -- Pipeline info ----------------------------------------------------

        size_t getStageCount() const;
        const PipelineStage& getStage(size_t index) const;
        std::vector<PipelineStage>& getStages();
        std::string getPipelineDescription() const;

        // -- Resource management ----------------------------------------------

        /// Pre-allocate GPU buffers for the given frame dimensions.
        /// Must be called (or re-called) whenever the frame size changes.
        void allocateResources(int width, int height, int channels);

        /// Release all GPU resources.
        void freeResources();

        /// @return true if allocateResources has been called and not yet freed.
        bool resourcesAllocated() const;

    private:
        std::vector<PipelineStage> m_stages;
        bool m_resourcesAllocated;
        int m_allocWidth, m_allocHeight, m_allocChannels;

        // Forward-declared so the header doesn't pull in cuda_runtime.h.
        struct GPUResources;
        std::unique_ptr<GPUResources> m_gpuResources;
    };

} // namespace cuda_filter
