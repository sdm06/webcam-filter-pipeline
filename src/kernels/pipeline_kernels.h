#pragma once

#include <opencv2/opencv.hpp>
#include <vector>

// Forward-declare CUDA types so this header is includable from .cpp files.
// cudaStream_t is actually a pointer typedef; we can include the CUDA runtime
// header when compiling with nvcc, and provide an opaque typedef otherwise.
#ifdef __CUDACC__
#include <cuda_runtime.h>
#else
// Minimal forward declarations so pure-C++ TUs can see the API signatures.
// cudaStream_t is defined as CUstream_st* in the CUDA headers.
struct CUstream_st;
using cudaStream_t = CUstream_st *;
#endif

namespace cuda_filter
{

    // -------------------------------------------------------------------------
    // Pipeline stage descriptor – one convolution pass.
    // -------------------------------------------------------------------------
    struct PipelineStageInfo
    {
        float *d_kernel;  ///< Device pointer to the convolution kernel weights
        int kernelSize;   ///< Side length of the square kernel (e.g. 3, 5, 7)
    };

    // -------------------------------------------------------------------------
    // RAII wrapper for pre-allocated GPU resources used by the pipeline.
    //
    // Call allocate() once (or whenever the frame geometry changes) and reuse
    // the buffers across frames to avoid per-frame cudaMalloc/cudaFree overhead.
    // -------------------------------------------------------------------------
    class GPUPipelineResources
    {
    public:
        GPUPipelineResources() = default;
        ~GPUPipelineResources();

        // Non-copyable, movable.
        GPUPipelineResources(const GPUPipelineResources &) = delete;
        GPUPipelineResources &operator=(const GPUPipelineResources &) = delete;
        GPUPipelineResources(GPUPipelineResources &&other) noexcept;
        GPUPipelineResources &operator=(GPUPipelineResources &&other) noexcept;

        /// Allocate (or re-allocate) all device memory and streams.
        /// @param width       Frame width in pixels.
        /// @param height      Frame height in pixels.
        /// @param channels    Number of colour channels (typically 3).
        /// @param maxStages   Maximum number of pipeline stages to support.
        void allocate(int width, int height, int channels, int maxStages);

        /// Release all device memory and destroy streams.
        void free();

        /// @return true if allocate() has been called and free() has not.
        [[nodiscard]] bool isAllocated() const noexcept { return allocated_; }

        /// @return the CUDA stream for pipeline stage @p idx.
        [[nodiscard]] cudaStream_t getStream(int idx) const;

        // --- public device pointers (used by pipeline launcher) ---
        unsigned char *d_input = nullptr;
        unsigned char *d_output = nullptr;
        unsigned char *d_pingpong[2] = {nullptr, nullptr};

        /// Per-stage device kernel pointers (populated by the caller).
        std::vector<float *> d_kernels;

        /// Per-stage CUDA streams.
        std::vector<cudaStream_t> streams;

    private:
        bool allocated_ = false;
        int width_ = 0;
        int height_ = 0;
        int channels_ = 0;
        int maxStages_ = 0;
    };

    // -------------------------------------------------------------------------
    // Convolution launch wrapper
    // -------------------------------------------------------------------------

    /// Launch the convolutionKernel on an explicit CUDA stream.
    /// This is a thin wrapper around the __global__ kernel defined in
    /// convolution_kernels.cu.
    void launchConvolution(const unsigned char *d_input,
                           unsigned char *d_output,
                           const float *d_kernel,
                           int width, int height, int channels,
                           int kernelSize,
                           cudaStream_t stream);

    // -------------------------------------------------------------------------
    // Pipeline execution
    // -------------------------------------------------------------------------

    /// Execute a chain of convolution stages sequentially on a single stream,
    /// using ping-pong intermediate buffers.
    void executePipelineGPU(const unsigned char *d_input,
                            unsigned char *d_output,
                            int width, int height, int channels,
                            const std::vector<PipelineStageInfo> &stages,
                            cudaStream_t stream);

    /// Execute a chain of convolution stages using one stream per stage.
    /// H2D / D2H transfers may overlap with kernel execution on different
    /// streams, but inter-stage dependencies are honoured via events.
    void executePipelineMultiStreamGPU(const unsigned char *d_input,
                                      unsigned char *d_output,
                                      int width, int height, int channels,
                                      const std::vector<PipelineStageInfo> &stages);

    // -------------------------------------------------------------------------
    // Wipe transition
    // -------------------------------------------------------------------------

    /// Host-side convenience: upload two cv::Mat frames, apply the wipe
    /// transition on the GPU, and download the result.
    /// @param inputA   First pipeline result (visible when progress == 0).
    /// @param inputB   Second pipeline result (fully visible when progress == 1).
    /// @param output   Output image (same size / type as inputs).
    /// @param progress Transition progress in [0.0, 1.0].
    void applyWipeTransition(const cv::Mat &inputA,
                             const cv::Mat &inputB,
                             cv::Mat &output,
                             float progress);

} // namespace cuda_filter
