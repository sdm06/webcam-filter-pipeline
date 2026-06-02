// =============================================================================
// pipeline_kernels.cu – GPU filter-pipeline infrastructure
//
// Provides:
//   • wipeTransitionKernel   – left-to-right wipe between two frames
//   • launchConvolution      – stream-aware wrapper around convolutionKernel
//   • executePipelineGPU     – single-stream sequential pipeline (ping-pong)
//   • executePipelineMultiStreamGPU – multi-stream pipeline with event sync
//   • applyWipeTransition    – host-level cv::Mat helper
//   • GPUPipelineResources   – RAII device-memory manager
// =============================================================================

#include "pipeline_kernels.h"
#include "kernels.h"

#include <cuda_runtime.h>
#include <plog/Log.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <utility>

namespace cuda_filter
{

// ── CUDA error-checking macro (same pattern as convolution_kernels.cu) ──────
#define CHECK_CUDA_ERROR(call)                                                          \
    {                                                                                   \
        cudaError_t err = call;                                                         \
        if (err != cudaSuccess)                                                         \
        {                                                                               \
            PLOG_ERROR << "CUDA error in " << #call << ": " << cudaGetErrorString(err); \
            return;                                                                     \
        }                                                                               \
    }

// ── Forward-declare the kernel from convolution_kernels.cu ──────────────────
// We cannot #include a __global__ function across TUs, so we re-declare it
// with the exact same signature.  Both TUs live in the same namespace so
// the linker resolves them without issues.
extern __global__ void convolutionKernel(const unsigned char *input,
                                         unsigned char *output,
                                         const float *kernel,
                                         int width, int height,
                                         int channels, int kernelSize);

// =============================================================================
// 1. Wipe-transition kernel
// =============================================================================

/// Left-to-right wipe between two images with a smooth blend zone.
///
/// For each pixel (x, y):
///   • If x is well to the left of the wipe edge  → use inputB.
///   • If x is well to the right                   → use inputA.
///   • Within ±BLEND_HALF_WIDTH of the boundary    → linear blend.
///
/// @param progress  Normalised transition in [0, 1].
///                  0 = show inputA entirely, 1 = show inputB entirely.
__global__ void wipeTransitionKernel(const unsigned char *inputA,
                                     const unsigned char *inputB,
                                     unsigned char *output,
                                     int width, int height, int channels,
                                     float progress)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    // Blend zone half-width in pixels.
    constexpr float BLEND_HALF_WIDTH = 5.0f;

    // Wipe boundary position (in pixel coordinates).
    float boundary = progress * static_cast<float>(width);

    // Signed distance from the boundary (negative = left of boundary).
    float dist = static_cast<float>(x) - boundary;

    // Compute blend factor: 0 → fully inputB, 1 → fully inputA.
    float alpha;
    if (dist <= -BLEND_HALF_WIDTH)
    {
        alpha = 0.0f; // fully inputB
    }
    else if (dist >= BLEND_HALF_WIDTH)
    {
        alpha = 1.0f; // fully inputA
    }
    else
    {
        // Linear ramp inside the blend zone.
        alpha = (dist + BLEND_HALF_WIDTH) / (2.0f * BLEND_HALF_WIDTH);
    }

    int idx = (y * width + x) * channels;
    for (int c = 0; c < channels; ++c)
    {
        float valA = static_cast<float>(inputA[idx + c]);
        float valB = static_cast<float>(inputB[idx + c]);
        float blended = valA * alpha + valB * (1.0f - alpha);
        output[idx + c] = static_cast<unsigned char>(fminf(fmaxf(blended, 0.0f), 255.0f));
    }
}

// =============================================================================
// 2. launchConvolution – stream-aware wrapper
// =============================================================================

void launchConvolution(const unsigned char *d_input,
                       unsigned char *d_output,
                       const float *d_kernel,
                       int width, int height, int channels,
                       int kernelSize,
                       cudaStream_t stream)
{
    dim3 block(16, 16);
    dim3 grid(cuda::divUp(width, static_cast<int>(block.x)),
              cuda::divUp(height, static_cast<int>(block.y)));

    convolutionKernel<<<grid, block, 0, stream>>>(d_input, d_output, d_kernel,
                                                  width, height, channels, kernelSize);

    CHECK_CUDA_ERROR(cudaGetLastError());
}

// =============================================================================
// 3. Single-stream sequential pipeline (ping-pong buffers)
// =============================================================================

void executePipelineGPU(const unsigned char *d_input,
                        unsigned char *d_output,
                        int width, int height, int channels,
                        const std::vector<PipelineStageInfo> &stages,
                        cudaStream_t stream)
{
    if (stages.empty())
    {
        PLOG_WARNING << "executePipelineGPU called with empty stage list";
        return;
    }

    const size_t imageBytes = static_cast<size_t>(width) * height * channels;

    // Allocate two intermediate ping-pong buffers.
    unsigned char *pp[2] = {nullptr, nullptr};
    CHECK_CUDA_ERROR(cudaMalloc(&pp[0], imageBytes));
    CHECK_CUDA_ERROR(cudaMalloc(&pp[1], imageBytes));

    // First stage reads from d_input.
    const unsigned char *src = d_input;

    for (size_t i = 0; i < stages.size(); ++i)
    {
        // Last stage writes directly to d_output; otherwise ping-pong.
        unsigned char *dst = (i == stages.size() - 1) ? d_output : pp[i & 1];

        launchConvolution(src, dst, stages[i].d_kernel,
                          width, height, channels,
                          stages[i].kernelSize, stream);

        src = dst;
    }

    // Synchronise the stream so the caller can safely read d_output.
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));

    cudaFree(pp[0]);
    cudaFree(pp[1]);
}

// =============================================================================
// 4. Multi-stream pipeline (one stream per stage, event-synchronised)
// =============================================================================

void executePipelineMultiStreamGPU(const unsigned char *d_input,
                                   unsigned char *d_output,
                                   int width, int height, int channels,
                                   const std::vector<PipelineStageInfo> &stages)
{
    if (stages.empty())
    {
        PLOG_WARNING << "executePipelineMultiStreamGPU called with empty stage list";
        return;
    }

    const size_t numStages = stages.size();
    const size_t imageBytes = static_cast<size_t>(width) * height * channels;

    // --- Create one stream + one event per stage ---
    std::vector<cudaStream_t> stageStreams(numStages);
    std::vector<cudaEvent_t> stageEvents(numStages);

    for (size_t i = 0; i < numStages; ++i)
    {
        CHECK_CUDA_ERROR(cudaStreamCreate(&stageStreams[i]));
        CHECK_CUDA_ERROR(cudaEventCreate(&stageEvents[i]));
    }

    // --- Ping-pong buffers ---
    unsigned char *pp[2] = {nullptr, nullptr};
    CHECK_CUDA_ERROR(cudaMalloc(&pp[0], imageBytes));
    CHECK_CUDA_ERROR(cudaMalloc(&pp[1], imageBytes));

    const unsigned char *src = d_input;

    for (size_t i = 0; i < numStages; ++i)
    {
        unsigned char *dst = (i == numStages - 1) ? d_output : pp[i & 1];

        // If there is a preceding stage, wait for it to finish before reading
        // its output as our input.
        if (i > 0)
        {
            CHECK_CUDA_ERROR(cudaStreamWaitEvent(stageStreams[i], stageEvents[i - 1], 0));
        }

        launchConvolution(src, dst, stages[i].d_kernel,
                          width, height, channels,
                          stages[i].kernelSize, stageStreams[i]);

        // Record an event so the *next* stage can depend on this one.
        CHECK_CUDA_ERROR(cudaEventRecord(stageEvents[i], stageStreams[i]));

        src = dst;
    }

    // Wait for the final stage to finish.
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stageStreams[numStages - 1]));

    // --- Cleanup ---
    for (size_t i = 0; i < numStages; ++i)
    {
        cudaEventDestroy(stageEvents[i]);
        cudaStreamDestroy(stageStreams[i]);
    }
    cudaFree(pp[0]);
    cudaFree(pp[1]);
}

// =============================================================================
// 5. Host-level wipe-transition helper (cv::Mat → GPU → cv::Mat)
// =============================================================================

void applyWipeTransition(const cv::Mat &inputA,
                         const cv::Mat &inputB,
                         cv::Mat &output,
                         float progress)
{
    if (inputA.empty() || inputB.empty())
    {
        PLOG_ERROR << "applyWipeTransition: one or both inputs are empty";
        return;
    }
    if (inputA.size() != inputB.size() || inputA.type() != inputB.type())
    {
        PLOG_ERROR << "applyWipeTransition: input size/type mismatch";
        return;
    }

    progress = std::clamp(progress, 0.0f, 1.0f);

    output.create(inputA.size(), inputA.type());

    const int width = inputA.cols;
    const int height = inputA.rows;
    const int channels = inputA.channels();
    const size_t imageBytes = static_cast<size_t>(width) * height * channels;

    unsigned char *d_a = nullptr;
    unsigned char *d_b = nullptr;
    unsigned char *d_out = nullptr;

    CHECK_CUDA_ERROR(cudaMalloc(&d_a, imageBytes));
    CHECK_CUDA_ERROR(cudaMalloc(&d_b, imageBytes));
    CHECK_CUDA_ERROR(cudaMalloc(&d_out, imageBytes));

    CHECK_CUDA_ERROR(cudaMemcpy(d_a, inputA.data, imageBytes, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_b, inputB.data, imageBytes, cudaMemcpyHostToDevice));

    dim3 block(16, 16);
    dim3 grid(cuda::divUp(width, static_cast<int>(block.x)),
              cuda::divUp(height, static_cast<int>(block.y)));

    wipeTransitionKernel<<<grid, block>>>(d_a, d_b, d_out,
                                          width, height, channels, progress);

    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    CHECK_CUDA_ERROR(cudaMemcpy(output.data, d_out, imageBytes, cudaMemcpyDeviceToHost));

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_out);
}

// =============================================================================
// 6. GPUPipelineResources – RAII device-memory manager
// =============================================================================

GPUPipelineResources::~GPUPipelineResources()
{
    free();
}

GPUPipelineResources::GPUPipelineResources(GPUPipelineResources &&other) noexcept
    : d_input(other.d_input),
      d_output(other.d_output),
      d_kernels(std::move(other.d_kernels)),
      streams(std::move(other.streams)),
      allocated_(other.allocated_),
      width_(other.width_),
      height_(other.height_),
      channels_(other.channels_),
      maxStages_(other.maxStages_)
{
    d_pingpong[0] = other.d_pingpong[0];
    d_pingpong[1] = other.d_pingpong[1];

    // Null-out the source so its destructor is a no-op.
    other.d_input = nullptr;
    other.d_output = nullptr;
    other.d_pingpong[0] = nullptr;
    other.d_pingpong[1] = nullptr;
    other.allocated_ = false;
}

GPUPipelineResources &GPUPipelineResources::operator=(GPUPipelineResources &&other) noexcept
{
    if (this != &other)
    {
        free();

        d_input = other.d_input;
        d_output = other.d_output;
        d_pingpong[0] = other.d_pingpong[0];
        d_pingpong[1] = other.d_pingpong[1];
        d_kernels = std::move(other.d_kernels);
        streams = std::move(other.streams);
        allocated_ = other.allocated_;
        width_ = other.width_;
        height_ = other.height_;
        channels_ = other.channels_;
        maxStages_ = other.maxStages_;

        other.d_input = nullptr;
        other.d_output = nullptr;
        other.d_pingpong[0] = nullptr;
        other.d_pingpong[1] = nullptr;
        other.allocated_ = false;
    }
    return *this;
}

void GPUPipelineResources::allocate(int width, int height, int channels, int maxStages)
{
    // Re-allocate only if the geometry or stage count changed.
    if (allocated_ && width == width_ && height == height_ &&
        channels == channels_ && maxStages == maxStages_)
    {
        return;
    }

    // Release any previous allocation.
    free();

    width_ = width;
    height_ = height;
    channels_ = channels;
    maxStages_ = maxStages;

    const size_t imageBytes = static_cast<size_t>(width) * height * channels;

    cudaError_t err;

    err = cudaMalloc(&d_input, imageBytes);
    if (err != cudaSuccess)
    {
        PLOG_ERROR << "cudaMalloc d_input failed: " << cudaGetErrorString(err);
        return;
    }

    err = cudaMalloc(&d_output, imageBytes);
    if (err != cudaSuccess)
    {
        PLOG_ERROR << "cudaMalloc d_output failed: " << cudaGetErrorString(err);
        free();
        return;
    }

    for (int i = 0; i < 2; ++i)
    {
        err = cudaMalloc(&d_pingpong[i], imageBytes);
        if (err != cudaSuccess)
        {
            PLOG_ERROR << "cudaMalloc d_pingpong[" << i << "] failed: " << cudaGetErrorString(err);
            free();
            return;
        }
    }

    // Pre-allocate device memory for each kernel (caller fills them later).
    d_kernels.resize(maxStages, nullptr);

    // Create one stream per stage.
    streams.resize(maxStages, nullptr);
    for (int i = 0; i < maxStages; ++i)
    {
        err = cudaStreamCreate(&streams[i]);
        if (err != cudaSuccess)
        {
            PLOG_ERROR << "cudaStreamCreate failed for stage " << i << ": " << cudaGetErrorString(err);
            free();
            return;
        }
    }

    allocated_ = true;
    PLOG_INFO << "GPUPipelineResources allocated: " << width << "x" << height
              << "x" << channels << ", " << maxStages << " stages";
}

void GPUPipelineResources::free()
{
    if (!allocated_)
        return;

    if (d_input)  { cudaFree(d_input);  d_input = nullptr; }
    if (d_output) { cudaFree(d_output); d_output = nullptr; }

    for (int i = 0; i < 2; ++i)
    {
        if (d_pingpong[i]) { cudaFree(d_pingpong[i]); d_pingpong[i] = nullptr; }
    }

    for (auto *dk : d_kernels)
    {
        if (dk) cudaFree(dk);
    }
    d_kernels.clear();

    for (auto &s : streams)
    {
        if (s) cudaStreamDestroy(s);
    }
    streams.clear();

    allocated_ = false;
    PLOG_INFO << "GPUPipelineResources freed";
}

cudaStream_t GPUPipelineResources::getStream(int idx) const
{
    assert(allocated_ && "GPUPipelineResources not allocated");
    assert(idx >= 0 && idx < static_cast<int>(streams.size()));
    return streams[idx];
}

} // namespace cuda_filter
